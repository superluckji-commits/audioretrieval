import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from hear21passt.base import AugmentMelSTFT, PasstBasicWrapper
from hear21passt.base import get_model_passt
from llm2vec import LLM2Vec


class WindowedPrediction(nn.Module):

    def __init__(self, model, segment_length=10, hop_size=10, sr=32000):
        super(WindowedPrediction, self).__init__()
        self.model = model
        self.segment_length = segment_length
        self.hop_size = hop_size
        self.sr = sr

    def forward(self, audio):
        B = len(audio)
        audio = split_audio(audio, segment_length=self.segment_length, hop_size=self.hop_size, sr=self.sr)
        audio_embeds = self.model(audio)  # [B*s, d]
        audio_embeds = audio_embeds.reshape(B, len(audio_embeds) // B, audio_embeds.shape[-1])  # [B, s, d]
        audio_embeds = audio_embeds.mean(dim=1)  # [B, d]
        return audio_embeds


def split_audio(x, segment_length=10, hop_size=10, sr=32000):
    segment_length = int(segment_length * sr)
    hop_size = int(hop_size * sr)
    if x.shape[1] < segment_length:
        return x
    x = x.unfold(dimension=1, size=segment_length, step=hop_size).reshape(-1, segment_length)
    return x


def get_passt(**kwargs):
    net = get_model_passt(arch="passt_s_p16_s16_128_ap468", input_tdim=1000, fstride=16, tstride=16,
                          s_patchout_t=15, s_patchout_f=2)

    mel = AugmentMelSTFT(n_mels=128, sr=32000, win_length=800, hopsize=320, n_fft=1024, freqm=0,
                         timem=0, htk=False, fmin=0.0, fmax=None, norm=1, fmin_aug_range=10,
                         fmax_aug_range=2000)

    model = PasstBasicWrapper(mel=mel, net=net, mode="embed_only", **kwargs)

    return WindowedPrediction(model, segment_length=10, hop_size=10, sr=32000)


def get_llm2vec(**kwargs):
    """
    初始化 LLM2Vec 模型
    可选的模型：
    - McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp
    - McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp
    - McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp (更小更快)
    """
    # 获取 token
    hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN")

    model = LLM2Vec.from_pretrained(
        "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp",
        peft_model_name_or_path="McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp",
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        token=hf_token,
        **kwargs
    )
    return model


class DualEncoderModel(nn.Module):

    def __init__(self, freeze_text_encoder=True, unfreeze_layers=4, **kwargs):
        super(DualEncoderModel, self).__init__()

        self.audio_encoder = get_passt()
        self.text_encoder = get_llm2vec()
        

        # Optional: Freeze the base model of the text encoder and only train the projection.
        if freeze_text_encoder:
            for param in self.text_encoder.model.parameters():
                param.requires_grad = False

            # unfreeze the last N layers
            total_layers = len(self.text_encoder.model.layers)  
            for i in range(total_layers - unfreeze_layers, total_layers):
                for param in self.text_encoder.model.layers[i].parameters():
                    param.requires_grad = True
            
            print(f"Unfroze last {unfreeze_layers} layers of {total_layers}")



        # Obtain the hidden layer dimension of LLM2Vec (typically 4096)
        text_hidden_size = self.text_encoder.config.hidden_size
        print(f"Text encoder hidden size: {text_hidden_size}")

        self.audio_proj = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 2048),        
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(2048, 2048)
        )




        self.text_proj = nn.Sequential(
            nn.LayerNorm(text_hidden_size),        
            nn.Linear(text_hidden_size, 2048),       
            nn.GELU(),                   
            nn.Dropout(0.3),             
            nn.Linear(2048, 2048)
        )

    def audio_branch(self, audio):
        audio_input = audio.to(self.text_encoder.model.device)
        audio_feats = self.audio_encoder(audio_input)
        audio_embeds = F.normalize(self.audio_proj(audio_feats), dim=-1)
        return audio_embeds

    def text_branch(self, text):
        """
        text: List[str] - 文本列表，不需要提前 tokenize
        """
        # LLM2Vec 直接处理文本列表
        text_embeds = self.text_encoder.encode(text)  # 返回 numpy array
        
        # 转换为 tensor 并移到正确的设备
        text_embeds = torch.tensor(text_embeds).to(self.text_encoder.model.device)
        
        # 投影到共享空间
        text_embeds = self.text_proj(text_embeds)
        text_embeds = F.normalize(text_embeds, dim=-1)
        
        return text_embeds

    def forward(self, audio, text):
        """
        audio: [B, T] - 音频波形
        text: List[str] - 文本列表，长度为 B
        """
        audio_embeds = self.audio_branch(audio)  # [B, 1024]
        text_embeds = self.text_branch(text)      # [B, 1024]
        return audio_embeds, text_embeds 