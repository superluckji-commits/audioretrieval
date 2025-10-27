import torch
import torch.nn as nn
import torch.nn.functional as F
from hear21passt.base import AugmentMelSTFT, PasstBasicWrapper
from hear21passt.base import get_model_passt
from transformers import RobertaModel, RobertaTokenizer
import math  # ✅ 新增



class PositionalEncoding(nn.Module):
    """位置编码 - 可学习的位置表示（初版先不用，等对齐稳定再开启）"""
    def __init__(self, d_model=768, max_len=2048):
        super().__init__()
        # 可学习的位置编码（初始化很小，避免扰动预训练权重）
        self.register_buffer('pe', torch.randn(1, max_len, d_model) * 0.02)
    
    def forward(self, x, use_pe=False):
        """
        x: [B, T, D]
        use_pe: bool - 是否启用位置编码（初版False，稳定后改True）
        返回: [B, T, D]
        """
        if not use_pe:
            return x
        
        T = x.shape[1]
        if T > self.pe.shape[1]:
            # 如果序列超出预定义长度，自动扩展
            device = x.device
            new_pe = torch.randn(1, T, x.shape[-1], device=device) * 0.02
            return x + new_pe
        
        return x + self.pe[:, :T, :]




class TemporalAttentionPooling(nn.Module):
    """
    时序注意力池化
    自动学习哪些时间段更重要
    """
    def __init__(self, dim):
        super().__init__()
        # 注意力网络：输入embedding，输出重要性分数
        self.attention = nn.Sequential(
            nn.Linear(dim, dim // 4),  # 降维
            nn.Tanh(),                 # 非线性
            nn.Linear(dim // 4, 1)     # 输出1维分数
        )
    
    def forward(self, x):
        # x: [B, T, D]
        # B = batch size, T = 时间段数量, D = embedding维度
        
        # 计算每个时间段的注意力分数
        scores = self.attention(x)  # [B, T, 1]
        
        # softmax归一化，变成概率分布（和为1）
        weights = torch.softmax(scores, dim=1)  # [B, T, 1]
        
        # 加权求和
        output = (x * weights).sum(dim=1)  # [B, D]
        
        return output



class ContextGate(nn.Module):
    """门控融合 - 学习如何混合原始特征和增强特征"""
    def __init__(self, d_model=768, init_bias=-2.0):
        super().__init__()
        
        # 接收拼接的特征 [原始; 增强]，输出门控值
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        
        # 初始化让门开始是"半关"状态（更稳定）
        # 修正：设置最后一层Linear的bias和权重
        final_linear = self.gate[-2]  # 最后的Sigmoid前的Linear
        final_linear.bias.data.fill_(init_bias)  # bias初始化为-2.0
        final_linear.weight.data.mul_(0.1)  # 权重也缩小，进一步稳定
    
    def forward(self, original, enhanced):
        """
        original: [B, T, D] 原始特征
        enhanced: [B, T, D] 增强特征
        返回: (融合特征, 门控值)
        """
        # 拼接原始和增强
        combined = torch.cat([original, enhanced], dim=-1)  # [B, T, 2D]
        
        # 计算门控值
        gate = self.gate(combined)  # [B, T, D]
        
        # 融合：保留原始 + 门控地添加增强
        # 修正：对enhanced先缩放，增强稳定性
        output = original + gate * (0.5 * enhanced)
        
        return output, gate



class WindowedPrediction(nn.Module):
    def __init__(self, model, segment_length=2, hop_size=10, sr=32000, use_attention=False):
        super().__init__()
        self.model = model
        self.segment_length = segment_length
        self.hop_size = hop_size
        self.sr = sr
        self.use_attention = use_attention
        if use_attention:
            self.attention_pooling = TemporalAttentionPooling(dim=768)

    def forward(self, audio, return_tokens=False, drop_cls=True):
        """
        audio: [B, T_audio]
        return_tokens:
          - True  -> 返回 [B, L, 768]（给 cross-attn）
          - False -> 返回 [B, 768]（clip 级）
        """
        B = audio.size(0)

        # 1) 切窗
        x = split_audio(audio, segment_length=self.segment_length,
                        hop_size=self.hop_size, sr=self.sr)          # [B*s, T_seg]

        # 2) 过 PaSST 包装
        y = self.model(x)  # embed_only: [B*s, 768]；full: [B*s, T_tok, 768]（理想情况）

        # 3) 兼容两种输出维度
        if y.dim() == 2:
            # ---- embed_only 路径：每窗一个 768 向量 ----
            Bs, D = y.shape               # [B*s, 768]
            assert D == 768, f"expect 768-d, got {y.shape}"
            s = Bs // B
            y = y.view(B, s, D).contiguous()    # [B, s, 768]

            if return_tokens:
                # 没有 token 维，只能“把每个窗当成一个 token”
                return y  # [B, s, 768]

            # 否则对窗维池化成 clip 向量
            if self.use_attention:
                return self.attention_pooling(y)         # [B, 768]
            else:
                return y.mean(dim=1)                     # [B, 768]

        elif y.dim() == 3:
            # ---- full 路径：窗内有 token 序列 ----
            Bs, T_tok, D = y.shape          # [B*s, T_tok, 768]
            assert D == 768, f"expect D=768, got {y.shape}"
            s = Bs // B
            y = y.view(B, s, T_tok, D).contiguous()  # [B, s, T_tok, 768]

            # 可选：丢 CLS
            if drop_cls and T_tok > 0:
                y = y[:, :, 1:, :].contiguous()     # [B, s, T_tok-1, 768]
                T_tok = T_tok - 1

            # 合并（窗×token）→ 序列维
            y_flat = y.view(B, s * T_tok, D).contiguous()  # ✅ [B, L, 768]

            if return_tokens:
                return y_flat  # ✅ 给 cross-attn 用

            if self.use_attention:
                return self.attention_pooling(y_flat)      # [B, 768]
            else:
                return y_flat.mean(dim=1)                  # [B, 768]

        else:
            raise ValueError(f"Unexpected PaSST output ndim={y.dim()} shape={tuple(y.shape)}")





class TokenLevelCrossModal(nn.Module):
    """Token级别的双向Cross-Attention模型 - 完全修正版"""
    
    def __init__(self, audio_encoder, text_encoder, d_model=768, 
                 enable_a2t=True, enable_t2a=True, use_pe=False):
        super().__init__()
        
        # 编码器
        self.audio_encoder = audio_encoder
        self.text_encoder = text_encoder
        
        # 配置项
        self.enable_a2t = enable_a2t
        self.enable_t2a = enable_t2a
        self.use_pe = use_pe  # 位置编码开关（初版False）
        self.d_model = d_model
        
        # ============ 修正1：动态获取隐层维度 ============
        # 音频编码器的输出维度（通过WindowedPrediction）
        d_a_in = 768  # PaSST输出768维
        
        # 文本编码器的输出维度
        d_t_in = getattr(text_encoder.config, "hidden_size", 1024)  # RoBERTa-large是1024
        
        print(f"[TokenLevelCrossModal] Audio encoder output dim: {d_a_in}")
        print(f"[TokenLevelCrossModal] Text encoder output dim: {d_t_in}")
        print(f"[TokenLevelCrossModal] Target d_model: {d_model}")
        
        # 维度对齐层（修正：使用实际的输入维度）
        self.audio_align = nn.Linear(d_a_in, d_model)
        self.text_align = nn.Linear(d_t_in, d_model)
        
        # ============ 规范化层（增强稳定性）============
        self.pre_ln_audio = nn.LayerNorm(d_model)
        self.pre_ln_text = nn.LayerNorm(d_model)
        self.post_ln_audio = nn.LayerNorm(d_model)
        self.post_ln_text = nn.LayerNorm(d_model)
        
        # ============ 位置编码（初版不启用） ============
        self.audio_pos = PositionalEncoding(d_model, max_len=2048)
        self.text_pos = PositionalEncoding(d_model, max_len=512)
        
        # ============ 双向Cross-Attention ============
        self.a2t_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=8,
            batch_first=True,
            dropout=0.1
        )
        
        self.t2a_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=8,
            batch_first=True,
            dropout=0.1
        )
        
        # ============ 门控融合 ============
        self.audio_gate = ContextGate(d_model, init_bias=-2.0)
        self.text_gate = ContextGate(d_model, init_bias=-2.0)
        
        # ============ 投影到共享空间 ============
        self.audio_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024)
        )
        
        self.text_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024)
        )


    def forward(self, audio_seq, text_seq, attention_mask, use_cross_attention=True):
        """
        audio_seq: [B, d_a_in] - 音频序列编码后的向量（来自get_passt()）
        text_seq: [B, T_t, d_t_in] - 文本序列（来自text_encoder）
        attention_mask: [B, T_t] - 文本attention mask (1=valid, 0=padding)
        use_cross_attention: bool - 是否启用cross-attention
        
        返回:
        audio_embed: [B, 1024] - 归一化的音频embedding
        text_embed: [B, 1024] - 归一化的文本embedding
        """

        # 期待 audio_seq: [B, T_a, 768]；text_seq: [B, T_t, 1024]
        assert audio_seq.dim() == 3 and audio_seq.size(-1) == 768, f"audio_seq shape={audio_seq.shape}"
        assert text_seq.dim() == 3 and text_seq.size(-1) == getattr(self.text_encoder.config, 'hidden_size', 1024), \
                f"text_seq shape={text_seq.shape}"
        
        # 修正：确保在训练时且要求cross_attention才使用
        use_cross_attn = use_cross_attention

        if use_cross_attn:
            print(f"[Cross-Attention] ENABLED - use_cross_attention={use_cross_attention}, model.training={self.training}")
        else:
            print(f"[Cross-Attention] DISABLED - use_cross_attention={use_cross_attention}, model.training={self.training}")    
        
        # ============ 阶段1：维度对齐 ============
        # 音频可能是[B, d_a_in]（如果已经池化了）
        # 如果是[B, T_a, d_a_in]（序列），也要处理
        
        if audio_seq.dim() == 2:
            # [B, d_a_in] -> [B, 1, d_a_in]（变成序列长度为1）
            audio_seq = audio_seq.unsqueeze(1)
        
        audio_seq = self.audio_align(audio_seq)  # [B, T_a, d_model]
        text_seq = self.text_align(text_seq)      # [B, T_t, d_model]
        
        # ============ 阶段2：位置编码（初版不启用） ============
        audio_seq = self.audio_pos(audio_seq, use_pe=self.use_pe)
        text_seq = self.text_pos(text_seq, use_pe=self.use_pe)
        
        # ============ 阶段3：前置规范化 ============
        
        audio_seq_ln = self.pre_ln_audio(audio_seq)  # [B, T_a, d_model]
        text_seq_ln = self.pre_ln_text(text_seq)      # [B, T_t, d_model]
        
        # ============ 阶段4：双向Cross-Attention ============
        
        if use_cross_attn:
            # 修正：构造正确的padding mask
            # attention_mask: [B, T_t] (1=valid, 0=padding from tokenizer)
            # key_padding_mask需要: True=padding
            text_key_padding_mask = (attention_mask == 0).bool()  # [B, T_t]
            
            # A2T：音频查询文本
            if self.enable_a2t:
                audio_enhanced, a2t_weights = self.a2t_attention(
                    query=audio_seq_ln,           # [B, T_a, d_model]
                    key=text_seq_ln,              # [B, T_t, d_model]
                    value=text_seq_ln,            # [B, T_t, d_model]
                    key_padding_mask=text_key_padding_mask  # [B, T_t]
                )
                # ✅ 新增debug信息
                print(f"[A2T] audio_enhanced mean: {audio_enhanced.mean():.6f}, std: {audio_enhanced.std():.6f}")
                print(f"[A2T] attention weights shape: {a2t_weights.shape}, mean: {a2t_weights.mean():.6f}")
            else:
                audio_enhanced = torch.zeros_like(audio_seq_ln)
            
            # T2A：文本查询音频
            # 修正：添加音频的key_padding_mask
            if self.enable_t2a:
                # 假设音频序列没有padding（常见情况），初始化为全False
                B, T_a = audio_seq_ln.shape[:2]
                audio_key_padding_mask = torch.zeros(
                    B, T_a, dtype=torch.bool, device=audio_seq_ln.device
                )
                
                text_enhanced, t2a_weights = self.t2a_attention(
                    query=text_seq_ln,            # [B, T_t, d_model]
                    key=audio_seq_ln,             # [B, T_a, d_model]
                    value=audio_seq_ln,           # [B, T_a, d_model]
                    key_padding_mask=audio_key_padding_mask  # [B, T_a]
                )
                # ✅ 新增debug信息
                print(f"[T2A] text_enhanced mean: {text_enhanced.mean():.6f}, std: {text_enhanced.std():.6f}")
                print(f"[T2A] attention weights shape: {t2a_weights.shape}, mean: {t2a_weights.mean():.6f}")
            else:
                text_enhanced = torch.zeros_like(text_seq_ln)
        else:
            # 推理时：不做cross-attention
            audio_enhanced = torch.zeros_like(audio_seq_ln)
            text_enhanced = torch.zeros_like(text_seq_ln)
        
        # ============ 阶段5：ContextGate融合 ============
        
        audio_fused, audio_gate_values = self.audio_gate(audio_seq_ln, audio_enhanced)
        text_fused, text_gate_values = self.text_gate(text_seq_ln, text_enhanced)
        
        # ============ 阶段6：后置规范化 ============
        
        audio_fused = self.post_ln_audio(audio_fused)  # [B, T_a, d_model]
        text_fused = self.post_ln_text(text_fused)      # [B, T_t, d_model]
        
        # ============ 阶段7：池化为句向量 ============
        
        # 音频侧：假设无padding，使用全True mask
        B, T_a = audio_fused.shape[:2]
        audio_valid_mask = torch.ones(B, T_a, dtype=torch.bool, device=audio_fused.device)
        audio_pool = (audio_fused * audio_valid_mask.unsqueeze(-1)).sum(dim=1) / \
                     audio_valid_mask.sum(dim=1, keepdim=True).clamp(min=1)
        
        # 文本侧：使用注意力mask
        # attention_mask: [B, T_t] (1=valid, 0=padding)
        text_valid_mask = (attention_mask == 1).bool()
        text_pool = (text_fused * text_valid_mask.unsqueeze(-1)).sum(dim=1) / \
                    text_valid_mask.sum(dim=1, keepdim=True).clamp(min=1)
        
        # ============ 阶段8：投影到共享空间 ============
        
        audio_embed = self.audio_proj(audio_pool)  # [B, 1024]
        text_embed = self.text_proj(text_pool)      # [B, 1024]
        
        # 返回归一化的embedding
        audio_embed = F.normalize(audio_embed, dim=-1)
        text_embed = F.normalize(text_embed, dim=-1)
        
        return audio_embed, text_embed




def split_audio(x, segment_length=10, hop_size=10, sr=32000):
    segment_length = int(segment_length * sr)
    hop_size = int(hop_size * sr)
    
    print(f"[split_audio] input shape: {x.shape}")
    print(f"[split_audio] segment_length: {segment_length}, hop_size: {hop_size}")
    
    if x.shape[1] < segment_length:
        return x
    
    x = x.unfold(dimension=1, size=segment_length, step=hop_size).reshape(-1, segment_length)
    
    print(f"[split_audio] output shape: {x.shape}, num_segments: {x.shape[0]}")
    return x

def get_passt(dataset='audiocaps', use_attention=False, return_tokens=True, **kwargs):
    """
    Args:
        dataset: 'clothov2' 或 'audiocaps'
        use_attention: bool - 是否用注意力池化
        return_tokens: bool - 是否返回token序列
    """
    net = get_model_passt(arch="passt_s_p16_s16_128_ap468", input_tdim=1000, 
                          fstride=16, tstride=16,
                          s_patchout_t=15, s_patchout_f=2)
    
    if dataset.lower() == 'clothov2':
        sr = 44100
        mel = AugmentMelSTFT(
            n_mels=128, sr=44100,
            win_length=1102, hopsize=441, n_fft=2048,
            freqm=0, timem=0, htk=False, fmin=0.0, fmax=None, norm=1,
            fmin_aug_range=10, fmax_aug_range=2000
        )
    elif dataset.lower() == 'audiocaps':
        sr = 32000
        mel = AugmentMelSTFT(
            n_mels=128, sr=32000,
            win_length=800, hopsize=320, n_fft=1024,
            freqm=0, timem=0, htk=False, fmin=0.0, fmax=None, norm=1,
            fmin_aug_range=10, fmax_aug_range=2000
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    # ✅ 选择mode
    mode = "embed_only"
    
    model = PasstBasicWrapper(mel=mel, net=net, mode=mode)
    
    wrapped = WindowedPrediction(model, segment_length=10, hop_size=10, sr=sr, 
                                 use_attention=use_attention)
    
    
    return wrapped


def get_roberta(**kwargs):
    model = RobertaModel.from_pretrained("roberta-large",
                                         add_pooling_layer=False, hidden_dropout_prob=0.2,
                                         attention_probs_dropout_prob=0.2, output_hidden_states=False)
    tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
    return model, tokenizer


class DualEncoderModel(nn.Module):

    def __init__(self, **kwargs):
        super(DualEncoderModel, self).__init__()

        self.audio_encoder = get_passt()
        self.text_encoder, self.tokenizer = get_roberta()

        self.audio_proj = nn.Sequential(
            # nn.LayerNorm(768),
            nn.Linear(768, 1024),
            nn.ReLU(),
            # nn.Dropout(0.3),
            nn.Linear(1024, 1024),
        )

        self.text_proj = nn.Sequential(
            # nn.LayerNorm(1024),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            # nn.Dropout(0.3),
            nn.Linear(1024, 1024),
        )

    def audio_branch(self, audio):
        audio_input = audio.to(self.text_encoder.device)
        audio_feats = self.audio_encoder(audio_input)
        audio_embeds = F.normalize(self.audio_proj(audio_feats), dim=-1)
        return audio_embeds

    def text_branch(self, text):
        text_input = self.tokenizer(text,
                                    add_special_tokens=True,
                                    padding="max_length",
                                    truncation=True,
                                    max_length=32,
                                    return_tensors="pt").to(self.text_encoder.device)
        text_feats = self.text_encoder(input_ids=text_input.input_ids,
                                       attention_mask=text_input.attention_mask)[0]
        text_embeds = F.normalize(self.text_proj(text_feats[:, 0, :]), dim=-1)
        return text_embeds

    def forward(self, audio, text):
        audio_embeds = self.audio_branch(audio)  # [B, d]
        text_embeds = self.text_branch(text)  # [B, d]
        return audio_embeds, text_embeds



class DualEncoderModelWithCrossAttention(nn.Module):
    def __init__(
        self, 
        dataset,
        enable_a2t=True, 
        enable_t2a=True, 
        use_pe=False,
        return_audio_tokens=True,  # ✅ 新增：是否返回音频tokens
        **kwargs
    ):
        super().__init__()
        
        # ✅ 改：让audio_encoder返回tokens
        self.audio_encoder = get_passt(dataset=dataset, return_tokens=return_audio_tokens)
        self.text_encoder, self.tokenizer = get_roberta()
        
        self.cross_modal = TokenLevelCrossModal(
            audio_encoder=self.audio_encoder,
            text_encoder=self.text_encoder,
            d_model=768,
            enable_a2t=enable_a2t,
            enable_t2a=enable_t2a,
            use_pe=use_pe
        )
    
    def forward(self, audio, input_ids, attention_mask, use_cross_attention=True):
        device = next(self.parameters()).device
        
        audio = audio.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        # ✅ 现在audio_seq是 [B, s, 768]（token序列）
        audio_seq = self.audio_encoder(audio, return_tokens=True)  # [B, s, 768]
        
        # 编码文本
        text_output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        text_seq = text_output.last_hidden_state  # [B, T_text, 1024]
        
        # 调用cross-modal
        audio_embed, text_embed = self.cross_modal(
            audio_seq=audio_seq,           # ✅ [B, s, 768] 而不是 [B, 768]
            text_seq=text_seq,
            attention_mask=attention_mask,
            use_cross_attention=use_cross_attention
        )
        
        return audio_embed, text_embed