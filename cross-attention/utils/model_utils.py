import numpy
import torch

from models import DualEncoderModelWithCrossAttention, get_roberta
from utils import optim_utils

# ✅ 新增
_tokenizer = None

def get_tokenizer():
    """获取全局tokenizer实例"""
    global _tokenizer
    if _tokenizer is None:
        _, _tokenizer = get_roberta()
    return _tokenizer


# ✅ 改后
def init_model(config=None):
    if config is None:
        config = {}
    
    model = DualEncoderModelWithCrossAttention(
        dataset=config.get('dataset', 'clothov2'),
        enable_a2t=config.get('enable_a2t', True),
        enable_t2a=config.get('enable_t2a', True),
        use_pe=config.get('use_pe', False),
        return_audio_tokens=config.get('return_audio_tokens', True),  # ← 补上
    )
    return model


def train(model, data_loader, criterion, optimizer, scaler, current_epoch, max_epochs, rampdown_stop):
    """
    训练一个epoch
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion.to(device=device)
    model.to(device=device)
    
    # ✅ 新增：获取tokenizer
    tokenizer = get_tokenizer()

    model.train()

    epoch_loss = 0.0

    for batch_idx, batch in enumerate(data_loader, 1):
        optimizer.zero_grad()
        epoch = current_epoch + (batch_idx / len(data_loader))
        optim_utils.update_lr(optimizer, epoch, max_epochs=max_epochs, rampdown_stop=rampdown_stop)

        # ============ ✅ 改动开始（新增） ============
        
        # 步骤1：获取音频和文本
        audio = batch['audio'].to(device)  # [B, T_audio]
        captions = batch['caption']        # 字符串列表，例如 ['a dog barks', 'rain falls', ...]
        
        # 步骤2：Tokenize文本
        text_input = tokenizer(
            captions,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=32,
            return_tensors="pt"
        ).to(device)
        # text_input.input_ids shape: [B, 32]
        # text_input.attention_mask shape: [B, 32]
        
        # 步骤3：模型前向传播（改了这里）
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            # ❌ 旧的方式（删除）：
            # audio_embeds, text_embeds = model(batch['audio'], batch['caption'])
            
            # ✅ 新的方式（替换为）：
            audio_embeds, text_embeds = model(
                audio,                          # [B, T_audio]
                text_input.input_ids,           # [B, 32]
                text_input.attention_mask,      # [B, 32]
                use_cross_attention=True        # ✅ 训练时启用cross-attention
            )
            # audio_embeds shape: [B, 1024]
            # text_embeds shape: [B, 1024]
            
            loss = criterion(audio_embeds, text_embeds, batch)

        # 步骤4：反向传播和优化
        scaler.scale(loss).backward()
        
        # ✅ 新增：在step前unscale并梯度裁剪
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()

        # ============ ✅ 改动结束 ============

        epoch_loss += loss.cpu().item()

    return epoch_loss / len(data_loader)



def test(model, data_loader, criterion, use_cross_attention=True):
    """
    评估模型
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion.to(device=device)
    model.to(device=device)
    
    # ✅ 新增：获取tokenizer
    tokenizer = get_tokenizer()

    model.eval()

    test_loss = 0.0
    idx_all, paths_all, audio_embeds_all, text_embeds_all = [], [], [], []

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader, 1):
            
            # ============ ✅ 改动开始（新增） ============
            
            # 步骤1：获取音频和文本
            audio = batch['audio'].to(device)  # [B, T_audio]
            captions = batch['caption']        # 字符串列表
            
            # 步骤2：Tokenize文本
            text_input = tokenizer(
                captions,
                add_special_tokens=True,
                padding="max_length",
                truncation=True,
                max_length=32,
                return_tensors="pt"
            ).to(device)
            
            # 步骤3：模型前向传播（改了这里）
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                # ❌ 旧的方式（删除）：
                # audio_embeds, text_embeds = model(batch['audio'], batch['caption'])
                
                # ✅ 新的方式（替换为）：
                audio_embeds, text_embeds = model(
                    audio,                          # [B, T_audio]
                    text_input.input_ids,           # [B, 32]
                    text_input.attention_mask,      # [B, 32]
                    use_cross_attention=use_cross_attention        # ✅ 验证时也启用
                )
                loss = criterion(audio_embeds, text_embeds, batch)

            # ============ ✅ 改动结束 ============

            test_loss += loss.cpu().item()
            idx_all.append(batch['idx'])
            paths_all.append(batch['path'])
            audio_embeds_all.append(audio_embeds.cpu())
            text_embeds_all.append(text_embeds.cpu())

    # ============ 以下代码不变 ============
    
    paths_all = numpy.array([p for b in paths_all for p in b])
    audio_embeds_all = torch.cat(audio_embeds_all, dim=0)
    text_embeds_all = torch.cat(text_embeds_all, dim=0)

    paths_set = numpy.unique(paths_all)
    audio_idx = numpy.array([numpy.where(paths_all == p)[0].min() for p in paths_set])
    audio_embeds_all = audio_embeds_all[audio_idx]

    predictions = audio_embeds_all @ text_embeds_all.t()

    # text-to-audio retrieval
    audio_topk = predictions.topk(10, dim=0)[1].numpy()
    audio_topk = paths_set[audio_topk]
    audio_target = numpy.tile(paths_all, (10, 1))

    audio_R1 = (audio_topk[:1, :] == audio_target[:1, :]).sum(axis=0).mean()
    audio_R5 = (audio_topk[:5, :] == audio_target[:5, :]).sum(axis=0).mean()
    audio_R10 = (audio_topk[:10, :] == audio_target[:10, :]).sum(axis=0).mean()

    audio_AP = 1 / ((audio_topk[:10, :] == audio_target[:10, :]).argmax(axis=0) + 1)
    audio_AP[~(audio_topk[:10, :] == audio_target[:10, :]).any(axis=0)] = 0.
    audio_mAP = audio_AP.mean()

    # audio-to-text retrieval
    text_topk = predictions.topk(10, dim=1)[1].numpy()
    text_topk = paths_all[text_topk]
    text_target = numpy.tile(paths_set.reshape(-1, 1), (1, 10))

    assert text_topk.shape == text_target.shape

    text_R1 = (text_topk[:, :1] == text_target[:, :1]).sum() / len(paths_all)
    text_R5 = (text_topk[:, :5] == text_target[:, :5]).sum() / len(paths_all)
    text_R10 = (text_topk[:, :10] == text_target[:, :10]).sum() / len(paths_all)

    text_IDX = text_topk[:, :10] == text_target[:, :10]
    text_SUM = numpy.cumsum(text_IDX, axis=1)
    text_POS = numpy.tile(numpy.arange(1, 11), (len(text_SUM), 1))
    text_AP = (text_SUM / text_POS) * text_IDX
    text_mAP = text_AP.sum(axis=1).mean() / 5.0

    return {
        "test_loss": test_loss / len(data_loader),
        "audio_retrieval": {
            "R1": audio_R1,
            "R5": audio_R5,
            "R10": audio_R10,
            "mAP": audio_mAP
        },
        "text_retrieval": {
            "R1": text_R1,
            "R5": text_R5,
            "R10": text_R10,
            "mAP": text_mAP
        }
    }


def restore(model, ckpt):
    with torch.no_grad():
        model_state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(model_state[0])
    del model_state
    torch.cuda.empty_cache()
    return model



