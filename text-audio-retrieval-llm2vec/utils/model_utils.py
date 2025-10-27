import numpy
import torch

from models import DualEncoderModel
from utils import optim_utils


def init_model(freeze_text_encoder=True):
    model = DualEncoderModel(freeze_text_encoder=freeze_text_encoder)
    return model


def train(model, data_loader, criterion, optimizer, scaler, current_epoch, max_epochs, rampdown_stop):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion.to(device=device)
    model.to(device=device)

    model.train()

    epoch_loss = 0.0

    for batch_idx, batch in enumerate(data_loader, 1):
        optimizer.zero_grad()
        epoch = current_epoch + (batch_idx / len(data_loader))
        
        # ✅ 使用合理的默认值
        optim_utils.update_lr(
            optimizer, 
            epoch, 
            max_epochs=max_epochs,
            base_lr=optimizer.defaults['lr'],
            min_lr=1e-7,
            warmup_length=2,          # 2 epoch warmup
            warmup_type='linear',
            rampdown_start=15,        # epoch 15 开始衰减
            rampdown_stop=rampdown_stop,
            rampdown_type='cosine'
        )

        with torch.autocast(device_type=device.type, dtype=torch.float16):
            audio_embeds, text_embeds = model(batch['audio'], batch['caption'])
            loss = criterion(audio_embeds, text_embeds, batch)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.cpu().item()

    return epoch_loss / len(data_loader)


def test(model, data_loader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion.to(device=device)
    model.to(device=device)

    model.eval()

    test_loss = 0.0
    idx_all, paths_all, audio_embeds_all, text_embeds_all = [], [], [], []

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader, 1):
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                audio_embeds, text_embeds = model(batch['audio'], batch['caption'])
                loss = criterion(audio_embeds, text_embeds, batch)

            test_loss += loss.cpu().item()
            idx_all.append(batch['idx'])
            paths_all.append(batch['path'])
            audio_embeds_all.append(audio_embeds.cpu())
            text_embeds_all.append(text_embeds.cpu())

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
