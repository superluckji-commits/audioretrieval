import torch
import torch.nn as nn


class CELoss(nn.Module):

    def __init__(self, **kwargs):
        super(CELoss, self).__init__()

        self.tau = kwargs['tau']
        self.target = kwargs['target']
        self.target_func = kwargs['target_func']
        self.audio_modality = kwargs['audio_modality']
        self.text_modality = kwargs['text_modality']

        self.threshold = kwargs['threshold']
        self.alpha = kwargs['alpha']
        self.beta = kwargs['beta']

    def forward(self, audio_embeds, text_embeds, batch):

        # predictions
        preds = audio_embeds @ text_embeds.t() / self.tau
        # preds = sentence_transformers.util.cos_sim(audio_embeds, text_embeds) / self.tau
        preds_audio = torch.log_softmax(preds, dim=0)
        preds_text = torch.log_softmax(preds, dim=1)

        loss = torch.tensor(0., device=audio_embeds.device, requires_grad=True)

        # hard targets
        if self.target in ['hard', 'mixed']:
            paths = torch.tensor([hash(p) for p in batch['path']]).view(-1, 1)
            hard = torch.eq(paths, paths.t()).float().to(audio_embeds.device)
            hard = hard / hard.sum(1, keepdim=True)

            if self.audio_modality:
                loss = loss - torch.sum(hard.t() * preds_audio, dim=0).mean()
            elif self.text_modality:
                loss = loss - torch.sum(hard * preds_text, dim=1).mean()
            else:
                loss = loss - 0.5 * (torch.sum(hard.t() * preds_audio, dim=0).mean() +
                                     torch.sum(hard * preds_text, dim=1).mean())

        # soft targets
        if self.target in ['soft', 'mixed']:
            caption_embeds = batch['caption_embed'].to(audio_embeds.device)
            soft = caption_embeds @ caption_embeds.t()

            # target functions
            if self.target_func == 'threshold':
                soft = torch.threshold(soft, threshold=self.threshold, value=0.0)
                soft = soft / soft.sum(1, keepdim=True)
            elif self.target_func == 'logistic':
                soft = torch.sigmoid(self.alpha * soft - self.beta)
                soft = soft / soft.sum(1, keepdim=True)
            elif self.target_func == 'rating':
                soft = torch.sigmoid(1.85 - 4.58 * (1.0 - soft))
                soft = torch.softmax(soft / self.tau, dim=1)

            if self.audio_modality:
                loss = loss - torch.sum(soft.t() * preds_audio, dim=0).mean()
            elif self.text_modality:
                loss = loss - torch.sum(soft * preds_text, dim=1).mean()
            else:
                loss = loss - 0.5 * (torch.sum(soft.t() * preds_audio, dim=0).mean() +
                                     torch.sum(soft * preds_text, dim=1).mean())

        return loss
