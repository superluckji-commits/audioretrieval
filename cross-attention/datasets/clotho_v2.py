import os
import pickle
import numpy as np                  # ← NEW: 统一 embed dtype
import pandas as pd
import torch

from torch_audiomentations import Compose, AddBackgroundNoise, Gain  # ← NEW

from datasets.dataset_base_classes import DatasetBaseClass
from utils.directories import get_dataset_dir

SPLITS = ['development', 'validation', 'evaluation']


def get_clotho_v2(split):
    splits = {'train': 'development', 'val': 'validation', 'test': 'evaluation'}
    assert split in list(splits.keys())
    ds = Clotho_v2Dataset(splits[split])
    return ds


class Clotho_v2Dataset(DatasetBaseClass):
    def __init__(
        self, 
        split, 
        folder_name='clotho_v2', 
        compress=False,
        add_hard_negatives=False, 
        add_hard_negatives_gpt=False, 
        ablate_while=False,
        use_augmentation=False,                         # ← NEW: 与 AudioCaps 对齐
        ESC50_dir_44k="/home/xjii0026/ml23_scratch2/xjii0026/ESC-50-master/audio"  # ← NEW
    ):
        super().__init__()
        self.compress = compress
        self.add_hard_negatives = add_hard_negatives
        self.add_hard_negatives_gpt = add_hard_negatives_gpt
        self.ablate_while = ablate_while
        self.use_augmentation = use_augmentation        # ← NEW
        self.split = split

        root_dir = os.path.join(get_dataset_dir(), folder_name)
        assert os.path.exists(root_dir), f'Parameter \'root_dir\' is invalid. {root_dir} does not exist.'
        assert split in SPLITS, f'Parameter \'split\' must be in {SPLITS}.'

        self.root_dir = root_dir
        self.files_dir = os.path.join(root_dir, split)
        captions_csv = f'clotho_captions_{split}.csv'
        metadata_csv = f'clotho_metadata_{split}.csv'

        # ---------- Augmentation init (only for development) ----------
        if self.split == 'development' and self.use_augmentation:               # ← NEW
            assert os.path.isdir(ESC50_dir_44k), f"[Clotho] ESC-50 dir not found: {ESC50_dir_44k}"
            print(f"[Clotho] Init audio augmentation with ESC-50 (44.1k): {ESC50_dir_44k}")
            self.audio_augment = Compose([
                AddBackgroundNoise(
                    background_paths=[ESC50_dir_44k],  # 用列表更稳
                    min_snr_in_db=5.0,
                    max_snr_in_db=20.0,
                    p=0.5,
                    output_type="tensor"
                ),
                Gain(min_gain_in_db=-12.0, max_gain_in_db=12.0, p=0.4, output_type="tensor"),
            ])
        else:
            self.audio_augment = None
            print(f"[Clotho] No augmentation for split={self.split}")            # ← NEW

        # ---------- Read CSVs ----------
        metadata = pd.read_csv(os.path.join(root_dir, metadata_csv), encoding="ISO-8859-1").set_index('file_name')
        captions = pd.read_csv(os.path.join(root_dir, captions_csv)).set_index('file_name')

        captions_sbert = f'clotho_captions_sbert_{split}.pkl'
        with open(os.path.join(root_dir, captions_sbert), "rb") as stream:
            captions_embed = pickle.load(stream)

        self.metadata = pd.concat([metadata, captions], axis=1).reset_index()
        self.num_captions = 5

        self.paths, self.attributes = [], []
        for i in range(len(self.metadata) * self.num_captions):
            attributes = dict(self.metadata.iloc[i // self.num_captions].items())

            # path
            path = os.path.join(self.files_dir, attributes['file_name'])
            self.paths.append(path)

            # caption & embed
            caption_idx = i % self.num_captions
            if f'caption_{caption_idx + 1}' in attributes:
                attributes['caption'] = attributes[f'caption_{caption_idx + 1}']
                # 统一为 float32，避免下游 dtype 隐式转换
                key = attributes['file_name'] + f'_{caption_idx + 1}'
                attributes['caption_embed'] = np.asarray(captions_embed[key], dtype=np.float32)  # ← NEW
                if 'caption_2' in attributes:
                    del attributes['caption_1'], attributes['caption_2'], attributes['caption_3'], attributes['caption_4'], attributes['caption_5']
                else:
                    del attributes['caption_1']
            else:
                attributes['caption'] = ''

            attributes['html'] = (
                f'<iframe frameborder="0" scrolling="no" src="https://freesound.org/embed/sound/iframe/{attributes["sound_id"]}/simple/small/" '
                f'width="375" height="30"></iframe>'
            )
            for k in ('sound_id', 'sound_link', 'start_end_samples', 'manufacturer', 'license', 'file_name'):
                if k in attributes:
                    del attributes[k]

            self.attributes.append(attributes)

        # hard negatives
        hard_captions_csv = f'hard_negative_captions_{split}.csv'
        self.hard_negatives = {}
        if os.path.exists(os.path.join(root_dir, hard_captions_csv)) and add_hard_negatives_gpt:
            import csv
            with open(os.path.join(root_dir, hard_captions_csv)) as csvfile:
                spamreader = csv.reader(csvfile)
                for row in spamreader:
                    self.hard_negatives[int(row[0])] = row[3:]

    def __get_audio_paths__(self):
        return self.paths

    def __getitem__(self, item):
        a = self.__get_audio__(item)                   # dict, 包含 'audio'，通常为 [C,S] 或 [S]

        # ---------- Apply augmentation at 44.1 kHz (development only) ----------
        if self.audio_augment is not None:
            audio = a['audio']

            # numpy → torch.float32（CPU）
            if isinstance(audio, np.ndarray):
                audio = torch.from_numpy(audio).float()
            else:
                audio = audio.float()

            # 若多声道，混单通道 -> [T]
            if audio.dim() == 2 and audio.size(0) > 1:  # [C,T]
                audio = audio.mean(dim=0)               # -> [T]
            elif audio.dim() == 2 and audio.size(0) == 1:
                audio = audio.squeeze(0)                # -> [T]
            assert audio.dim() == 1, f"Expect 1-D waveform before augment, got {audio.shape}"

            # 幅度裁剪，防止增益爆炸
            audio = torch.clamp(audio, -1.0, 1.0)

            # [T] → [1,1,T] → 增强 → [T]
            audio = audio.unsqueeze(0).unsqueeze(0)
            audio = self.audio_augment(audio, sample_rate=44100)  # ✅ Clotho 固定 44.1 kHz
            audio = audio.squeeze(0).squeeze(0)

            # 连续 + float32
            audio = audio.contiguous().float()

            # 若原始是 numpy，转回 numpy
            if isinstance(a['audio'], np.ndarray):
                audio = audio.cpu().numpy()

            a['audio'] = audio


        # 填充其余字段
        attributes = self.attributes[item]
        for k in attributes:
            a[k] = attributes[k]
        a['idx'] = item
        a['caption_hard'] = ''
        a['xhid'] = ''

        if a['caption_hard'] != '' and self.hard_negatives.get(item):
            hard_index = torch.randint(len(self.hard_negatives.get(item)), (1,)).item()
            a['caption_hard'] = self.hard_negatives.get(item)[hard_index]

        return a

    def __len__(self):
        return len(self.metadata) * self.num_captions

    def __str__(self):
        return f'ClothoV2_{self.split}'
