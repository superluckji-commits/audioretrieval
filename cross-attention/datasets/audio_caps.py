# datasets/audio_caps.py

import os
import csv
import pickle
from typing import List, Tuple, Optional

import numpy as np
import torch  # ✅ 确保导入了 torch
from torch_audiomentations import Compose, AddBackgroundNoise, Gain

from datasets.dataset_base_classes import DatasetBaseClass
from utils.directories import get_dataset_dir

SPLITS = ["train", "val", "test"]


def get_audiocaps(split: str):
    """与作者保持相同入口。"""
    return AudioCapsDataset(split)


class AudioCapsDataset(DatasetBaseClass):
    """简化版 AudioCaps 数据集"""

    def __init__(self, split: str, folder_name: str = "audiocaps", 
                 compress: bool = True, mp3: bool = False, 
                 use_augmentation: bool = True):  
        super().__init__()

        # -------- 基础检查 --------
        root_dir = os.path.join(get_dataset_dir(), folder_name)
        assert os.path.exists(root_dir), f"[AudioCaps] Root dir not found: {root_dir}"
        assert split in SPLITS or split == "validation", f"[AudioCaps] split must be in {SPLITS}."

        self.split = "val" if split == "validation" else split
        self.audio_caps_root = root_dir
        self.compress = compress
        self._prefer_mp3 = mp3
        self.use_augmentation = use_augmentation

        # ✅ 音频增强初始化
        if self.split == 'train':
            if self.use_augmentation:
                ESC50_path = "/home/xjii0026/ml23_scratch2/xjii0026/ESC-50-master/audio_32khz"
                print(f"[AudioCaps] ESC-50 path: {ESC50_path}")
                print(f"[AudioCaps] Path exists: {os.path.exists(ESC50_path)}")
                
                print("[AudioCaps] Initializing audio augmentation...")
                self.audio_augment = Compose([
                    AddBackgroundNoise(
                        background_paths=ESC50_path,
                        min_snr_in_db=5.0,
                        max_snr_in_db=20.0,
                        p=0.5,
                        output_type="tensor"
                    ),
                    Gain(
                        min_gain_in_db=-12.0,
                        max_gain_in_db=12.0,
                        p=0.4,
                        output_type="tensor"
                    ),
                ], output_type="tensor")
                print("[AudioCaps] Audio augmentation initialized.")
            else:
                self.audio_augment = None
                print("[AudioCaps] Audio augmentation disabled")
        else:
            self.audio_augment = None
            print(f"[AudioCaps] No augmentation for split={self.split}")

        # -------- 读取 CSV --------
        csv_path = os.path.join(self.audio_caps_root, "dataset", f"{self.split}.csv")
        assert os.path.exists(csv_path), f"[AudioCaps] CSV not found: {csv_path}"

        rows: List[Tuple[str, str, str, str]] = []
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f, delimiter=",", quotechar='"', doublequote=True, 
                               escapechar=None, strict=False)
            header = next(reader, None)
            for row in reader:
                if not row or all((c is None) or (str(c).strip() == "") for c in row):
                    continue
                if len(row) >= 4:
                    aid, ytid, start = row[0], row[1], row[2]
                    caption = ",".join(row[3:]).strip()
                    rows.append((aid, ytid, start, caption))

        # -------- 读取 caption teacher 向量 --------
        pkl_path = os.path.join(self.audio_caps_root, f"captions_sbert_{self.split}.pkl")
        assert os.path.exists(pkl_path), f"[AudioCaps] Teacher pkl not found: {pkl_path}"
        with open(pkl_path, "rb") as f:
            captions_embed_map = pickle.load(f)

        # -------- 构造样本列表 --------
        audio_dir = os.path.join(self.audio_caps_root, "audio")
        assert os.path.isdir(audio_dir), f"[AudioCaps] Audio dir not found: {audio_dir}"

        self.paths: List[str] = []
        self.captions: List[str] = []
        self.captions_embed: List[np.ndarray] = []
        self.keywords: List[str] = []
        self._indices: List[int] = []

        missing_audio = 0
        missing_embed = 0

        for i, (aid, ytid, start, cap) in enumerate(rows):
            embed = captions_embed_map.get(aid, None)
            if embed is None:
                missing_embed += 1
                continue

            audio_path = self._find_audio_path(audio_dir, ytid, start, prefer_mp3=self._prefer_mp3)
            if audio_path is None:
                missing_audio += 1
                continue

            self.paths.append(audio_path)
            self.captions.append(cap)
            self.captions_embed.append(np.asarray(embed))
            self.keywords.append("")
            self._indices.append(i)

        if missing_audio > 0:
            print(f"[AudioCaps] Warning: {missing_audio} samples skipped (audio not found).")
        if missing_embed > 0:
            print(f"[AudioCaps] Warning: {missing_embed} samples skipped (caption_embed missing).")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        """返回与作者一致的键"""
        a = self.__get_audio__(index)
        
        # ✅✅✅ 应用audio augmentation（训练时且启用了augmentation）
        if self.audio_augment is not None:
            audio = a['audio']
            
            # 统一到 torch.float32（CPU）
            if isinstance(audio, np.ndarray):
                audio = torch.from_numpy(audio).float()
            else:
                audio = audio.float()

            # 若多声道，先混单通道 -> [T]
            if audio.dim() == 2 and audio.size(0) > 1:
                audio = audio.mean(dim=0)

            # 确保现在是 [T]
            if audio.dim() == 2 and audio.size(0) == 1:
                audio = audio.squeeze(0)
            assert audio.dim() == 1, f"Expect 1-D waveform before augment, got {audio.shape}"

            # ======= 真正地调用增强（期望 [B, C, T]）=======
            # [T] -> [1, 1, T]
            audio = audio.unsqueeze(0).unsqueeze(0)

            # 调用增强（32k 采样率）
            audio = self.audio_augment(audio, sample_rate=32000)  # -> [1, 1, T]

            # 去掉 batch 和通道 -> [T]
            audio = audio.squeeze(0).squeeze(0)
            # ============================================

            # 保证连续 & float32
            audio = audio.contiguous().float()

            # 若原来是 numpy，就转回 numpy（必须在 CPU 上）
            if isinstance(a['audio'], np.ndarray):
                audio = audio.cpu().numpy()

            # 更新回dict
            a['audio'] = audio
        
        # 其他字段
        a["path"] = self.paths[index]
        a["caption"] = self.captions[index]
        a["caption_embed"] = self.captions_embed[index]
        a["idx"] = int(self._indices[index]) + 1000000

        # 兼容占位
        a["keywords"] = ""
        a["caption_hard"] = ""
        a["html"] = ""
        a["xhid"] = ""

        return a

    @staticmethod
    def _find_audio_path(audio_dir: str, ytid: str, start: str, 
                        prefer_mp3: bool = False) -> Optional[str]:
        """依据常见命名规则尝试匹配文件"""
        start_str = str(start).strip()
        try:
            start_int = int(round(float(start_str)))
        except Exception:
            start_int = None

        exts_pref = [".mp3", ".wav"] if prefer_mp3 else [".wav", ".mp3"]

        candidates: List[str] = []
        for ext in exts_pref:
            candidates.append(os.path.join(audio_dir, f"{ytid}_{start_str}{ext}"))
            if start_int is not None:
                candidates.append(os.path.join(audio_dir, f"{ytid}_{start_int}{ext}"))
            candidates.append(os.path.join(audio_dir, f"{ytid}{ext}"))

        for p in candidates:
            if os.path.exists(p):
                return p
        return None

    def __get_audio_paths__(self) -> List[str]:
        """供父类缓存/读取使用"""
        return self.paths

    def __str__(self) -> str:
        return f"AudioCaps_{self.split}"