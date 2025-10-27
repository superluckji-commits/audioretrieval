# -*- coding: utf-8 -*-
# File: datasets/audio_caps.py
# 说明：
# - 不依赖 get_audioset()，直接使用本地 AudioCaps 音频文件
# - 需要你已经在 DATA_HOME/audiocaps 下准备：
#     dataset/train.csv, val.csv, test.csv
#     captions_sbert_train.pkl, captions_sbert_val.pkl, captions_sbert_test.pkl
#     audio/*.wav（或 *.mp3）
# - 与作者代码保持接口一致：get_audiocaps(split) 返回 DatasetBaseClass 的子类
# - __getitem__ 会调用父类 DatasetBaseClass.__get_audio__ 来加载音频波形
# - 评测与损失需要的字段：audio / caption / caption_embed / path / idx

import os
import csv
import pickle
from typing import List, Tuple, Optional

import numpy as np

from datasets.dataset_base_classes import DatasetBaseClass
from utils.directories import get_dataset_dir

SPLITS = ["train", "val", "test"]


def get_audiocaps(split: str):
    """与作者保持相同入口。"""
    return AudioCapsDataset(split)


class AudioCapsDataset(DatasetBaseClass):
    """
    简化版 AudioCaps 数据集（直接使用本地 AudioCaps 音频），去掉对 AudioSet 的依赖。
    需要确保你的音频文件命名能与 (youtube_id, start_time) 匹配。
    常见命名：{youtube_id}_{start}.wav  （start 为秒的整数）
    """

    def __init__(self, split: str, folder_name: str = "audiocaps", compress: bool = True, mp3: bool = False):
        super().__init__()

        # -------- 基础检查 --------
        root_dir = os.path.join(get_dataset_dir(), folder_name)
        assert os.path.exists(root_dir), f"[AudioCapsDataset] Root dir not found: {root_dir}"
        assert split in SPLITS or split == "validation", f"[AudioCapsDataset] split must be in {SPLITS}."

        self.split = "val" if split == "validation" else split
        self.audio_caps_root = root_dir
        self.compress = compress
        self._prefer_mp3 = mp3  # 若你的音频是 mp3，则会优先找 mp3

        # -------- 读取 CSV --------
        # CSV 列：audiocap_id, youtube_id, start_time, caption
        csv_path = os.path.join(self.audio_caps_root, "dataset", f"{self.split}.csv")
        assert os.path.exists(csv_path), f"[AudioCapsDataset] CSV not found: {csv_path}"

        rows: List[Tuple[str, str, str, str]] = []
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f, delimiter=",", quotechar='"', doublequote=True, escapechar=None, strict=False)
            header = next(reader, None)  # 跳过表头
            for row in reader:
                if not row or all((c is None) or (str(c).strip() == "") for c in row):
                    continue
                # 兼容 caption 中包含逗号的情况：只取前3列，其余合并为 caption
                if len(row) >= 4:
                    aid, ytid, start = row[0], row[1], row[2]
                    caption = ",".join(row[3:]).strip()
                    rows.append((aid, ytid, start, caption))

        # -------- 读取 caption teacher 向量 --------
        pkl_path = os.path.join(self.audio_caps_root, f"captions_sbert_{self.split}.pkl")
        assert os.path.exists(pkl_path), f"[AudioCapsDataset] Teacher pkl not found: {pkl_path}"
        with open(pkl_path, "rb") as f:
            # 约定：key = audiocap_id；value = 向量(np.array 或 list)
            captions_embed_map = pickle.load(f)

        # -------- 构造样本列表（路径匹配）--------
        # 你的音频目录
        audio_dir = os.path.join(self.audio_caps_root, "audio")
        assert os.path.isdir(audio_dir), f"[AudioCapsDataset] Audio dir not found: {audio_dir}"

        self.paths: List[str] = []
        self.captions: List[str] = []
        self.captions_embed: List[np.ndarray] = []
        self.keywords: List[str] = []  # 检索任务用不到，留空字符串占位
        self._indices: List[int] = []  # 保留原始顺序索引（用于 idx）

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
            self.keywords.append("")      # 不再依赖 AudioSet 标签
            self._indices.append(i)

        if missing_audio > 0:
            print(f"[AudioCapsDataset] Warning: {missing_audio} samples skipped (audio not found).")
        if missing_embed > 0:
            print(f"[AudioCapsDataset] Warning: {missing_embed} samples skipped (caption_embed missing in pkl).")

        # 方便：作者风格常会在 __getitem__ 里用 idx + 1000000
        # 我们保留 _indices，并在 __getitem__ 时加偏移
        # 其他：set_fixed_length / cache_audios 由 DatasetBaseClass 提供

    # ---------- Dataset 标准接口 ----------
    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        """
        返回与作者一致的键：
          - 'audio'：由父类 __get_audio__ 根据 self.paths[index] 读取（波形/特征）
          - 'caption'：字符串（在线文本编码器用）
          - 'caption_embed'：teacher 向量（float32）
          - 'path'：音频路径（评测/对齐使用）
          - 'idx'：稳定 ID（作者做了 +1000000 偏移）
          - 兼容占位键：'keywords' / 'caption_hard' / 'html' / 'xhid'
        """
        a = self.__get_audio__(index)  # 父类会基于 self.paths[index] 读取音频并放入 a['audio']
        a["path"] = self.paths[index]
        a["caption"] = self.captions[index]
        # 确保是 float32 tensor/array 的上游会在 collate 时处理；保持与作者 CELoss 兼容
        a["caption_embed"] = self.captions_embed[index]
        a["idx"] = int(self._indices[index]) + 1000000

        # 兼容占位（训练流程不使用）
        a["keywords"] = ""       # 原来是 AudioSet 标签，这里留空即可
        a["caption_hard"] = ""   # 若未来你做 hard negative，可在外部填充
        a["html"] = ""           # 可视化占位
        a["xhid"] = ""           # 额外 ID 占位

        return a

    # ---------- 辅助：根据 ytid / start_time 推断本地文件名 ----------
    @staticmethod
    def _find_audio_path(audio_dir: str, ytid: str, start: str, prefer_mp3: bool = False) -> Optional[str]:
        """
        依据常见命名规则尝试匹配文件：
          1) {ytid}_{start}.wav / .mp3       （start 原样）
          2) {ytid}_{int(start)}.wav / .mp3  （start 取整）
          3) {ytid}.wav / .mp3               （有些数据集没有起始秒）
        你可以按自己的实际命名规则在这里继续添加候选。
        """
        # 归一化 start
        start_str = str(start).strip()
        try:
            start_int = int(round(float(start_str)))
        except Exception:
            start_int = None

        # 组合候选列表（优先 mp3 或 wav）
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

    # ---------- 父类 DatasetBaseClass 期望的方法 ----------
    def __get_audio_paths__(self) -> List[str]:
        """供父类缓存/读取使用。"""
        return self.paths

    def __str__(self) -> str:
        return f"AudioCaps_{self.split}"
