# datasets/audio_caps_check.py
import os, sys
# 把项目根目录加入 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import data_utils
from data_loader import get_train_data_loader

def main():
    # 加载 AudioCaps 训练集
    train_ds = data_utils.get_data_set("audiocaps", "train")
    print("Dataset size:", len(train_ds))

    # 取一个 batch
    train_dl = get_train_data_loader(train_ds, batch_size=2, targets=None)
    batch = next(iter(train_dl))

    # 打印 batch 的基本信息
    print("Keys:", batch.keys())
    print("Audio tensor shape:", batch["audio"].shape)
    print("Captions:", batch["caption"])
    # 注意：有些 DataLoader 会把 numpy list 堆成 ndarray；转一下仅为显示
    import numpy as np
    ce = batch["caption_embed"]
    ce_shape = ce.shape if hasattr(ce, "shape") else np.asarray(ce, dtype=float).shape
    print("Teacher embed shape:", ce_shape)
    print("First path:", batch["path"][0])
    print("First idx:", batch["idx"][0])

if __name__ == "__main__":
    main()


