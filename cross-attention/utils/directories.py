import os

def make_if_not_exists(path):
    os.makedirs(path, exist_ok=True)
    return path

def get_dataset_dir():
    # AudioCaps 数据根（里面应有 audio/ 和 dataset/ 子目录）
    return make_if_not_exists("/home/xjii0026/ml23_scratch2/xjii0026")

def get_persistent_cache_dir():
    # 预处理/训练产生的缓存文件（如 captions_sbert_*.pkl）
    return make_if_not_exists("/home/xjii0026/ml23_scratch2/xjii0026/cache/tar_audiocaps")