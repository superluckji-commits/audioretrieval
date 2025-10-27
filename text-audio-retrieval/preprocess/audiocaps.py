import csv
import os
import pickle

from sentence_transformers import SentenceTransformer

# ===== 与你源代码保持一致的配置 =====
model_name = "sbert_mpnet"
model = SentenceTransformer("all-mpnet-base-v2")  # 768-dimensional embeddings

root_dir = "/home/xjii0026/ml23_scratch2/xjii0026/audiocaps"
split = 'test'
csv_path = os.path.join(root_dir, 'dataset', f'{split}.csv')

# ===== 稳健读取：对第4列之后的内容重新拼回 caption =====
lines = []
with open(csv_path, 'r', encoding='utf-8', newline="") as f:
    reader = csv.reader(
        f,
        delimiter=',',
        quotechar='"',
        doublequote=True,
        escapechar=None,
        strict=False
    )
    header = next(reader, None)  # 跳过表头
    for row in reader:
        if not row or all((c is None) or (str(c).strip() == "") for c in row):
            continue
        if len(row) < 4:
            # 跳过异常行（保持与原逻辑最接近：原逻辑实际会在 zip(*) 时崩溃，这里直接忽略坏行）
            continue
        aid = row[0].strip()
        ytid = row[1].strip()
        start = row[2].strip()
        caption = ",".join(row[3:]).strip()  # 关键：把第4列及以后拼回
        # 去掉首尾成对引号 & 还原内部转义
        if len(caption) >= 2 and caption[0] == '"' and caption[-1] == '"':
            caption = caption[1:-1]
        caption = caption.replace('""', '"').strip()
        lines.append([aid, ytid, start, caption])

# ===== 与你源代码一致的列拆分与排序 =====
audiocap_ids, ytids, _, captions = list(map(list, zip(*lines)))
ytids, audiocap_ids, captions = list(zip(*sorted(zip(ytids, audiocap_ids, captions))))

# ===== 与你源代码一致的编码与保存（覆盖相同 aid 的后者）=====
text_embeds = {}
for aid, caption in zip(audiocap_ids, captions):
    text_embeds[aid] = model.encode(caption)
    print(aid, caption)

# Save text embeddings（路径和文件名与原脚本一致）
embed_fpath = os.path.join("/home/xjii0026/ml23_scratch2/xjii0026/retrieval_cache",
                           f"captions_sbert_{split}.pkl")
os.makedirs(os.path.dirname(embed_fpath), exist_ok=True)
with open(embed_fpath, "wb") as stream:
    pickle.dump(text_embeds, stream)
print("Save", embed_fpath)
