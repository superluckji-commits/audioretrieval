import os
import pickle

import pandas
from sentence_transformers import SentenceTransformer

model_name = "sbert_mpnet"
model = SentenceTransformer("all-mpnet-base-v2")  # 768-dimensional embeddings

root_dir = "/home/xjii0026/ml23_scratch2/xjii0026/clotho_v2"
split = 'validation'

captions_csv = f'clotho_captions_{split}.csv'
captions = pandas.read_csv(os.path.join(root_dir, captions_csv))
# captions = captions.set_index('file_name')

text_embeds = {}
for idx in captions.index:
    item = captions.loc[idx]

    for caption_idx in range(5):
        cid = item['file_name'] + f'_{caption_idx + 1}'
        caption = item['caption' + f'_{caption_idx + 1}']
        text_embeds[cid] = model.encode(caption)

    print(idx, item['file_name'])

# Save text embeddings
embed_fpath = os.path.join(root_dir, f"clotho_captions_sbert_{split}.pkl")
with open(embed_fpath, "wb") as stream:
    pickle.dump(text_embeds, stream)
print("Save", embed_fpath)
