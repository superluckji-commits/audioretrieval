import os
import pickle

from sentence_transformers import SentenceTransformer

from utils.data_utils import get_data_set

model_name = "sbert_mpnet"
model = SentenceTransformer("all-mpnet-base-v2")  # 768-dimensional embeddings

root_dir = "~/Audio_Datasets/wavcaps"

wavcaps = get_data_set('wavcaps', "train")
samples = wavcaps.dataset.samples

text_embeds = {}
for idx, s in enumerate(samples):
    cid = s['xhid']
    caption = s['caption']
    text_embeds[cid] = model.encode(caption)
    print(idx, cid)

# Save text embeddings
embed_fpath = os.path.join(root_dir, f"wavcaps_captions_sbert.pkl")
with open(embed_fpath, "wb") as stream:
    pickle.dump(text_embeds, stream)
print("Save", embed_fpath)
