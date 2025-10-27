# Text-Audio Retrieval

This repository provides the implementation of a dual-encoder model for Text-Audio Cross-Modal Retrieval using PaSST (audio encoder) and RoBERTa (text encoder).

## Quick Start Guide

This repository is developed with Python 3.9 and [PyTorch 1.13.1](https://pytorch.org/).

1. Check out source code and install required python libraries.

```
git clone https://github.com/xieh97/text-audio-retrieval.git
pip install -r requirements.txt
```

2. Download audio-caption datasets. Note: AudioCaps is sourced from YouTube, and some of the original videos are no longer available.

| Dataset   | Train  | Validation | Test | Link                                                          |
|-----------|--------|------------|------|---------------------------------------------------------------|
| AudioCaps | 48548  | 380        | 940  | [GitHub](https://github.com/cdjkim/audiocaps)                 |
| Clotho    | 19195  | 1045       | 1045 | [Zenodo](https://zenodo.org/records/3490684)                  |
| WavCaps   | 401195 | N/A        | N/A  | [Hugging Face](https://huggingface.co/datasets/cvssp/WavCaps) |

3. Preprocess audio and caption data.

```
preprocess
├─ audiocaps.py                 # generate AudioCaps caption embeddings
├─ clothov2.py                  # generate Clotho caption embeddings
├─ wavcaps.py                   # generate WavCaps mp3 audio files
└─ wavcaps2.py                  # generate WavCaps caption embeddings
```

4. Train the model.

```
datasets
├─ audioset.py                  # load WavCaps (AudioSet)
├─ audio_caps.py                # load AudioCaps
├─ clotho_v2.py                 # load Clotho
├─ dataset_base_classes.py      # cache data
├─ wavcaps.py                   # load WavCaps
└─ __init__.py

utils
├─ criterion_utils.py           # cross-entropy losses
├─ data_utils.py                # load datasets
├─ directories.py               # dataset and cache directories
├─ model_utils.py               # model.train(), model.test(), etc.
└─ optim_utils.py               # learning rate schedulers

data_loader.py                  # Pytorch dataloaders
models.py                       # dual-encoder models
ex_baseline.py                  # main()
```

## Attribution and Acknowledgment

This repository contains code adapted from [Estimated Audio–Caption Correspondences Improve Language-Based Audio Retrieval](https://github.com/OptimusPrimus/salsa).
Changes have been made to the original code to suit the specific requirements of this project.
Special thanks to Authors of [[1]](https://github.com/OptimusPrimus/salsa) and [[2]](https://github.com/kkoutini/PaSST) for their contribution to the open-source community.

## References

- [1] https://github.com/OptimusPrimus/salsa
- [2] https://github.com/kkoutini/PaSST
