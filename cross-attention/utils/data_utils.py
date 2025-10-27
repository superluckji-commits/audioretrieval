from datasets.audio_caps import get_audiocaps
from datasets.clotho_v2 import get_clotho_v2
from datasets.dataset_base_classes import ConcatDataset
from datasets.wavcaps import get_wavecaps


def get_data_set(data_set_id, mode):
    if data_set_id == 'clothov2' or (data_set_id in ['wavcaps', 'all'] and mode != 'train'):
        assert mode in ['train', 'val', 'test']
        ds = get_clotho_v2(mode)
        ds.set_fixed_length(30)
    elif data_set_id == 'audiocaps':
        assert mode in ['train', 'val', 'test']
        ds = get_audiocaps(mode)
        ds.compress = False
        ds.set_fixed_length(10)
    elif data_set_id == 'wavcaps':
        ds = get_wavecaps()
        ds.compress = True
        ds.set_fixed_length(30)
    elif data_set_id == 'all':
        ds = ConcatDataset(
            [
                get_clotho_v2('train'),
                get_audiocaps('train'),
                get_wavecaps()
            ]
        )
        ds.set_fixed_length(30)
    else:
        raise NotImplementedError(f'Data set {data_set_id} unknown.')

    ds.cache_audios()
    return ds
