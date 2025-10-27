import multiprocessing
import os
import subprocess

from datasets.wavcaps import get_freesound_subset


def transform(params, sr=32000, codec='mp3'):
    i, file = params
    # taken from https://github.com/kkoutini/PaSST/blob/main/audioset/prepare_scripts
    fname = os.path.basename(file)
    fdir = os.path.basename(os.path.dirname(file))
    tdir = os.path.join('~/Audio_Datasets/wavcaps/mp3', fdir)
    os.makedirs(tdir, exist_ok=True)
    # target_file = os.path.join(get_persistent_cache_dir(), f'{i}.{codec}')
    target_file = os.path.join(tdir, fname[:-4] + codec)
    if not os.path.exists(target_file):
        print(i, file)
        print(target_file)
        subprocess.run(['ffmpeg', '-hide_banner', '-nostats', '-loglevel', 'error', '-n', '-i', file,
                        '-codec:a', codec, '-ar', f'{sr}', '-ac', '1', target_file])


if __name__ == '__main__':
    files = get_freesound_subset('~/Audio_Datasets/wavcaps')
    paths = [(i, s["path"]) for i, s in enumerate(files)]
    print("Files:", len(paths))

    # compress and load files
    with multiprocessing.Pool(processes=10) as pool:
        pool.map(transform, paths)
