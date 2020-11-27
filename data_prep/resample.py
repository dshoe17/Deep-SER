import sys
import numpy as np
import math
from collections import Counter

from preprocess_audio import *

def main():
    path = sys.argv[1]
    dest = sys.argv[2]

    counts = Counter([CODES[os.path.basename(i)[5]] for i in glob.glob(path + '*.wav')])
    resample_table = {i:500/j for i,j in counts.items()}

    os.makedirs(dest, exist_ok=True)
    for i in counts:
        os.makedirs(dest + i + '/', exist_ok=True)

    for i,f in enumerate(glob.glob(path + '*.wav')):
        cat = CODES[os.path.basename(f)[5]]
        print(i)
        frac,whole = math.modf(resample_table[cat])
        n_samp = int(whole + np.random.binomial(1,frac))
        y,sr = librosa.load(f)
        splits = get_splits(y,sr,n_samp)
        for ix,seg in enumerate(splits):
            with FigManager(f, 'mp_spect', y=seg, sr=sr, shape=(251/256,128/256),
                            verbose=False, dest=dest, show=False, ext=ix+1) as fig:
                librosa.display.specshow(fig.graph_data, sr=fig.sr, cmap=fig.cmap)

if __name__=='__main__':
    main()
