import os, glob, sys
import librosa
import librosa.display
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import scale
from collections import Counter
from pyts.image import GramianAngularField
import warnings

# Emotion class mapping codes
CODES = {'W':'Anger', 'L':'Boredom', 'E':'Disgust', 'A':'Fear',
        'F':'Happiness', 'T':'Sadness', 'N':'Neutral'}

## helper function to generate graph_data for graph type
def preprocess_audio(opt, y, sr):
    if opt == 'spect':
        D = librosa.stft(y, n_fft=480, hop_length=160, win_length=480, window='hamming')
        spect,phase = librosa.magphase(D)
        graph_data = log_spect = np.log(spect)

    elif opt == 'mp_spect':
        S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
        graph_data = log_S = librosa.power_to_db(S, ref=np.max)

    elif opt == 'cqt':
        graph_data = C = librosa.cqt(y,sr)

    elif opt == 'chrom':
        C = np.abs(librosa.cqt(y,sr))
        graph_data = chroma = librosa.feature.chroma_cqt(C=C,sr=sr)

    elif opt == 'mfcc':
        raw_mfcc = librosa.feature.mfcc(y=y,sr=sr)
        graph_data = scaled_mfcc = scale(raw_mfcc, axis=1)

    elif opt == 'gasf':
        gasf = GramianAngularField(image_size=256, method='summation')
        graph_data = np.squeeze(gasf.fit_transform(y.reshape(1,-1)))

    elif opt == 'gadf':
        gadf = GramianAngularField(image_size=256, method='difference')
        graph_data = np.squeeze(gadf.fit_transform(y.reshape(1,-1)))

    else:
        raise NotImplementedError('Graph data not known for {}'.format(opt))

    return graph_data

def get_splits(y, sr, n=5):
    ixs = np.random.randint(0, high=len(y)-sr, size=n)
    splits = [y[i:i+sr] for i in ixs]
    return splits

## context manager for pyplot memory management when saving figures
class FigManager():

    cmap = cm.get_cmap('viridis')
    verbose_kwargs = {
        'spect':('time','linear','%+2.0f dB'),
        'mp_spect':('time','mel','%+2.0f dB'),
        'cqt':('time','cqt_note','%+2.0f dB'),
        'chrom':('time','chroma',None),
        'mfcc':('time',None,None)
    }
    dpi = 256

    def __init__(self, f, opt, y=None, sr=None, show=True, shape=None, verbose=None, dest=None, ext=None):
        self.f = f
        self.opt = opt

        if any([i is None for i in [y,sr]]):
            self.y, self.sr = librosa.load(self.f)
        else:
            self.y,self.sr = y,sr

        self.show = show
        self.shape = shape
        self.verbose = verbose
        self.dest = dest
        self.ext = ext

        self.fig, self.ax = plt.subplots()

        if self.verbose:
            x_lab,y_lab,bar_fmt = self.verbose_kwargs.get(self.opt, (None,None,None))
            self.kwargs = dict(zip(('x_axis', 'y_axis'), (x_lab,y_lab)))
            self.bar_fmt = bar_fmt

        try:
            self.graph_data = preprocess_audio(self.opt, self.y, self.sr)
        except NotImplementedError:
            warnings.warn('Warning: graph data not known for {} data type'.format(self.opt))
            self.graph_data = None

    def __enter__(self):
        return self

    def __exit__(self, *excs):
        if self.verbose:
            self.fig.colorbar(self.img, format=self.bar_fmt, ax=self.ax)
        else:
            plt.axis('off')

        if self.shape:
            assert self.fig == plt.gcf()
            self.fig.set_size_inches(*self.shape)

        if self.show:
            plt.show()

        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.ax.axis('off')
        plt.axis('off')

        if self.dest:
            basename = os.path.basename(self.f)
            if self.shape:
                assert self.fig == plt.gcf()
                self.fig.set_size_inches(*self.shape)
            self.ext = '_{0:02d}'.format(self.ext) if self.ext else ''
            self.fig.savefig(self.dest + CODES[basename[5]] + '/' + basename[:-4] + self.ext + '.png',
                             dpi=self.dpi, frameon=False)

        self.fig.clear()
        plt.close(self.fig)
        del self.fig
