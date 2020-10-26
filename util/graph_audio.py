import os, sys, glob
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import IPython.display as ipd
import plotly.express as px
import statsmodels.api as sm
import math
from sklearn.preprocessing import scale
from matplotlib import cm
from PIL import Image
import multiprocess as mp


## Function to generate spectrograms from .wav files
def get_spectrogram(wav):
	D = librosa.stft(wav, n_fft=480, hop_length=160, win_length=480, window='hamming')
	# D = librosa.stft(wav, n_fft=2048, hop_length=160, win_length=480, window='hamming')  # alternative parameters
	spect,phase = librosa.magphase(D)
	return spect


def get_category(f):
    codes = {'W':'Anger', 'L':'Boredom', 'E':'Disgust', 'A':'Fear',
          'F':'Happiness', 'T':'Sadness', 'N':'Neutral'}

    return codes[f[5]]


## All-encompassing audio graphing function
def graph_audio(f, opt, y=None, sr=None, show=True, shape=None, dest=None, ext=None, verbose=True):
    '''
    This function generates various audio representation graphs for specified .wav files
    (or given audio time series and sampling rate values). It also accepts an optional parameter
    to save the generated graphs to categorized directories based on the corresponding emotion
    conveyed in the audio sample.

    Args:
        f (str): the absolute path to the input .wav file
        opt (str): the type of audio graph representation to be generated ("spect" => spectrogram,
                   "mp_spect" => mel-power spectrogram, "cqt" => constant-Q transform, "chrom" => chromagram,
                   "mfcc" => MFCC intensity values)
        y (np.ndarray): supplied audio time series; optional
        sr (int): supplied sampling rate of audio time series y; optional
        show (bool): specifies whether or not to show the resulting graph (default is True, which always
                     depicts the resulting graph)
        shape (tuple(int, int)): the dimensions (in inches) of the image to display
        dest (str): if a value is given, this will serve as the path of the root directory to write to (default
                    value is None, which does not save the resulting graph)
        ext (int): if supplied, adds "..._<ext>.png" to saved audio file
        verbose (bool): specifies whether or not to add axis labels, ticks, and colorbars to resulting plots
                        (default value is True, which adds the aforementioned details)

    Returns:
        None (function may display a graph and / or save resulting graph file to a specified directory)
    '''
	if None in [y,sr]:
        y, sr = librosa.load(f)
    cmap = cm.get_cmap('viridis')

    # Spectrogram
    if opt == 'spect':
        log_spect = np.log(get_spectrogram(y))

        if verbose:
            librosa.display.specshow(log_spect, sr=sr, x_axis='time', y_axis='linear', cmap=cmap)
            plt.colorbar(format='%+2.0f dB')
        else:
            fig, ax = plt.subplots(1)
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            ax.axis('off')
            librosa.display.specshow(log_spect, sr=sr, cmap=cmap)
            plt.axis('off')

    # Mel Power Spectrogram
    elif opt == 'mp_spect':
        S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
        log_S = librosa.power_to_db(S, ref=np.max)

        if verbose:
            librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel', cmap=cmap)
            plt.colorbar(format='%+2.0f dB')
        else:
            fig, ax = plt.subplots(1)
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            ax.axis('off')
            librosa.display.specshow(log_S, sr=sr, cmap=cmap)
            plt.axis('off')

    # Constant-Q Transform
    elif opt == 'cqt':
        C = librosa.cqt(y, sr)

        if verbose:
            librosa.display.specshow(librosa.amplitude_to_db(C**2),
                                     x_axis='time', y_axis='cqt_note', cmap=cmap)
            plt.colorbar(format='%+2.0f dB')
        else:
            fig,ax = plt.subplots(1)
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            ax.axis('off')
            librosa.display.specshow(librosa.amplitude_to_db(C**2), cmap=cmap)
            plt.axis('off')

    # Chromagram
    elif opt == 'chrom':
        C = np.abs(librosa.cqt(y, sr))
        chroma = librosa.feature.chroma_cqt(C=C, sr=sr)

        if verbose:
            librosa.display.specshow(chroma, x_axis='time', y_axis='chroma', cmap=cmap)
            plt.colorbar()
        else:
            fig,ax = plt.subplots(1)
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            ax.axis('off')
            librosa.display.specshow(chroma, cmap=cmap)
            plt.axis('off')

    # MFCC Intensity
    elif opt == 'mfcc':
        raw_mfcc = librosa.feature.mfcc(y=y,sr=sr)
        scaled_mfcc = scaled = scale(raw_mfcc, axis=1)

        if verbose:
            librosa.display.specshow(scaled, sr=sr, x_axis='time', cmap=cmap)
            plt.colorbar()

        else:
            fig, ax = plt.subplots(1)
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            ax.axis('off')
            librosa.display.specshow(scaled, sr=sr, cmap=cmap)
            plt.axis('off')

    if shape:
        fig = plt.gcf()
        dpi = 256
        fig.set_size_inches(*shape)

    if show:
        plt.show()

    if dest:
        basename = os.path.basename(f)
        if shape:
            fig.set_size_inches(*shape)
        ext = '_{0:02d}'.format(ext) if ext else ''
        fig.savefig(dest + get_category(basename) + '/' + basename[:-4] + ext + '.png',
         dpi=256, frameon=False)
        plt.close()


def get_splits(y, sr, n=5):
    ixs = np.random.randint(0, high=len(y) - sr, size=n)
    splits = [y[i:i+sr] for i in ixs]
    return splits
