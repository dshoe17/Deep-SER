import os, sys, glob
import multiprocess as mp
from collections import Counter
from graph_audio import *

def main():
	root = './'
	path = sys.argv[1]

	counts = Counter([get_category(os.path.basename(i)) for i in glob.glob(path + '*.wav')])
	resample_table = {i:500/j for i,j in counts.items()}

	dest = root + 'resamples/'

	if not os.path.exists(dest):
	  os.makedirs(dest)
	  for i in counts:
	    os.makedirs(dest + i + '/')

	for i in counts:
	  if not os.path.exists(dest + i + '/'):
	    os.makedirs(dest + i + '/')

	def worker(graph_audio=graph_audio):
	  for ix, seg in enumerate(splits):
	    graph_audio(f, 'mp_spect', y=seg, sr=sr, shape=(251/256,128/256),
	               verbose=False, dest=dest, show=False, ext=ix+1)

	for i, f in enumerate(glob.glob(path + '*')):
	    cat = get_category(os.path.basename(f))
	    print(i)
	    frac, whole = math.modf(resample_table[cat])
	    n_samp = int(whole + np.random.binomial(1,frac))
	    y,sr = librosa.load(f)
	    splits = get_splits(y,sr,n_samp)
	    proc = mp.Process(target=worker)
	    proc.daemon=True
	    proc.start()
	    proc.join()

if __name__ == '__main__':
	main()
