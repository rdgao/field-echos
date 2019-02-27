import os, sys

import numpy as np
from scipy import io, signal

from neurodsp import spectral
from fooof import FOOOFGroup

sys.path.append('..')
import utils

def compute_spectrum_w_holes(data, fs, window, nperseg, noverlap):
    f_axis,t_axis,spg = signal.spectrogram(data, fs, 'hann', nperseg, noverlap)
    psd_mean = spg[:,~np.all(spg==0, axis=0)].mean(1)
    psd_median = np.median(spg[:,~np.all(spg==0, axis=0)],axis=1)
    return f_axis, psd_mean, psd_median

def compute_psds_whole(data, fs, nperseg, noverlap):
    siglen, nchans = data.shape
    # compute psd and toss slices with holes
    psd_mean = np.zeros((int(nperseg/2+1), nchans))
    psd_med = np.zeros((int(nperseg/2+1), nchans))
    for chan in range(nchans):
        f_axis, psd_mean[:,chan], psd_med[:,chan] = compute_spectrum_w_holes(data[:,chan], fs, 'hann', nperseg, noverlap)
    return f_axis, psd_mean, psd_med

def main(argv):
    # defining paths
    basepath = '/Users/rdgao/Documents/data/MNI_rest/'
    datafile = basepath + 'WakefulnessMatlabFile.mat'
    result_basepath = '/Users/rdgao/Documents/code/research/field-echos/results/MNI_rest/'

    # load data
    data_dict = io.loadmat(datafile, squeeze_me = True)
    fs = data_dict['SamplingFrequency']
    data = data_dict['Data']

    if 'do_psds' in argv:
        print('Computing PSDs...')
        # 1 second window
        nperseg, noverlap = int(fs), int(fs/2)
        f_axis, psd_mean, psd_med = compute_psds_whole(data, fs, nperseg, noverlap)
        saveout_path = utils.makedir(result_basepath, 'psd/1sec/', timestamp=False)
        np.savez(saveout_path+'psds.npz', psd_mean=psd_mean, psd_med=psd_med,
            f_axis=f_axis, nperseg=nperseg, noverlap=noverlap)

        # 5 second window
        nperseg, noverlap = int(fs*5), int(fs*4)
        f_axis, psd_mean, psd_med = compute_psds_whole(data, fs, nperseg, noverlap)
        saveout_path = utils.makedir(result_basepath, '/psd/5sec/', timestamp=False)
        np.savez(saveout_path+'psds.npz', psd_mean=psd_mean, psd_med=psd_med,
            f_axis=f_axis, nperseg=nperseg, noverlap=noverlap)

    if 'do_fooof' in argv:
        print('FOOOFing...')
        fooof_settings = [['knee', 2, (1,55)],
                 ['fixed', 2, (1,55)],
                 ['fixed', 1, (1,10)],
                 ['fixed', 1, (30,55)]]

        for psd_win in ['1sec/', '5sec/']:
            psd_folder = result_basepath+'/psd/'+psd_win
            psd_data = np.load(psd_folder+'psds.npz')
            for f_s in fooof_settings:
                # fit to mean and median psd
                for psd_mode in ['psd_mean', 'psd_med']:
                    fg = FOOOFGroup(aperiodic_mode=f_s[0], max_n_peaks=f_s[1])
                    fg.fit(psd_data['f_axis'], psd_data[psd_mode].T, freq_range=f_s[2])
                    fooof_savepath = utils.makedir(psd_folder, '/fooof/'+psd_mode+'/', timestamp=False)
                    fg.save('fg_%s_%ipks_%i-%iHz'%(f_s[0],f_s[1],f_s[2][0],f_s[2][1]), fooof_savepath, save_results=True, save_settings=True)

    print('Done.')

if __name__ == "__main__":
   main(sys.argv[1:])
