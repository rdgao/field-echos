import os, sys
import numpy as np
from scipy import io, signal
from neurodsp import spectral
from fooof import FOOOFGroup
from fooof.funcs import fit_fooof_group_3d, combine_fooofs
import pandas as pd

sys.path.append('..')
import utils

def main(argv):
    # defining basepaths
    basepath = '/Users/rdgao/Documents/data/CRCNS/fcx1/'
    rec_dirs = [f for f in np.sort(os.listdir(basepath)) if os.path.isdir(basepath+f)]
    result_basepath = '/Users/rdgao/Documents/code/research/field-echos/results/fcx1/wakesleep/'

    if 'do_psds' in argv:
        print('Computing PSDs...')

        for cur_rec in range(len(rec_dirs))[21:]:
            print(rec_dirs[cur_rec])
            # compute PSDs
            psd_path = result_basepath + rec_dirs[cur_rec] + '/psd/'

            # load data
            ephys_data = io.loadmat(basepath+rec_dirs[cur_rec]+'/'+rec_dirs[cur_rec]+'_ephys.mat', squeeze_me=True)
            behav_data = pd.read_csv(basepath+rec_dirs[cur_rec]+'/'+rec_dirs[cur_rec]+'_wakesleep.csv', index_col=0)

            # get some params
            nchan,nsamp = ephys_data['lfp'].shape
            fs = ephys_data['fs']
            ephys_data['t_lfp'] = np.arange(0,nsamp)/fs
            elec_region = np.unique(ephys_data['elec_regions'])[0]

            # get subset of behavior that marks wake and sleep
            behav_sub = behav_data[behav_data['Label'].isin(['Wake', 'Sleep'])]

            # name, nperseg, noverlap, f_range, outlier_pct
            p_configs = [['1sec', int(fs), int(fs/2), [0., 200.], 5],
                            ['5sec', int(fs*5), int(fs*4), [0., 200.], 5]]

            for p_cfg in p_configs:
                # parameter def
                print(p_cfg)
                saveout_path = psd_path+ p_cfg[0]
                nperseg, noverlap, f_range, outlier_pct = p_cfg[1:]

                psd_mean, psd_med,  = [], []
                for ind, cur_eps in behav_sub.iterrows():
                    # find indices of LFP that correspond to behavior
                    lfp_inds = np.where(np.logical_and(ephys_data['t_lfp']>=cur_eps['Start'],ephys_data['t_lfp']<cur_eps['End']))[0]

                    # compute mean and median welchPSD
                    p_squished = spectral.compute_spectrum(ephys_data['lfp'][:,lfp_inds], ephys_data['fs'], method='welch',avg_type='mean', nperseg=nperseg, noverlap=noverlap, f_range=f_range, outlier_pct=outlier_pct)
                    f_axis, cur_psd_mean = p_squished[0,:], p_squished[1::2,:] # work-around for ndsp currently squishing together the outputs
                    p_squished = spectral.compute_spectrum(ephys_data['lfp'][:,lfp_inds], ephys_data['fs'], method='welch',avg_type='median', nperseg=nperseg, noverlap=noverlap, f_range=f_range, outlier_pct=outlier_pct)
                    f_axis, cur_psd_med = p_squished[0,:], p_squished[1::2,:]

                    # append to list
                    psd_mean.append(cur_psd_mean)
                    psd_med.append(cur_psd_med)

                # collect, stack, and save out
                psd_mean, psd_med, behav_info = np.array(psd_mean), np.array(psd_med), np.array(behav_sub)
                save_dict = {}
                for name in ['psd_mean', 'psd_med','nperseg','noverlap','fs','outlier_pct', 'behav_info', 'elec_region', 'f_axis']:
                    save_dict[name] = eval(name)
                utils.makedir(saveout_path, timestamp=False)
                np.savez(file=saveout_path+'/psds.npz', **save_dict)

    if 'do_fooof' in argv:
        fooof_settings = [['knee', 4, (0.1,200)],
                            ['fixed', 4, (0.1,200)],
                            ['fixed', 2, (0.1,10)],
                            ['fixed', 2, (30,55)]]
        for cur_rec in range(len(rec_dirs)):
            print(rec_dirs[cur_rec])
            psd_path = result_basepath + rec_dirs[cur_rec] + '/psd/'
            for psd_win in ['1sec/', '5sec/']:
                psd_folder = psd_path+psd_win
                psd_data = np.load(psd_folder+'psds.npz')
                for psd_mode in ['psd_mean', 'psd_med']:
                    for f_s in fooof_settings:
                        fg = FOOOFGroup(aperiodic_mode=f_s[0], max_n_peaks=f_s[1])
                        fgs = fit_fooof_group_3d(fg, psd_data['f_axis'], psd_data[psd_mode], freq_range=f_s[2])
                        fg_all = combine_fooofs(fgs)
                        fooof_savepath = utils.makedir(psd_folder, '/fooof/'+psd_mode+'/', timestamp=False)
                        fg_all.save('fg_%s_%ipks_%i-%iHz'%(f_s[0],f_s[1],f_s[2][0],f_s[2][1]), fooof_savepath, save_results=True, save_settings=True)


if __name__ == "__main__":
   main(sys.argv[1:])
