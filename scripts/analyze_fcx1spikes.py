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

        for cur_rec in range(len(rec_dirs)):
            print(rec_dirs[cur_rec])
            # compute PSDs
            psd_path = result_basepath + rec_dirs[cur_rec] + '/psd_spikes/'

            # load data
            ephys_data = io.loadmat(basepath+rec_dirs[cur_rec]+'/'+rec_dirs[cur_rec]+'_ephys.mat', squeeze_me=True)
            behav_data = pd.read_csv(basepath+rec_dirs[cur_rec]+'/'+rec_dirs[cur_rec]+'_wakesleep.csv', index_col=0)
            elec_region = np.unique(ephys_data['elec_regions'])[0]
            elec_shank_map = ephys_data['elec_shank_map']

            # some organization of spike meta datafile
            # NOTE that all this had to be done because I was an idiot and
            # organized the spikeinfo table and spikes in some dumb way
            # make spike info into df and access based on cell, and add end time
            df_spkinfo = pd.DataFrame(ephys_data['spike_info'], columns=ephys_data['spike_info_cols'])
            df_spkinfo.insert(len(df_spkinfo.columns)-1, 'spike_start_ind', np.concatenate(([0], df_spkinfo['num_spikes_cumsum'].iloc[:-1].values)))
            df_spkinfo.rename(columns={'num_spikes_cumsum': 'spike_end_ind'}, inplace=True)

            # this is now a list of N arrays, where N is the number of cells
            #    now we can aggregate arbitrarily based on cell index
            spikes_list = utils.spikes_as_list(ephys_data['spiketrain'], df_spkinfo)

            # pooling across populations from the same shanks
            df_spkinfo_pooled = df_spkinfo.copy()
            for g_i, g in df_spkinfo.groupby(['shank', 'cell_EI_type']):
                # super python magic that collapses all the spikes of the same pop on the same shank into one array
                spikes_list.append(np.sort(np.hstack([spikes_list[c_i] for c_i, cell in g.iterrows()])))
                # update spike info dataframe
                df_pop = pd.DataFrame({'shank': g['shank'].head(1),
                                       'cell_EI_type': g['cell_EI_type'].head(1),
                                       'num_spikes': g['num_spikes'].sum(),
                                       'cell_id': 0 })
                df_spkinfo_pooled = df_spkinfo_pooled.append(df_pop, ignore_index=True)

            # pooling across entire recording
            for g_i, g in df_spkinfo.groupby(['cell_EI_type']):
                spikes_list.append(np.sort(np.hstack([spikes_list[c_i] for c_i, cell in g.iterrows()])))
                df_pop = pd.DataFrame({'shank': 0,
                                       'cell_id': 0,
                                       'cell_EI_type': g['cell_EI_type'].head(1),
                                       'num_spikes': g['num_spikes'].sum()})
                df_spkinfo_pooled = df_spkinfo_pooled.append(df_pop, ignore_index=True)

            # save spikeinfo table to recording folder
            utils.makedir(psd_path, timestamp=False)
            df_spkinfo_pooled.to_csv(psd_path+'/spike_info.csv')

            ##### ------------- #####
            # compute PSDs across conditions and populations
            # individual cells
            dt = 0.005
            fs = 1/dt

            # name, nperseg, noverlap, f_range, outlier_pct
            p_configs = [['2sec', int(2*fs), int(2*fs*4/5)],
                            ['5sec', int(5*fs), int(5*fs*4/5)]]

            behav_sub = behav_data[behav_data['Label'].isin(['Wake', 'Sleep'])].reset_index()
            behav_info = np.array(behav_sub)
            num_block, num_cell = len(behav_sub), len(spikes_list)
            for p_cfg in p_configs:
                print(p_cfg)
                saveout_path = psd_path + p_cfg[0]
                nperseg, noverlap = p_cfg[1:]

                psd_mean = np.zeros((num_block, num_cell, int(p_cfg[1]/2+1)))
                psd_med = np.zeros((num_block, num_cell, int(p_cfg[1]/2+1)))
                for cell, spikes in enumerate(spikes_list):
                    print(cell,end='|')
                    for block, cur_eps in behav_sub.iterrows():
                        spikes_eps = spikes[np.logical_and(spikes>=cur_eps['Start'],spikes<cur_eps['End'])]
                        t_spk, spikes_binned = utils.bin_spiketrain(spikes_eps, dt, cur_eps[['Start', 'End']])
                        f_axis, psd_mean[block, cell, :] = spectral.compute_spectrum(spikes_binned, fs, method='welch',
                            avg_type='mean', nperseg=nperseg, noverlap=noverlap)
                        f_axis, psd_med[block, cell, :] = spectral.compute_spectrum(spikes_binned, fs, method='welch',
                            avg_type='median', nperseg=nperseg, noverlap=noverlap)

                # save PSDs and spike_info dataframe
                save_dict = {}
                for name in ['psd_mean', 'psd_med','nperseg','noverlap','fs', 'behav_info', 'elec_region', 'elec_shank_map', 'f_axis']:
                    save_dict[name] = eval(name)
                utils.makedir(saveout_path, timestamp=False)
                np.savez(file=saveout_path+'/psds.npz', **save_dict)


    if 'do_fooof' in argv:
        fooof_settings = [['fixed', 2, (.5,80)],
                            ['fixed', 1, (.5,5)],
                            ['fixed', 1, (10,20)],
                            ['fixed', 1, (30,80)]]


        for cur_rec in range(len(rec_dirs)):
            print(rec_dirs[cur_rec])
            psd_path = result_basepath + rec_dirs[cur_rec] + '/psd_spikes/'
            df_spkinfo_pooled = pd.read_csv(psd_path+'/spike_info.csv', index_col=0)

            # grab only the aggregate cells
            df_pops = df_spkinfo_pooled[df_spkinfo_pooled['cell_id']==0]
            df_pops.to_csv(psd_path+'/pop_spike_info.csv')

            for psd_win in ['2sec/', '5sec/']:
                psd_folder = psd_path+psd_win
                psd_data = np.load(psd_folder+'psds.npz')
                for psd_mode in ['psd_mean']:
                    psd_spikes = psd_data[psd_mode][:,df_pops.index.values,:]
                    if np.any(np.isinf(np.log(psd_spikes[:,:,0]))):
                        # if any PSDs are 0s, set it to ones
                        print('Null PSDs found.')
                        zero_inds = np.where(np.isinf(np.log(psd_spikes[:,:,0])))
                        psd_spikes[zero_inds]=1.

                    fg_all = []
                    for f_s in fooof_settings:
                        fg = FOOOFGroup(aperiodic_mode=f_s[0], max_n_peaks=f_s[1], peak_width_limits=(5,20))
                        fgs = fit_fooof_group_3d(fg, psd_data['f_axis'], psd_spikes, freq_range=f_s[2]);
                        fg_all = combine_fooofs(fgs)
                        fooof_savepath = utils.makedir(psd_folder, '/fooof/'+psd_mode+'/', timestamp=False)
                        fg_all.save('fg_%s_%ipks_%i-%iHz'%(f_s[0],f_s[1],f_s[2][0],f_s[2][1]), fooof_savepath, save_results=True, save_settings=True)


    print('Done.')

if __name__ == "__main__":
   main(sys.argv[1:])
