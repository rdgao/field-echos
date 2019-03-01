import sys, os
import numpy as np
from scipy import io
sys.path.append('/Users/rdgao/Documents/code/research/spectralCV/')
from scv_funcs import access_nt
import neurodsp as ndsp
from fooof import FOOOFGroup

sys.path.append('../')
import utils


def main(argv):

    # define data and result paths
    basepath = '/Users/rdgao/Documents/data/NeuroTycho/'
    result_basepath = '/Users/rdgao/Documents/code/research/field-echos/results/NeuroTycho/rest_anes/'

    datasets = ['Propofol/20120730PF_Anesthesia+and+Sleep_Chibi_Toru+Yanagawa_mat_ECoG128',
               'Propofol/20120731PF_Anesthesia+and+Sleep_George_Toru+Yanagawa_mat_ECoG128',
               'Propofol/20120802PF_Anesthesia+and+Sleep_Chibi_Toru+Yanagawa_mat_ECoG128',
               'Propofol/20120803PF_Anesthesia+and+Sleep_George_Toru+Yanagawa_mat_ECoG128',
                'Ketamine/20120719KT_Anesthesia+and+Sleep_Chibi_Toru+Yanagawa_mat_ECoG128',
                'Ketamine/20120724KT_Anesthesia+and+Sleep_George_Toru+Yanagawa_mat_ECoG128',
                'Ketamine/20120810KT_Anesthesia+and+Sleep_George_Toru+Yanagawa_mat_ECoG128',
                'Ketamine/20120813KT_Anesthesia+and+Sleep_Chibi_Toru+Yanagawa_mat_ECoG128'
               ]
    sess_append = '/Session%d/'
    session_indices = [(1,0,1), (1,2,3), (2,0,1), (2,1,2)]
    session_labels = ['EyesOpen', 'EyesClosed', 'Delivery', 'Anesthesia']

    if 'do_psds' in argv:
        for ds in datasets:
            print('------\n',ds)
            for i in range(len(session_indices)):
                session = session_indices[i]
                outfile = str(i)+'_'+(session_labels[i]+'_'+ds).replace('/','_').replace('+','_')
                print(outfile)

                # grab ECoG data
                indices = access_nt.get_cond(data_path+ds+sess_append, session[0], session[1], session[2])
                data = access_nt.get_ECoG(data_path+ds+sess_append, session[0], range(1,129), indices)
                fs = 1000.

                # compute PSDs
                psd_path = result_basepath + outfile + '/psd/'

                # 1Hz resolution psd
                saveout_path = utils.makedir(psd_path,'/1sec/', timestamp=False)
                nperseg, noverlap, f_lim, spg_outlier_pct = int(fs), int(fs/2), 200., 5
                f_axis, psd_mean = ndsp.spectral.compute_spectrum(data, fs, method='mean', nperseg=nperseg, noverlap=noverlap, f_lim=f_lim, spg_outlier_pct=spg_outlier_pct)
                f_axis, psd_med = ndsp.spectral.compute_spectrum(data, fs, method='median', nperseg=nperseg, noverlap=noverlap, f_lim=f_lim, spg_outlier_pct=spg_outlier_pct)
                save_dict = dict((name,eval(name)) for name in ['f_axis','psd_mean', 'psd_med','nperseg','noverlap','fs','spg_outlier_pct'])
                np.savez(file=saveout_path+'psd.npz', **save_dict)

                # 0.2Hz resolution psd
                saveout_path = utils.makedir(psd_path,'/5sec/', timestamp=False)
                nperseg, noverlap, f_lim, spg_outlier_pct = int(fs*5), int(fs*4), 200., 5
                f_axis, psd_mean = ndsp.spectral.compute_spectrum(data, fs, method='mean', nperseg=nperseg, noverlap=noverlap, f_lim=f_lim, spg_outlier_pct=spg_outlier_pct)
                f_axis, psd_med = ndsp.spectral.compute_spectrum(data, fs, method='median', nperseg=nperseg, noverlap=noverlap, f_lim=f_lim, spg_outlier_pct=spg_outlier_pct)
                save_dict = dict((name,eval(name)) for name in ['f_axis','psd_mean', 'psd_med','nperseg','noverlap','fs','spg_outlier_pct'])
                np.savez(file=saveout_path+'psd.npz', **save_dict)

    if 'do_fooof' in argv:
        fooof_settings = [['knee', 3, (1,70)],
                 ['fixed', 3, (1,70)],
                 ['fixed', 1, (1,10)],
                 ['fixed', 1, (30,70)]]

        session_resultpath = [result_basepath+f+'/' for f in os.listdir(result_basepath) if os.path.isdir(result_basepath+f)]
        print('FOOOFing...')
        for s in session_resultpath:
            for psd_win in ['1sec/', '5sec/']:
                psd_folder = s+'/psd/'+psd_win
                psd_data = np.load(psd_folder+'psd.npz')
                for f_s in fooof_settings:
                    # fit to mean and median psd
                    for psd_mode in ['psd_mean', 'psd_med']:
                        fg = FOOOFGroup(aperiodic_mode=f_s[0], max_n_peaks=f_s[1])
                        fg.fit(psd_data['f_axis'], psd_data[psd_mode], freq_range=f_s[2])
                        fooof_savepath = utils.makedir(psd_folder, '/fooof/'+psd_mode+'/', timestamp=False)
                        fg.save('fg_%s_%ipks_%i-%iHz'%(f_s[0],f_s[1],f_s[2][0],f_s[2][1]), fooof_savepath, save_results=True, save_settings=True)

    print('Done.')


if __name__ == "__main__":
   main(sys.argv[1:])
