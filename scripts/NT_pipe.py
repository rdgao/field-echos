import sys
sys.path.append('/Users/rdgao/Documents/code/research/fooof/')
sys.path.append('/Users/rdgao/Documents/code/research/neurodsp/')
sys.path.append('/Users/rdgao/Documents/code/research/SpaceRecon/')
sys.path.append('/Users/rdgao/Documents/code/research/spectralCV/')

import numpy as np
import scipy as sp
from scipy import io
import os
from scv_funcs import access_nt, utils
import neurodsp as ndsp
from nlds import dfa, delayembed

# Import the FOOOF object
from fooof import FOOOF


def compute_psds(data, fs, nperseg, noverlap, winlen_roll, f_lim=200.):
    """
    Compute PSDs under various parameter settings.

    winlen_roll: float
        rolling window length in seconds
    """
    numchan = data.shape[0]

    # 1-sec PSD
    f_axis, t_axis, spg = sp.signal.spectrogram(data, fs, nperseg=nperseg, noverlap=noverlap)
    psd = spg.mean(axis=-1)

    # 1-sec rolling PSD
    win_roll = int(winlen_roll/(1-noverlap/nperseg))
    t_roll = np.array([ts[0] for ts in utils.yield_sliding_window_ts(t_axis, win_roll, 0)])
    psd_rolling = np.zeros((numchan, len(f_axis1), len(t_roll)))
    for chan in range(numchan):
        psd_rolling[chan,:,:] = np.array([chunk.mean(axis=-1) for chunk in utils.yield_sliding_window_ts(spg[chan], win_roll, 0)]).T

    # 10-sec PSD
    f_axis_L, psd_L = ndsp.spectral.psd(data, fs, nperseg=nperseg*10, noverlap=nperseg*5)

    f_ind = np.where(f_axis>f_lim)[0][0]
    f_L_ind = np.where(f_axis_L>f_lim)[0][0]
    return f_axis[:f_ind], psd[:,:f_ind], t_roll, psd_rolling[:,:f_ind,:], f_axis_L[:f_L_ind], psd_L[:,:f_L_ind]


def compute_dfa(data, fs, ac_max_lag=1000, dfa_n_scales=10, dfa_min_scale=0.01, dfa_max_scale=10):
    """
    Compute autocorrelation and DFA features.
    """
    numchan=data.shape[0]
    ACs = np.zeros((numchan,ac_max_lag))
    DFs = np.zeros((numchan,dfa_n_scales))
    alphas = np.zeros(numchan)
    for chan in range(numchan):
        # compute autocorr
        t_ac, ACs[chan,:] = delayembed.autocorr(data[chan,:],max_lag=ac_max_lag)
        # compute DFA
        t_scales, DFs[chan,:], alphas[chan] = dfa.dfa(data[chan,:],fs, dfa_n_scales, dfa_min_scale, dfa_max_scale)

    return ACs, t_scales, DFs, alphas

# def compute_nonlinear(data, fs, ac_max_lag=1000, dfa_n_scales=10, dfa_min_scale=0.01, dfa_max_scale=10, de_max_tau=1000, de_max_dim=5):
#     """
#     Compute nonlinear features: autocorrelation & DFA and delay embedding dimension.
#     """
#     numchan=data.shape[0]
#     ACs = np.zeros((numchan,ac_max_lag))
#     DFs = np.zeros((numchan,dfa_n_scales))
#     alphas = np.zeros(numchan)
#     dMIs = np.zeros((numchan,de_max_tau))
#     tau_MI = np.zeros(numchan)
#     pfnns = np.zeros((numchan, de_max_dim))
#     recon_dims = np.zeros(numchan)
#     for chan in range(numchan):
#         print('AC')
#         # compute autocorr
#         t_ac, ACs[chan,:] = delayembed.autocorr(data[chan,:],max_lag=ac_max_lag)
#         print('DFA')
#         # compute DFA
#         t_scales, DFs[chan,:], alphas[chan] = dfa.dfa(data[chan,:],fs, dfa_n_scales, dfa_min_scale, dfa_max_scale)
#
#         # compute MI
#         print('MI')
#         tMI, dMIs[chan,:] = delayembed.compute_delay_MI(data[chan,:],50,de_max_tau)
#         opt_delay = delayembed.find_valley(dMIs[chan,:])
#         tau_MI[chan] = opt_delay[0]
#
#         print('DE')
#         # compute delay-embedding pfnn
#         recon_dims[chan], pfnns[chan,:] = delayembed.pfnn_de_dim(data[chan,:],tau=opt_delay[0],max_dim=de_max_dim)
#
#     return ACs, t_scales, DFs, alphas, dMIs, tau_MI, pfnns, recon_dims
