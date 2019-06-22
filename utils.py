import os, datetime
import numpy as np
import scipy as sp
import pandas as pd
from fooof import FOOOFGroup, synth
import matplotlib.pyplot as plt

def makedir(base, itm='/', timestamp=True):
    """
    Utility for checking and making a directory, with an option of creating
    a final level folder that is named the date and hour at runtime.
    """
    now = datetime.datetime.now().strftime("%Y%m%d%H%M")
    if timestamp:
        saveout_path = base+itm+now+'/'
    else:
        saveout_path = base+itm

    if os.path.exists(saveout_path) is False:
        print(saveout_path)
        os.makedirs(saveout_path)
    return saveout_path

def return_fooof_regen(fg):
    """
    Takes a fitted FOOOFGroup model and returns the fitted (modeled) PSDs in
    linear power.
    """
    f_regen = synth.gen_freqs(fg.freq_range, fg.freq_res)
    n_psds = fg.get_params('error').shape[0]
    psds_regen = 10**np.array([fg.get_fooof(ind,regenerate=True).fooofed_spectrum_ for ind in range(n_psds)])
    return f_regen, psds_regen


def return_fg_fits(fg_file, fg_folder):
    """
    Return fitted parameters from FOOOFGroup, in the following order:
    aperiodic, peaks, error, r-squared.
    """
    fg = FOOOFGroup()
    fg.load(fg_file, fg_folder)
    aps = fg.get_params('aperiodic_params') # get aperiodic parameters
    # need to resave the old fooof files for this loading to work 
    #pks = fg.get_params('peak_params')
    pks = []
    err = fg.get_params('error')
    r2s = fg.get_params('r_squared')
    return aps, pks, err, r2s

def convert_knee_val(knee, exponent=2.):
    """
    Convert knee parameter to frequency and time-constant value.
    Can operate on array or float.

    Default exponent value of 2 means take the square-root, but simulation shows
    taking the exp-th root returns a more accurate drop-off frequency estimate
    when the PSD is actually Lorentzian.
    """
    knee_freq = knee**(1./exponent)
    knee_tau = 1./(2*np.pi*knee_freq)
    return knee_freq, knee_tau

# def compute_aggregate(df, aggr_func, group_col, loud=False):
#     """
#     Compute one- or two-level aggregate over the dataframe.
#     """
#     if len(group_col)==2:
#         if loud: print('-> aggregating over %s.'%group_col[0])
#         # group by specified column first and aggregate across specified types
#         group_agg = pd.DataFrame()
#         for group_id, group in df.groupby(group_col[0][0]):
#             group_agg = group_agg.append(group.groupby(group_col[0][1]).aggregate(aggr_func))
#
#         if loud: print('---> aggregating over %s.'%group_col[1])
#         grand_agg = group_agg.groupby(group_col[1])
#
#     elif len(group_col)==1 or type(group_col) is str:
#         if type(group_col) is str:
#             group_col = [group_col]
#         # no first-level grouping, directly average across all electrodes
#         if loud: print('-> aggregating over %s.'%group_col[0])
#         group_agg = None
#         grand_agg = df.groupby(group_col[0])
#     return grand_agg, group_agg
#
# def compute_avg_sem(grouped_by, feature, aggr_func):
#     """
#     Computes group average via aggr_func, and standard error.
#     grouped_by must be a grouped_by dataframe
#     """
#     agg_avg = grouped_by[feature].apply(eval(aggr_func))
#     agg_sem = grouped_by[feature].apply(np.nanstd)/(grouped_by[feature].count()**0.5)
#     return agg_avg, agg_sem

def plot_psd_fits(f_axis, psds, chan, fgs, fg_labels):
    C_ORD = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # plot data PSD
    plt.loglog(f_axis, np.squeeze(psds[chan,:]), 'k', alpha=0.2,lw=4)
    YL=plt.ylim()
    for i_f, fg in enumerate(fgs):
        # regen and plot fits
        f_regen, psds_regen = return_fooof_regen(fg)
        plt.loglog(f_regen, np.squeeze(psds_regen[chan,:]), alpha=0.5,lw=2, label=fg_labels[i_f])
        # retrieve and plot peaks if any
        pk_params = fg.get_params('gaussian_params')
        pk_inds = np.where(pk_params[:,-1]==chan)[0]
        for i_p in pk_inds:
            plt.plot([pk_params[i_p,0]]*2, YL, '-', color=C_ORD[i_f],alpha=0.5, lw=0.25)
            plt.plot(pk_params[i_p,0], YL[1], 'o', color=C_ORD[i_f],alpha=0.5, lw=0.25, ms=8)

        # plot knee location if any
        ap_params = fg.get_params('aperiodic_params')
        if ap_params.shape[1]==3:
            k_freq, _ = convert_knee_val(ap_params[chan,1],ap_params[chan,2])
            plt.plot([k_freq]*2, YL, '-', color=C_ORD[i_f],alpha=0.5, lw=0.25)
            plt.plot(k_freq, YL[1], 'v', color=C_ORD[i_f],alpha=0.5, lw=0.25, ms=8)

    plt.legend(frameon=False, fontsize=10, loc='lower left')
    plt.tick_params(axis='y',which='both',labelleft=False)
    plt.xlabel('Frequency (Hz)'), plt.ylabel('Power'), plt.tight_layout()
    remove_spines(plt.gca())

def remove_spines(axis, bounds=['right', 'top']):
    for b in bounds:
        axis.spines[b].set_visible(False)


def acf_events(spks, t_max, dt=0.001, norm=True, remove_zero=False):
    # get time bins on positive side and reflect around 0
    # lazy way of getting symmetric bin edges
    # this needs to be more clever when handling gigantic spiketrains
    bins = np.sort(np.concatenate((-np.arange(dt/2,t_max+dt,dt),np.arange(dt/2,t_max+dt,dt))))
    ac_times = []
    for i, spk in enumerate(spks):
        # a little hack: only get the positive side and just reflect to save time
        # this only works for acorr, not xcorr
        del_t = spks[i:]-spk
        ac_times.append(del_t[np.abs(del_t)<=t_max])

    ac_times = np.concatenate(ac_times)
    if remove_zero:
        ac_times = ac_times[ac_times!=0]
    ac, t_ac = np.histogram(ac_times, bins=bins)

    # fill the negative side
    ac[:int(np.floor(len(ac)/2))] = ac[-int(np.floor(len(ac)/2)):][::-1]
    t_ac = t_ac[1:]-dt/2
    # normalize by total spike count
    if norm: ac = ac/len(spks)
    return t_ac, ac

def bin_spiketrain(spike_times, dt, t_bounds=None):
    """
    Binarize spike train using an array of spike times and a given time interval.
    Note that this will return spike counts in time, i.e., some bins can have values
    greater than 1.

    Parameters
    ----------
    spike_times : 1D np array or list
        Array of spike times.
    dt : float
        Bin size in seconds.
    t_bounds : [float, float] (optional)
        Start and end time of binarized array, defaults to timestamp of first and last spike.

    Returns
    -------
    t_spk:
        Time vector of corresponding binarized spike times.
    spk_binned:
        Binarized spike times.

    """    # This method computes the bin index each spike falls into and
    # increases counts in those bins accordingly. Very fast.
    if t_bounds is None:
        t_bounds = [np.floor(spike_times[0]), np.ceil(spike_times[-1])]

    t_spk = np.arange(t_bounds[0],t_bounds[1],dt)
    spk_binned = np.zeros_like(t_spk)
    spk_inds, spk_cnts = np.unique(np.floor((spike_times-t_bounds[0])/dt).astype(int), return_counts=True)
    spk_binned[spk_inds] = spk_cnts
    return t_spk, spk_binned

def spikes_as_list(spike_train_array, df_spkinfo):
    """
    Takes spike time array and cut them into a list of arrays per cell.
    This is all just to deal with the fact that I wanted to somehow merge
        the spiketimes into one array during the conversion from the original data format.

    Parameters
    ----------
    spike_train_array : np array
    df_spkinfo : pandas dataframe

    Returns
    -------
    spikes_lise : list of np arrays
    """

    spikes_list = []
    for cell, df_cell in df_spkinfo.iterrows():
        spikes_list.append(spike_train_array[df_cell['spike_start_ind']:df_cell['spike_end_ind']])
    return spikes_list
