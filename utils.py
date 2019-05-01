import os, datetime
import numpy as np
import scipy as sp
import pandas as pd
from fooof import FOOOFGroup


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
    f_regen = fg.freqs
    n_psds = fg.get_all_data('error').shape[0]
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
    pks = fg.get_params('peak_params')
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

def compute_aggregate(df, aggr_func, group_col, loud=False):
    """
    Compute one- or two-level aggregate over the dataframe.
    """
    if len(group_col)==2:
        if loud: print('-> aggregating over %s.'%group_col[0])
        # group by specified column first and aggregate across specified types
        group_agg = pd.DataFrame()
        for group_id, group in df.groupby(group_col[0][0]):
            group_agg = group_agg.append(group.groupby(group_col[0][1]).aggregate(aggr_func))

        if loud: print('---> aggregating over %s.'%group_col[1])
        grand_agg = group_agg.groupby(group_col[1])

    elif len(group_col)==1 or type(group_col) is str:
        if type(group_col) is str:
            group_col = [group_col]
        # no first-level grouping, directly average across all electrodes
        if loud: print('-> aggregating over %s.'%group_col[0])
        group_agg = None
        grand_agg = df.groupby(group_col[0])
    return grand_agg, group_agg

def compute_avg_sem(grouped_by, feature, aggr_func):
    """
    Computes group average via aggr_func, and standard error.
    grouped_by must be a grouped_by dataframe
    """
    agg_avg = grouped_by[feature].apply(eval(aggr_func))
    agg_sem = grouped_by[feature].apply(np.nanstd)/(grouped_by[feature].count()**0.5)
    return agg_avg, agg_sem

def remove_spines(axis, bounds=['right', 'top']):
    for b in bounds:
        axis.spines[b].set_visible(False)
