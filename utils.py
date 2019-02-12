import os, datetime
import numpy as np
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
    aps = fg.get_all_data('aperiodic_params') # get aperiodic parameters
    pks = fg.get_all_data('peak_params')
    err = fg.get_all_data('error')
    r2s = fg.get_all_data('r_squared')
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
