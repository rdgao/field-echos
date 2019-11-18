import os, datetime, mne
import numpy as np
import scipy as sp
import pandas as pd
import nibabel as ni
from fooof import FOOOFGroup, sim
import matplotlib.pyplot as plt
from surfer import Brain
from fooof.sim import gen as synth

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


### ------ for MNI analyses and visualization
def compute_avg_dev(grouped_by, avg_dev_func):
    """
    Computes group average via aggr_func, and standard error.
    grouped_by must be a grouped_by dataframe
    """
    agg_avg = grouped_by.agg(avg_dev_func[0])
    agg_dev = grouped_by.agg(avg_dev_func[1])
    return agg_avg, agg_dev

def sort_per_lobe(sort_by_array, lobe_boundaries):
    """
    Return sorted indices, sorted by values in sort_by_array, given the constraint
    that sorting is done per lobe, i.e., values within frontal lobe are sorted independent
    of other regions.
    """
    reg_sorted_inds = []
    for i,l in enumerate(lobe_boundaries.T):
        reg_sorted_inds.append(sort_by_array[l[0]:l[1]].sort_values().index.values)
        #inverse_map.append(sort_by_array[l[0]:l[1]].ar sort_values().index.values)

    reg_sorted_inds = np.concatenate(reg_sorted_inds).astype(int)
    return reg_sorted_inds


def plot_agg_by_region(agg_avg, agg_dev, region_labels, group_agg=None, log_y=True, m_cfg=['o','k',10], sort_by='avg', plot_vert=False, shade_by_lobe=False):
    """
    Produce the aggregate per region plot.
    """
    C_ORD = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # compute lobe boundaries
    lobe_boundaries = np.sort(np.array([region_labels.groupby('Lobe')['Region #'].min(), region_labels.groupby('Lobe')['Region #'].max()]))

    # sort axis per lobe
    if sort_by is False:
        # unsorted indices, keep as original
        reg_sorted_inds = agg_avg.index.astype(int)
    else:
        # if 'avg', sort index by average value computed above
        # otherwise sort_by is a pre-computed 2-col dataframe with region as indices and metric as values
        if sort_by is 'avg': sort_by = agg_avg
        reg_sorted_inds = sort_per_lobe(sort_by, lobe_boundaries)

    # retrieve reverse mapping: given sorted ind
    reverse_inds_map = pd.DataFrame(agg_avg.index.values, index=reg_sorted_inds)

    # plot aggregate average & deviation
    if plot_vert:
        # plot vertically
        plt.errorbar(agg_avg[reg_sorted_inds], agg_avg.index, xerr=agg_dev[reg_sorted_inds], fmt=m_cfg[0]+m_cfg[1], ms=m_cfg[2], alpha=0.5)
    else:
        plt.errorbar(agg_avg.index, agg_avg[reg_sorted_inds], yerr=agg_dev[reg_sorted_inds], fmt=m_cfg[0]+m_cfg[1], ms=m_cfg[2], alpha=0.5)

    # plot per-subj average
    if group_agg is not None:
        for group_id, group in group_agg.groupby('region'):
            # plot per-patient regional average
            if plot_vert:
                plt.plot(group, [reverse_inds_map.loc[group_id.astype(int)]]*len(group), 'k.', color=m_cfg[1], alpha=0.5, label=group_id, ms=m_cfg[2]/2)
            else:
                plt.plot([reverse_inds_map.loc[group_id.astype(int)]]*len(group), group, 'k.', color=m_cfg[1], alpha=0.5, label=group_id, ms=m_cfg[2]/2)

    # label axes & ticks
    # note the minus one because reg_sorted_inds are in terms of regions (1-38), but the array is 0-indexed
    if plot_vert:
        plt.yticks(region_labels['Region #'].values, region_labels['Region name'].values[reg_sorted_inds-1]);
    else:
        plt.xticks(region_labels['Region #'].values, region_labels['Region name'].values[reg_sorted_inds-1], rotation=300, ha='left', rotation_mode='anchor');

    # setting log axis
    if log_y: plt.xscale('log') if plot_vert else plt.yscale('log')

    # setting axis limit
    ax_lim = [region_labels['Region #'].min()-0.5, region_labels['Region #'].max()+0.5]
    plt.ylim(ax_lim) if plot_vert else plt.xlim(ax_lim)

    # plot shading for lobe
    if shade_by_lobe:
        if plot_vert:
            XL = plt.xlim()
            for i,l in enumerate(lobe_boundaries.T):
                plt.fill_betweenx([l[0]-0.5,l[1]+0.5], XL[0], XL[1], alpha=0.15, color=C_ORD[i])
        else:
            YL = plt.ylim()
            for i,l in enumerate(lobe_boundaries.T):
                plt.fill_between([l[0]-0.5,l[1]+0.5], YL[0], YL[1], alpha=0.15, color=C_ORD[i])

    # erase the top and right of box
    remove_spines(plt.gca())
    plt.tight_layout()

def compute_distance_to_seed(df, lobe_seed_coors, seed_by_lobe=True, norm_by_lobe=False):
    """
    Compute Euclidian distance of every electrode position to the seed location in each lobe.
    """
    if seed_by_lobe:
        # grab seed coordinate based on primary sensory coordinate of that lobe
        seed_coor = lobe_seed_coors[df['lobe'].values.astype(int)]
        seed_dist = np.sum(np.abs(df[['x_pos','y','z']].values - seed_coor)**2,axis=-1)**0.5
    else:
        # otherwise, grab the closest seed
        # do a bunch of matrix gymnastics to compute distance to all 5 seeds in one line lol
        seed_dist = np.sum(np.abs(np.repeat(df[['x_pos','y','z']].values[:,:,None],lobe_seed_coors.shape[0],2) \
            - np.transpose(np.repeat(lobe_seed_coors[:,:,None], len(df_combined.index), -1), (2,1,0)))**2.,1)**0.5
        closest_lobe_id = seed_dist.argmin(1) # id of lobe containing closest seed
        seed_dist = seed_dist.min(1)

    if norm_by_lobe:
        normed_seed_dist = np.zeros_like(seed_dist)
        # normalize distance from seed across regions in each lobe individually
        for l in np.sort(np.unique(df['lobe'].values.astype(int))):
            # norm by zscore
            if norm_by_lobe is 'zscore':
                normed_seed_dist[df['lobe']==l] = sp.stats.zscore(seed_dist[df['lobe']==l])
            elif norm_by_lobe is 'max':
                normed_seed_dist[df['lobe']==l] = seed_dist[df['lobe']==l]/seed_dist[df['lobe']==l].max()

        seed_dist = normed_seed_dist

    df_out = df.copy()
    df_out.insert(len(df_out.columns), 'seed_dist', seed_dist)
    return df_out

def plot_from_seeddist(df_w_dist, feature, lobe_id, log_y=True, hold_axis=True):
    """
    Plot distance from primary area (seed) against the value of interest (feature).
    """
    C_ORD = plt.rcParams['axes.prop_cycle'].by_key()['color']
    xy_axis_frac = (0.45, 0.85)
    ax = plt.subplot(2,3,1)
    plt.plot(df_w_dist['seed_dist'], df_w_dist[feature], '.k', alpha=0.5, mec='none')
    if log_y: plt.yscale('log')
    remove_spines(plt.gca())
    rho, pv = sp.stats.spearmanr(df_w_dist['seed_dist'], df_w_dist[feature], nan_policy='omit')
    s = r'$\rho$ = %.3f '%rho + np.sum(pv<=np.array([0.05, 0.01, 0.001]))*'*'
    plt.annotate(s, xy=xy_axis_frac, xycoords='axes fraction', size=14)
    plt.xticks([]);plt.yticks([])
    plt.title('All')
    XL = plt.xlim(); YL = plt.ylim()

    for k,v in lobe_id.items():
        ax =plt.subplot(2,3,v+2)
        x,y = df_w_dist['seed_dist'][df_w_dist['lobe']==v], df_w_dist[feature][df_w_dist['lobe']==v]
        plt.plot(x,y, '.', color=C_ORD[v], alpha=0.7, mec='none')
        if log_y: plt.yscale('log')
        remove_spines(plt.gca())
        #ax.spines['right'].set_visible(False); ax.spines['top'].set_visible(False)
        if hold_axis:
            plt.xlim(XL);plt.ylim(YL)
            plt.xticks([]);plt.yticks([])

        rho, pv = sp.stats.spearmanr(x, y, nan_policy='omit')
        s = r'$\rho$ = %.3f '%rho + np.sum(pv<=np.array([0.05, 0.01, 0.001]))*'*'
        plt.annotate(s, xy=xy_axis_frac, xycoords='axes fraction', size=14)
        plt.title(k)

    plt.subplot(2,3,4)
    plt.xticks([0,1]);plt.yticks([0.01,0.1,1])
    plt.xlabel('Distance from Primary Area'); plt.ylabel(feature)
    plt.tight_layout()

def apply_affine(f_affine, coord, ijk2xyz):
    """
    Apply forward (index to MNI coor) or reverse (MNI to index) affine transformation.
    """
    M_aff, M_trsl = f_affine[:3,:3], f_affine[:3,-1]
    if ijk2xyz:
        # index to MNI coordinate
        return np.dot(M_aff,coord)+M_trsl
    else:
        # MNI coordinate to index
        return np.dot(np.linalg.inv(M_aff),coord-M_trsl)

def project_ecog_mmp(ecog_coors_transformed, MMP_sparse_coords, r_search=3, find_nearest=False):
    """
    Project ecog electrodes onto MMP based on the electrode's MNI coordinate.
    Finds all voxels within radius of r_search in MMP, where there's a non-zeros
    parcellation value, and return those voxel distances from the elec and indices
    """
    # build kd_tree and do a ball search around all electrodes
    kd_tree = sp.spatial.KDTree(MMP_sparse_coords)
    elec_mmp_proj = kd_tree.query_ball_point(ecog_coors_transformed, r_search)

    # find the distances and sort by that
    # also, fill any empties with closest point if given option
    dist_ind_pairs = []
    for p_i, projs in enumerate(elec_mmp_proj):
        if len(projs):
            dists = np.linalg.norm(MMP_sparse_coords[projs]-ecog_coors_transformed[p_i], axis=1)
            pair = np.vstack((dists[dists.argsort()],np.array(projs)[dists.argsort()])).T
        else:
            pair = np.array([])
            if find_nearest:
                pair = np.array(kd_tree.query(ecog_coors_transformed[p_i]))[None,:]

        dist_ind_pairs.append(pair)

    return dist_ind_pairs


def ecog_gene_corr(df_combined, df_anat, avg_dev_func, ephys_feat='tau', anat_feat='T1T2', group_patients=True, age_subset=None):
    """
    Return ephys and anatomy feature correlation and arrays for parcel average and per-electrode.
    """
    if group_patients:
        df_run = df_combined.groupby(['patient', 'gparcel'], as_index=False).agg(avg_dev_func[0])
    else:
        df_run = df_combined
    # if no age restricted indices are specified, take all values
    if age_subset is None: age_subset=df_run['age']>0
    # parcel-level average
    ephys_avg = df_run[age_subset].groupby('gparcel').agg(avg_dev_func[0])[ephys_feat]
    ephys_dev = df_run[age_subset].groupby('gparcel').agg(avg_dev_func[1])[ephys_feat]
    anat_avg = df_anat[anat_feat][ephys_avg.index]
    rho_agg, pv_agg = sp.stats.spearmanr(ephys_avg, anat_avg, nan_policy='omit')

    # return electrode level
    anat_elec = np.array([df_anat.iloc[int(p_i-1)][anat_feat] if ~np.isnan(p_i) else np.nan for p_i in df_run['gparcel'][age_subset]])
    ephys_elec = df_run[ephys_feat][age_subset]
    rho_elec, pv_elec = sp.stats.spearmanr(ephys_elec, anat_elec, nan_policy='omit')
    return ephys_avg, ephys_dev, anat_avg, (rho_agg, pv_agg), ephys_elec, anat_elec, (rho_elec, pv_elec)

def plot_MMP(data, save_file, minmax=None, cmap='inferno', alpha=1, add_border=False):
    """
    Plots arbitrary array of data onto MMP parcellation
    """
    ## Folloing this tutorial
    # https://github.com/nipy/PySurfer/blob/master/examples/plot_parc_values.py

    # I assume I will always be using this parcellation, at least for now
    subjects_dir = mne.datasets.sample.data_path() + '/subjects'
    annot_file = subjects_dir + '/fsaverage/label/lh.HCPMMP1.annot'
    mmp_labels, ctab, names = ni.freesurfer.read_annot(annot_file)

    if len(names)-len(data)==1:
        # short one label cuz 0 is unassigned in the data, fill with -1
        data_app = np.hstack((-1,data))
        vtx_data = data_app[mmp_labels]
    else:
        vtx_data = data[mmp_labels]
        vtx_data[mmp_labels < 1] = -1

    # plot brain
    brain = Brain('fsaverage', 'lh', 'inflated', subjects_dir=subjects_dir,
                  cortex='bone', background='white', size=800, show_toolbar=True, offscreen=True)#, views=['med', 'lat'])

    if add_border:
        brain.add_annotation((mmp_labels, np.array([[0,0,0, c[3], c[4]] for c in ctab])))


    if minmax is None:
        minmax = [np.min(data), np.max(data)]
    # add data
    brain.add_data(vtx_data, minmax[0], minmax[1], colormap=cmap, alpha=alpha)

    # save
    brain.show_view('med')
    brain.save_imageset(save_file, ['med', 'lat'], 'png')

### ------ for handling spiking analyses

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
