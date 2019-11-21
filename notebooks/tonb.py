

#%%

%load_ext autoreload
%autoreload 2
%matplotlib inline
sys.path

from fooof import FOOOFGroup
from gen_utils import flat, Map, bound, reveal
from neurodsp import sim as nsim, spectral
from numpy import r_, s_
from seaborn import despine
from surfer import Brain
import echo_utils
import numpy as np
import scipy as sp
import sys, hdf5storage as hdf
from analysis.plotting import *

#%%md
"""
<h3 style="font-size:12px; text-transform: capitalize;">Simulated population poisson spiking of exponential-/alpha shaped neuronal spikes</h3>"""
#%%
"""
Set (sim) / Record (exp) Parameters"""
#%%


#Both:
sim, exp, f, t, tau, num = Map.create('sim', 'exp', 'f', 't', 'tau', 'num')
sim.fs = exp.fs = fs = 1000. # Hz
sim.t.period  = exp.t.period = t.period  = 3. # s
sim.f.range  = exp.f.range  = f.range = bound(2,200)

num.samples = bound(int(t.period / (1/fs))) # samples
t.onset = -0.5  # s : event max occurs approx. 500ms after event onset.

#Sim:
sim
tau.rise = 0
tau.decay = np.arange(0.005, 0.08, 0.0018) #synaptic decay constants
sim.t.axis = np.arange(0, t.period, 1 / fs)

#Exp:
exp.t.kernel_times = flat(hdf.loadmat('t.mat')['T'])
exp.t.axis = np.arange(t.onset, t.onset + t.period, 1 / fs)
num.kernels = bound(len(exp.t.kernel_times))

#%%md
"""
Get Synaptic Kernels (Sim) & Network Kernel Events (Exp)"""
#%%

#Sim:
sim.kernels = np.array([nsim.sim_synaptic_kernel(sim.t.period, fs, tau_r=tau.rise, tau_d=td) for td in tau.decay]).T

#Exp:
kernels = squeeze(hdf.loadmat('kernels.mat')['kernels'])
exp.kernels = np.zeros(np.array([num.samples, num.kernels]).astype(int))

for i, kernel in enumerate(kernels):
    for well in np.squeeze(kernel):
        if np.any(well):
            exp.kernels[:, i] += sum(well,1)[:num.samples]


#%%
#Plot:
fg = plt.figure()
g = fg.add_gridspec(2, 1)
c = getColors(num.kernels + 1); c.pop()
ax1, ax2 = axes = fg.add_subplot(g[0, 0]), fg.add_subplot(g[1, 0])

for i in num.kernels():
    ax1.plot(sim.t.axis[:100], sim.kernels[:100, i], color = c[i])
    ax2.plot(exp.t.axis, exp.kernels[:, i], color = c[i])
ax1.set_title("Simulation : Decaying Synaptic PSPs")
ax2.set_title("Experimental : Network Events");
fg.show()


#%%md
"""
Compute Spectra"""
#%%

#Exp:
spectra = squeeze(hdf.loadmat('PSDw.mat')['PSDw'])
exp.well.f.axis = bound(shape(squeeze(spectra[0])[-1])[0])
exp.well.psds = np.zeros(np.array([exp.well.f.axis, num.kernels]).astype(int))

for kernel, well_psd in enumerate(spectra):
    for well in squeeze(well_psd):
        if np.any(well):
            exp.well.psds[:, kernel] += sum(well,1)[:exp.well.f.axis]


#Both:
sim.f.axis = exp.f.axis = f.axis = np.fft.fftfreq(num.samples, 1/fs)[:int(np.floor(num.samples/2))]


sim.psds, exp.psds = list(
        map(lambda k: np.abs(sp.fft(k,axis=0)**2)[:len(sim.f.axis)], [sim.kernels, exp.kernels])) # Exp spectrum reflects spectral density w.r.t. the kernel of the 3s, population-synchronou, network spike event.

psds = Map.create('psds')
psds.update(
        sim=sim.psds,
        exp=exp.psds,
        well=copy(exp.
        well.psds))
psds.values()
psds.items()
#%%
#Plot:
fg = plt.figure()
g = fg.add_gridspec(3, 1)

f.axes = squeeze([sim.f.axis, exp.f.axis, exp.well.f.axis()])
axes = fg.add_subplot(g[0, 0]), fg.add_subplot(g[1, 0]), fg.add_subplot(g[2, 0])

for ax, f_axis, psd in zip(axes, f.axes, [sim.psds, exp.psds, exp.well.psds]):
    for i in num.kernels():
        # print([shape(ax), shape(f_axis), shape(psd)])
        ax.loglog(f.axis[f.range()], psd[f.range(), i])
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")

ax1, ax2, ax3 = axes
ax1.set_title("Synaptic Kernel PSD, Simulated")
ax2.set_title("Organoi Network Kernel PSD, Experimental");
ax3.set_title("Organoid Well PSD, Experimental");
fg.show()


#%%md
#
#Fit FOOOF & get knee parameter
#%%


def fit_knee(freqs, psd):
    fooof = FOOOFGroup(aperiodic_mode='knee', max_n_peaks=0)
    fooof.fit(freqs=freqs, power_spectra=psd.T, freq_range=f.range)
    knee_fit = fooof.get_params('aperiodic_params', 'knee')
    exp_fit = fooof.get_params('aperiodic_params', 'exponent')
    knee_freq, taus = echo_utils.convert_knee_val(knee_fit, exp_fit)
    knee_p = [psd[np.argmin(np.abs((f.axis[f.range()] - (knee_freq[i])))), i] for i in num.kernels()]
    return taus, knee_freq, knee_p

modes = Map.create("modes")
modes.update(dict(sim=sim.copy(), exp=exp.copy(), well=exp.well.copy()))
modes.values()
modes.keys()
sim
Map("sim", sim.copy())
from gen_utils import Map
modes.sim.name
modes.well
dir(modes)
for mode, freqs, psd in zip(modes.values(), f.axes, psds):
    mode.knee = Map("knee")
    taus, mode.knee.freq, mode.knee.p = fit_knee(freqs, psd)
    mode.knee.update(dict(
            taus = taus,
            freq = knee.freq,
            p = knee.p))
"""

Running FOOOFGroup across 8 power spectra.

FOOOF WARNING: Lower-bound peak width limit is < or ~= the frequency resolution: 0.50 <= 0.50
	Lower bounds below frequency-resolution have no effect (effective lower bound is the frequency resolution).
	Too low a limit may lead to overfitting noise as small bandwidth peaks.
	We recommend a lower bound of approximately 2x the frequency resolution.
"""




"""Spontaneous Synaptic Fluctuations"""
#%%md

> Using those synaptic kernels with varying decay constants, we can simulate synaptic fluctuations over time by simply convolving white noise (approximating Poisson population spiking) with the kernels. This will produce time series with varying autocorrelation time constants, and thus PSD knees. We will then FOOOF and fit the knee parameter, retrieve the time constant with the equation above, and confirm that we get back the parameter values we put in.
#%%md
#
# Simulate noise
#%%


T = 180
noise, ac = [], []
t_max = False

for t_d in tau.decay:
    _noise = nsim.sim_synaptic_current(T, fs, tau_d=t_d)
    noise.append(_noise)

    _ac = sp.signal.correlate(_noise, _noise)[int(T * fs) - 1:]
    ac.append(_ac)

    _max = len(_ac) if len(_ac) < len(_noise) else len(_noise)
    t_max = _max if not t_max or _max < t_max else t_max

for i, n in enumerate(noise):
    noise[i] = noise[i][:t_max]
    ac[i] = ac[i][:t_max]


# FOOOF PSDs without knee
f_axis, PSD = spectral.compute_spectrum(noise, fs)
taus, knee_freq, knee_p = fit_knee(f_axis, PSD.T)
N
t_axis = np.arange(0,T,1/fs)
plt.figure(figsize=(12,3))
plt.plot(t_axis[:int(fs)], noise.T[:int(fs),:], alpha=0.9)
plt.xlabel('Time (s)'); plt.ylabel('Voltage')
plt.xlim([0,1]);despine()
plt.tight_layout()




plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.plot(sim.t.axis[:500], ac[:500,:]/ac[0,:])
plt.xlabel('Time (s)'); plt.ylabel('Autocorrelation')

plt.subplot(1,3,2)
plt.loglog(f_axis[f.range()], PSD[:,f.range()].T);
plt.loglog(knee_freq,knee_p, 'ow', mec='k', mew=2)
plt.scatter(knee_freq,knee_p, c=color, s=50, edgecolor='k', zorder=100)
plt.xlabel('Frequency (Hz)'); plt.ylabel('Power Spectral Density')

plt.subplot(1,3,3)
plt.scatter(t_ds*1000,taus*1000, c=color, s=50, edgecolor='k', zorder=100)
plt.xlim([0,t_ds.max()*1200]);plt.ylim([0,t_ds.max()*1200])
plt.plot(plt.xlim(), plt.xlim(), 'k%% md', alpha=0.8);
plt.xlabel('PSP time constant (ms)'); plt.ylabel('Fit time constant (ms)')
[despine() for a in plt.gcf().axes]
plt.tight_layout()
