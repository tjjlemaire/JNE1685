# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2018-10-01 20:45:29
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-09-30 19:23:52

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PySONIC.utils import *
from PySONIC.core import NeuronalBilayerSonophore
from PySONIC.neurons import *
from PySONIC.postpro import computeSpikingMetrics
from PySONIC.plt import cm2inch, ActivationMap
from ExSONIC.core import SonicNode, ExtendedSonicNode


def saveFigsAsPDF(figs, figindex):
    cdir = os.path.dirname(os.path.abspath(__file__))
    figdir = os.path.join(cdir, 'figs')
    figbase = f'fig{figindex:02}'
    if not os.path.isdir(figdir):
        os.mkdir(figdir)
    for fname, fig in figs.items():
        fig.savefig(os.path.join(figdir, f'{figbase}{fname}.pdf'), transparent=True)


def getCWtitrations_vs_Fdrive(neurons, a, freqs, tstim, toffset, fpath):
    fkey = 'Fdrive (kHz)'
    freqs = np.array(freqs)
    if os.path.isfile(fpath):
        df = pd.read_csv(fpath, sep=',', index_col=fkey)
    else:
        df = pd.DataFrame(index=freqs * 1e-3)
    for neuron in neurons:
        if neuron not in df:
            pneuron = getPointNeuron(neuron)
            nbls = NeuronalBilayerSonophore(a, pneuron)
            for i, Fdrive in enumerate(freqs):
                logger.info('Running CW titration for %s neuron @ %sHz',
                            neuron, si_format(Fdrive))
                Athr = nbls.titrate(Fdrive, tstim, toffset)  # Pa
                df.loc[Fdrive * 1e-3, neuron] = Athr * 1e-3  # kPa
    df.sort_index(inplace=True)
    df.to_csv(fpath, sep=',', index_label=fkey)
    return df


def getAthr_vs_radius(pneuron, radii, Fdrive, tstim, toffset):
    akey = 'radius (nm)'
    radii = np.array(radii)
    Athrs = np.array([NeuronalBilayerSonophore(a, pneuron).titrate(Fdrive, tstim, toffset) for a in radii])  # Pa
    df = pd.DataFrame({akey: radii * 1e9, 'threshold amplitude (kPa)': Athrs * 1e-3})
    df.set_index(akey, inplace=True)
    return df


def getSims(outdir, neuron, a, queue):
    fpaths = []
    updated_queue = []
    pneuron = getPointNeuron(neuron)
    nbls = NeuronalBilayerSonophore(a, pneuron)
    for i, item in enumerate(queue):
        Fdrive, Adrive, tstim, toffset, PRF, DC, fs, method = item
        print(item)
        fcode = nbls.filecode(Fdrive, Adrive, tstim, toffset, PRF, DC, fs, method)
        fpath = os.path.join(outdir, '{}.pkl'.format(fcode))
        if not os.path.isfile(fpath):
            print(fpath, 'does not exist')
            item.insert(0, outdir)
            updated_queue.append(item)
        fpaths.append(fpath)
    if len(updated_queue) > 0:
        print(updated_queue)
        # pneuron = getPointNeuron(neuron)
        # nbls = NeuronalBilayerSonophore(a, pneuron)
        # batch = Batch(nbls.run, updated_queue)
        # batch.run(mpi=True)
    return fpaths


def getSpikingMetrics(xvar, xkey, data_fpaths, metrics_fpath):
    metrics = {}
    if os.path.isfile(metrics_fpath):
        logger.info(f'loading spiking metrics from file: "{metrics_fpath}"')
        metrics = pd.read_csv(metrics_fpath, sep=',')
    else:
        logger.warning(f'computing spiking metrics vs. {xkey} for {len(data_fpaths)} files')
        metrics = computeSpikingMetrics(data_fpaths)
        metrics[xkey] = pd.Series(xvar, index=metrics.index)
        metrics.to_csv(metrics_fpath, sep=',', index=False)
    return metrics


def plotSpikingMetrics(xvar, xlabel, metrics_dict, logscale=False, spikeamp=True, colors=None,
                       fs=8, lw=2, ps=4, figsize=cm2inch(7.25, 5.8)):
    ''' Plot the evolution of spiking metrics as function of a specific stimulation parameter. '''

    ls = {'full': 'o-', 'sonic': 'o--'}
    cdefault = {'full': 'silver', 'sonic': 'k'}

    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=figsize)
    ibase = 0 if spikeamp else 1
    axes[ibase].set_ylabel('Latency\n (ms)', fontsize=fs, rotation=0, ha='right', va='center')
    axes[ibase + 1].set_ylabel(
        'Firing\n rate (Hz)', fontsize=fs, rotation=0, ha='right', va='center')
    if spikeamp:
        axes[2].set_ylabel('Spike amp.\n ($\\rm nC/cm^2$)', fontsize=fs, rotation=0, ha='right',
                           va='center')
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if logscale:
            ax.set_xscale('log')
        for item in ax.get_yticklabels():
            item.set_fontsize(fs)
    for ax in axes[:-1]:
        ax.spines['bottom'].set_visible(False)
        ax.set_xticks([])
        plt.setp(ax.get_xticklabels(minor=True), visible=False)
        ax.get_xaxis().set_tick_params(which='minor', size=0)
        ax.get_xaxis().set_tick_params(which='minor', width=0)
    axes[-1].set_xlabel(xlabel, fontsize=fs)
    if not logscale:
        axes[-1].set_xticks([min(xvar), max(xvar)])
    for item in axes[-1].get_xticklabels():
        item.set_fontsize(fs)

    # Plot metrics for each neuron
    for i, neuron in enumerate(metrics_dict.keys()):
        full_metrics = metrics_dict[neuron]['full']
        sonic_metrics = metrics_dict[neuron]['sonic']
        c = colors[neuron] if colors is not None else cdefault

        # Latency
        rf = 10
        ax = axes[ibase]
        ax.plot(xvar, full_metrics['latencies (ms)'].values, ls['full'], color=c['full'],
                linewidth=lw, markersize=ps)
        ax.plot(xvar, sonic_metrics['latencies (ms)'].values, ls['sonic'], color=c['sonic'],
                linewidth=lw, markersize=ps, label=neuron)

        # Firing rate
        rf = 10
        ax = axes[ibase + 1]
        ax.errorbar(xvar, full_metrics['mean firing rates (Hz)'].values,
                    yerr=full_metrics['std firing rates (Hz)'].values,
                    fmt=ls['full'], color=c['full'], linewidth=lw, markersize=ps)
        ax.errorbar(xvar, sonic_metrics['mean firing rates (Hz)'].values,
                    yerr=sonic_metrics['std firing rates (Hz)'].values,
                    fmt=ls['sonic'], color=c['sonic'], linewidth=lw, markersize=ps)

        # Spike amplitudes
        if spikeamp:
            ax = axes[2]
            rf = 10
            ax.errorbar(xvar, full_metrics['mean spike amplitudes (nC/cm2)'].values,
                        yerr=full_metrics['std spike amplitudes (nC/cm2)'].values,
                        fmt=ls['full'], color=c['full'], linewidth=lw, markersize=ps)
            ax.errorbar(xvar, sonic_metrics['mean spike amplitudes (nC/cm2)'].values,
                        yerr=sonic_metrics['std spike amplitudes (nC/cm2)'].values,
                        fmt=ls['sonic'], color=c['sonic'], linewidth=lw, markersize=ps)

    # Adapt axes y-limits
    rf = 10
    for ax in axes:
        ax.set_ylim([np.floor(ax.get_ylim()[0] / rf) * rf, np.ceil(ax.get_ylim()[1] / rf) * rf])
        ax.set_yticks([max(ax.get_ylim()[0], 0), ax.get_ylim()[1]])

    # Legend
    if len(metrics_dict.keys()) > 1:
        leg = axes[0].legend(fontsize=fs, frameon=False, bbox_to_anchor=(0., 0.9, 1., .102),
                             loc=8, ncol=2, borderaxespad=0.)
        for l in leg.get_lines():
            l.set_linestyle('-')

    fig.subplots_adjust(hspace=.3, bottom=0.2, left=0.35, right=0.95, top=0.95)
    return fig


def extractCompTimes(filenames):
    ''' Extract computation times from a list of simulation files. '''
    tcomps = np.empty(len(filenames))
    for i, fn in enumerate(filenames):
        logger.info('Loading data from "%s"', fn)
        with open(fn, 'rb') as fh:
            frame = pickle.load(fh)
            meta = frame['meta']
        tcomps[i] = meta['tcomp']
    return tcomps


def getCompTimesQuant(outdir, neuron, xvars, xkey, data_fpaths, comptimes_fpath):
    if os.path.isfile(comptimes_fpath):
        logger.info('reading computation times from file: "%s"', comptimes_fpath)
        comptimes = pd.read_csv(comptimes_fpath, sep=',', index_col=xkey)
    else:
        logger.warning('extracting computation times for %s neuron', neuron)
        comptimes = pd.DataFrame(index=xvars)
        for stype in data_fpaths.keys():
            for i, xvar in enumerate(xvars):
                comptimes.loc[xvar, stype] = extractCompTimes([data_fpaths[stype][i]])
        comptimes.to_csv(comptimes_fpath, sep=',', index_label=xkey)
    return comptimes


def getLookupsCompTime(pneuron):
    # Check lookup file existence
    nbls = NeuronalBilayerSonophore(32e-9, pneuron)
    lookup_path = nbls.getLookupFilePath()
    if not os.path.isfile(lookup_path):
        raise FileNotFoundError('Missing lookup file: "{}"'.format(lookup_path))

    # Load lookups dictionary
    logger.debug('Loading comp times')
    with open(lookup_path, 'rb') as fh:
        df = pickle.load(fh)

    # Extract computation times and return sum
    tcomps4D = df['tcomp']
    return np.sum(tcomps4D)


def plotMapAndTraces(inputdir, pneuron, a, Fdrive, tstim, toffset, amps, PRF, DCs, cov,
                     FRbounds=None, insets=None, tbounds=None, Vbounds=None, thresholds=True,
                     map_figsize=None, trace_figsize=None, fs=8):

    mapcodes = codes(a, pneuron, Fdrive, PRF, tstim)
    subdir = os.path.join(inputdir, ' '.join(mapcodes))
    figs = {}

    # Activation map
    actmap = ActivationMap(subdir, pneuron, a, Fdrive, tstim, PRF, amps, DCs)
    mapfig = actmap.render(FRbounds=FRbounds, thresholds=thresholds, figsize=map_figsize, fs=fs)
    ax = mapfig.axes[0]
    if insets is not None:
        DC_insets, A_insets = zip(*insets)
        ax.scatter(np.array(DC_insets) * 1e2, np.array(A_insets) * 1e-3,
                   s=80, facecolors='none', edgecolors='k', linestyle='--')
    figs['map_' + '_'.join(mapcodes)] = mapfig

    # Related inset traces
    nbls = NeuronalBilayerSonophore(a, pneuron)
    tracefigs = {}
    for inset in insets:
        DC, Adrive = inset
        fname = '{}.pkl'.format(nbls.filecode(Fdrive, Adrive, tstim, toffset, PRF, DC, cov, 'sonic'))
        fpath = os.path.join(subdir, fname)
        tracefig = actmap.plotQVeff(fpath, trange=tbounds, ybounds=Vbounds, figsize=trace_figsize, fs=fs)
        figcode = 'VQ trace {} {:.1f}kPa {:.0f}%DC'.format(pneuron.name, Adrive * 1e-3, DC * 1e2)
        figs[figcode] = tracefig

    return figs


def codes(a, pneuron, Fdrive, PRF, tstim):
    return [
        pneuron.name,
        f'{si_format(a, space="")}m',
        f'{si_format(Fdrive, space="")}Hz',
        f'PRF{si_format(PRF, space="")}Hz',
        f'{si_format(tstim, space="")}s'
    ]


def plotThresholdAmps(pneurons, radii, freqs, tstim, toffset, PRF, DCs, cov,
                      fs=10, colors=None, figsize=cm2inch(10, 8)):
    ''' Plot threshold excitation amplitudes of several neurons determined by titration procedures,
        as a function of duty cycle, for various combinations of sonophore radius and US frequency.

        :param neurons: list of neuron names
        :param radii: list of sonophore radii (m)
        :param freqs: list US frequencies (Hz)
        :param PRF: pulse repetition frequency used for titration procedures (Hz)
        :param tstim: stimulus duration used for titration procedures
        :return: figure handle
    '''
    if isIterable(radii) and isIterable(freqs):
        raise ValueError('cannot plot threshold curves for more than 1 varying condition')
    if len(pneurons) > 3:
        raise ValueError('cannot plot threshold curves for more than 3 neuron types')

    if not isIterable(radii):
        radii = [radii]
    if not isIterable(freqs):
        freqs = [freqs]
    ncomb = len(pneurons) * len(freqs) * len(radii)
    if colors is None:
        colors = ['C{}'.format(i) for i in range(ncomb)]

    linestyles = ['--', ':', '-.']
    assert len(freqs) <= len(linestyles), 'too many frequencies'
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlabel('Duty cycle (%)', fontsize=fs)
    ax.set_ylabel('Amplitude (kPa)', fontsize=fs)
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(fs)
    ax.set_yscale('log')
    ax.set_xlim([0, 100])
    ax.set_ylim([10, 600])
    linestyles = ['-', '--', '-.']
    for pneuron, ls in zip(pneurons, linestyles):
        icolor = 0
        for i, a in enumerate(radii):
            nbls = NeuronalBilayerSonophore(a, pneuron)
            for j, Fdrive in enumerate(freqs):
                Athrs = np.array([
                    nbls.titrate(Fdrive, tstim, toffset, PRF, DC, cov, 'sonic') for DC in DCs])
                lbl = '{} neuron, {:.0f} nm, {}Hz, {}Hz PRF'.format(
                    pneuron.name, a * 1e9, *si_format([Fdrive, PRF], 0, space=' '))
                ax.plot(DCs * 1e2, Athrs * 1e-3, ls, c=colors[icolor], label=lbl)
                icolor += 1
    ax.legend(fontsize=fs - 2, frameon=False)
    fig.tight_layout()
    return fig


def computeAthr0D(pneuron, a, Fdrive, tstim, toffset, PRF, DC, fs_range):
    Athr = np.empty(fs_range.size)
    for i, fs in enumerate(fs_range):
        logger.info('computing threshold amplitude for fs = {:.0f}%'.format(fs * 1e2))
        model = SonicNode(pneuron, a=a, Fdrive=Fdrive, fs=fs)
        Athr[i] = model.titrate(tstim, toffset, PRF=PRF, DC=DC)
        model.clear()
    return Athr


def computeAthr1D(pneuron, a, Fdrive, tstim, toffset, PRF, DC, rs, deff, fs_range):
    Athr = np.empty(fs_range.size)
    for i, fs in enumerate(fs_range):
        logger.info('computing threshold amplitude for deff = {}m, fs = {:.0f}%'.format(
            si_format(deff), fs * 1e2))
        model = ExtendedSonicNode(pneuron, rs, a=a, Fdrive=Fdrive, fs=fs, deff=deff)
        Athr[i] = model.titrate(tstim, toffset, PRF=PRF, DC=DC)
        model.clear()
    return Athr


def getAthrData(fpath, func, args, fs_range):
    fs_key = 'fs (%)'
    Athr_key = 'Athr (kPa)'
    if os.path.isfile(fpath):
        logger.info('loading data from file: "{}"'.format(fpath))
        df = pd.read_csv(fpath, sep=',')
        assert np.allclose(df[fs_key].values, fs_range * 1e2), 'fs range not matching'
        Athr = df[Athr_key].values
    else:
        logger.info('computing data')
        Athr = func(*args, fs_range)
        df = pd.DataFrame({fs_key: fs_range * 1e2, Athr_key: Athr})
        df.to_csv(fpath, sep=',', index=False)
    return Athr


def prependTimeSeries(df, tonset):
    df0 = pd.DataFrame([df.iloc[0]])
    df = pd.concat([df0, df], ignore_index=True)
    df['t'][0] = -tonset
    return df