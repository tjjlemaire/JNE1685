# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2018-06-06 18:38:04
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-06-28 16:14:02

''' Sub-panels of the NICE and SONIC accuracies comparative figure. '''


import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from PySONIC.core import NeuronalBilayerSonophore
from PySONIC.utils import *
from PySONIC.neurons import *
from PySONIC.plt import CompTimeSeries, cm2inch
from PySONIC.parsers import FigureParser

from utils import *

# Plot parameters
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'arial'

# Figure basename
figbase = os.path.splitext(__file__)[0]


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


def Qprofiles_vs_amp(neuron, a, Fdrive, CW_Athrs, tstim, toffset, inputdir):
    ''' Comparison of resulting charge profiles for CW stimuli at sub-threshold,
        threshold and supra-threshold amplitudes. '''
    Athr = CW_Athrs[neuron].loc[Fdrive * 1e-3]  # kPa
    amps = np.array([Athr - 5., Athr, Athr + 20.]) * 1e3  # Pa
    subdir = os.path.join(inputdir, neuron)
    sonic_fpaths = getSims(subdir, neuron, a, NeuronalBilayerSonophore.simQueue(
        [Fdrive], amps, [tstim], [toffset], [None], [1.], 'sonic'))
    full_fpaths = getSims(subdir, neuron, a, NeuronalBilayerSonophore.simQueue(
        [Fdrive], amps, [tstim], [toffset], [None], [1.], 'full'))
    regimes = ['AT - 5 kPa', 'AT', 'AT + 20 kPa']
    comp_plot = CompTimeSeries(sum([[x, y] for x, y in zip(full_fpaths, sonic_fpaths)], []), 'Qm')
    fig = comp_plot.render(
        labels=sum([['', x] for x in regimes], []),
        lines=['-', '--'] * len(regimes),
        colors=plt.get_cmap('Paired').colors[:2 * len(regimes)],
        fs=8,
        patches='one',
        xticks=[0, 250],
        yticks=[getPointNeuron(neuron).Vm0, 25],
        straightlegend=True, figsize=cm2inch(12.5, 5.8)
    )
    fig.axes[0].get_xaxis().set_label_coords(0.5, -0.05)
    fig.subplots_adjust(bottom=0.2, right=0.95, top=0.95)
    fig.canvas.set_window_title(figbase + 'a Qprofiles')
    return fig


def spikemetrics_vs_amp(neuron, a, Fdrive, amps, tstim, toffset, inputdir):
    ''' Comparison of spiking metrics for CW stimuli at various supra-threshold amplitudes. '''
    subdir = os.path.join(inputdir, neuron)
    sonic_fpaths = getSims(subdir, neuron, a, NeuronalBilayerSonophore.simQueue(
        [Fdrive], amps, [tstim], [toffset], [None], [1.], 'sonic'))
    full_fpaths = getSims(subdir, neuron, a, NeuronalBilayerSonophore.simQueue(
        [Fdrive], amps, [tstim], [toffset], [None], [1.], 'full'))
    data_fpaths = {'full': full_fpaths, 'sonic': sonic_fpaths}
    metrics_files = {x: '{}_spikemetrics_vs_amplitude_{}.csv'.format(neuron, x)
                     for x in ['full', 'sonic']}
    metrics_fpaths = {key: os.path.join(inputdir, value) for key, value in metrics_files.items()}
    xlabel = 'Amplitude (kPa)'
    metrics = getSpikingMetrics(
        subdir, neuron, amps * 1e-3, xlabel, data_fpaths, metrics_fpaths)
    fig = plotSpikingMetrics(amps * 1e-3, xlabel, {neuron: metrics}, logscale=True)
    fig.canvas.set_window_title(figbase + 'a spikemetrics')
    return fig


def Qprofiles_vs_freq(neuron, a, freqs, CW_Athrs, tstim, toffset, inputdir):
    ''' Comparison of resulting charge profiles for supra-threshold CW stimuli
        at low and high US frequencies. '''
    subdir = os.path.join(inputdir, neuron)
    sonic_fpaths, full_fpaths = [], []
    for Fdrive in freqs:
        Athr = CW_Athrs[neuron].loc[Fdrive * 1e-3]  # kPa
        Adrive = (Athr + 20.) * 1e3  # Pa
        sonic_fpaths += getSims(subdir, neuron, a, NeuronalBilayerSonophore.simQueue(
            [Fdrive], [Adrive], [tstim], [toffset], [None], [1.], 'sonic'))
        full_fpaths += getSims(subdir, neuron, a, NeuronalBilayerSonophore.simQueue(
            [Fdrive], [Adrive], [tstim], [toffset], [None], [1.], 'full'))
    comp_plot = CompTimeSeries(sum([[x, y] for x, y in zip(full_fpaths, sonic_fpaths)], []), 'Qm')
    fig = comp_plot.render(
        labels=sum([['', '{}Hz'.format(si_format(f))] for f in freqs], []),
        lines=['-', '--'] * len(freqs), colors=plt.get_cmap('Paired').colors[6:10], fs=8,
        patches='one', xticks=[0, 250], yticks=[getPointNeuron(neuron).Vm0, 25],
        straightlegend=True, figsize=cm2inch(12.5, 5.8),
        inset={'xcoords': [5, 40], 'ycoords': [-35, 45], 'xlims': [57.5, 58.5], 'ylims': [10, 35]}
    )
    fig.axes[0].get_xaxis().set_label_coords(0.5, -0.05)
    fig.subplots_adjust(bottom=0.2, right=0.95, top=0.95)
    fig.canvas.set_window_title(figbase + 'b Qprofiles')
    return fig


def spikemetrics_vs_freq(neuron, a, freqs, CW_Athrs, tstim, toffset, inputdir):
    ''' Comparison of spiking metrics for supra-threshold CW stimuli at various US frequencies. '''
    subdir = os.path.join(inputdir, neuron)
    sonic_fpaths, full_fpaths = [], []
    for Fdrive in freqs:
        Athr = CW_Athrs[neuron].loc[Fdrive * 1e-3]  # kPa
        Adrive = (Athr + 20.) * 1e3  # Pa
        sonic_fpaths += getSims(subdir, neuron, a, NeuronalBilayerSonophore.simQueue(
            [Fdrive], [Adrive], [tstim], [toffset], [None], [1.], 'sonic'))
        full_fpaths += getSims(subdir, neuron, a, NeuronalBilayerSonophore.simQueue(
            [Fdrive], [Adrive], [tstim], [toffset], [None], [1.], 'full'))
    data_fpaths = {'full': full_fpaths, 'sonic': sonic_fpaths}
    metrics_files = {x: '{}_spikemetrics_vs_frequency_{}.csv'.format(neuron, x)
                     for x in ['full', 'sonic']}
    metrics_fpaths = {key: os.path.join(inputdir, value) for key, value in metrics_files.items()}
    xlabel = 'Frequency (kHz)'
    metrics = getSpikingMetrics(
        subdir, neuron, freqs * 1e-3, xlabel, data_fpaths, metrics_fpaths)
    fig = plotSpikingMetrics(freqs * 1e-3, xlabel, {neuron: metrics}, logscale=True)
    fig.canvas.set_window_title(figbase + 'b spikemetrics')
    return fig


def Qprofiles_vs_radius(neuron, radii, Fdrive, CW_Athrs, tstim, toffset, inputdir):
    ''' Comparison of resulting charge profiles for supra-threshold CW stimuli
        for small and large sonophore radii. '''
    subdir = os.path.join(inputdir, neuron)
    sonic_fpaths, full_fpaths = [], []
    for a in radii:
        Athr = CW_Athrs[neuron].loc[a * 1e9]  # kPa
        Adrive = (Athr + 20.) * 1e3  # Pa
        sonic_fpaths += getSims(subdir, neuron, a, NeuronalBilayerSonophore.simQueue(
            [Fdrive], [Adrive], [tstim], [toffset], [None], [1.], 'sonic'))
        full_fpaths += getSims(subdir, neuron, a, NeuronalBilayerSonophore.simQueue(
            [Fdrive], [Adrive], [tstim], [toffset], [None], [1.], 'full'))

    tmp = plt.get_cmap('Paired').colors
    colors = tmp[2:4] + tmp[10:12]
    comp_plot = CompTimeSeries(sum([[x, y] for x, y in zip(full_fpaths, sonic_fpaths)], []), 'Qm')
    fig = comp_plot.render(
        labels=sum([['', '{:.0f} nm'.format(a * 1e9)] for a in radii], []),
        lines=['-', '--'] * len(radii), colors=colors, fs=8,
        patches='one', xticks=[0, 250], yticks=[getPointNeuron(neuron).Vm0, 25],
        straightlegend=True, figsize=cm2inch(12.5, 5.8)
    )
    fig.axes[0].get_xaxis().set_label_coords(0.5, -0.05)
    fig.subplots_adjust(bottom=0.2, right=0.95, top=0.95)
    fig.canvas.set_window_title(figbase + 'c Qprofiles')
    return fig


def spikemetrics_vs_radius(neuron, radii, Fdrive, CW_Athrs, tstim, toffset, inputdir):
    ''' Comparison of spiking metrics for supra-threshold CW stimuli
        with various sonophore diameters. '''
    subdir = os.path.join(inputdir, neuron)
    sonic_fpaths, full_fpaths = [], []
    for a in radii:
        Athr = CW_Athrs[neuron].loc[np.round(a * 1e9, 1)]  # kPa
        Adrive = (Athr + 20.) * 1e3  # Pa
        sonic_fpaths += getSims(subdir, neuron, a, NeuronalBilayerSonophore.simQueue(
            [Fdrive], [Adrive], [tstim], [toffset], [None], [1.], 'sonic'))
        full_fpaths += getSims(subdir, neuron, a, NeuronalBilayerSonophore.simQueue(
            [Fdrive], [Adrive], [tstim], [toffset], [None], [1.], 'full'))
    data_fpaths = {'full': full_fpaths, 'sonic': sonic_fpaths}
    metrics_files = {x: '{}_spikemetrics_vs_radius_{}.csv'.format(neuron, x)
                     for x in ['full', 'sonic']}
    metrics_fpaths = {key: os.path.join(inputdir, value) for key, value in metrics_files.items()}
    xlabel = 'Sonophore radius (nm)'
    metrics = getSpikingMetrics(
        subdir, neuron, radii * 1e9, xlabel, data_fpaths, metrics_fpaths)
    fig = plotSpikingMetrics(radii * 1e9, xlabel, {neuron: metrics}, logscale=True)
    fig.canvas.set_window_title(figbase + 'c spikemetrics')
    return fig


def Qprofiles_vs_DC(neurons, a, Fdrive, Adrive, tstim, toffset, PRF, DC, inputdir):
    ''' Comparison of resulting charge profiles for PW stimuli at 5% duty cycle
        for different neuron types. '''
    sonic_fpaths, full_fpaths = [], []
    for neuron in neurons:
        subdir = os.path.join(inputdir, neuron)
        sonic_fpaths += getSims(subdir, neuron, a, NeuronalBilayerSonophore.simQueue(
            [Fdrive], [Adrive], [tstim], [toffset], [PRF], [DC], 'sonic'))
        full_fpaths += getSims(subdir, neuron, a, NeuronalBilayerSonophore.simQueue(
            [Fdrive], [Adrive], [tstim], [toffset], [PRF], [DC], 'full'))
    colors = list(plt.get_cmap('Paired').colors[:6])
    del colors[2:4]
    comp_plot = CompTimeSeries(sum([[x, y] for x, y in zip(full_fpaths, sonic_fpaths)], []), 'Qm')
    fig = comp_plot.render(
        labels=sum([['', '{}, {:.0f}% DC'.format(x, DC * 1e2)] for x in neurons], []),
        lines=['-', '--'] * len(neurons), colors=colors, fs=8, patches='one',
        xticks=[0, 250], yticks=[min(getPointNeuron(n).Vm0 for n in neurons), 50],
        straightlegend=True, figsize=cm2inch(12.5, 5.8)
    )
    fig.axes[0].get_xaxis().set_label_coords(0.5, -0.05)
    fig.subplots_adjust(bottom=0.2, right=0.95, top=0.95)
    fig.canvas.set_window_title(figbase + 'd Qprofiles')
    return fig


def spikemetrics_vs_DC(neurons, a, Fdrive, Adrive, tstim, toffset, PRF, DCs, inputdir):
    ''' Comparison of spiking metrics for PW stimuli at various duty cycle for
        different neuron types. '''
    metrics_dict = {}
    xlabel = 'Duty cycle (%)'
    colors = list(plt.get_cmap('Paired').colors[:6])
    del colors[2:4]
    colors_dict = {}
    for i, neuron in enumerate(neurons):
        subdir = os.path.join(inputdir, neuron)
        sonic_fpaths = getSims(subdir, neuron, a, NeuronalBilayerSonophore.simQueue(
            [Fdrive], [Adrive], [tstim], [toffset], [PRF], DCs, 'sonic'))
        full_fpaths = getSims(subdir, neuron, a, NeuronalBilayerSonophore.simQueue(
            [Fdrive], [Adrive], [tstim], [toffset], [PRF], DCs, 'full'))
        metrics_files = {x: '{}_spikemetrics_vs_DC_{}.csv'.format(neuron, x)
                         for x in ['full', 'sonic']}
        metrics_fpaths = {key: os.path.join(inputdir, value) for key, value in metrics_files.items()}
        sonic_fpaths = sonic_fpaths[1:] + [sonic_fpaths[0]]
        full_fpaths = full_fpaths[1:] + [full_fpaths[0]]
        data_fpaths = {'full': full_fpaths, 'sonic': sonic_fpaths}
        metrics_dict[neuron] = getSpikingMetrics(
            subdir, neuron, DCs * 1e2, xlabel, data_fpaths, metrics_fpaths)
        colors_dict[neuron] = {'full': colors[2 * i], 'sonic': colors[2 * i + 1]}
    fig = plotSpikingMetrics(DCs * 1e2, xlabel, metrics_dict, spikeamp=False, colors=colors_dict)
    fig.canvas.set_window_title(figbase + 'd spikemetrics')
    return fig


def Qprofiles_vs_PRF(neuron, a, Fdrive, Adrive, tstim, toffset, PRFs, DC, inputdir):
    ''' Comparison of resulting charge profiles for PW stimuli at 5% duty cycle
        with different pulse repetition frequencies. '''
    subdir = os.path.join(inputdir, neuron)
    sonic_fpaths = getSims(subdir, neuron, a, NeuronalBilayerSonophore.simQueue(
        [Fdrive], [Adrive], [tstim], [toffset], PRFs, [DC], 'sonic'))
    full_fpaths = getSims(subdir, neuron, a, NeuronalBilayerSonophore.simQueue(
        [Fdrive], [Adrive], [tstim], [toffset], PRFs, [DC], 'full'))
    patches = [False, True] * len(PRFs)
    patches[-1] = False
    comp_plot = CompTimeSeries(sum([[x, y] for x, y in zip(full_fpaths, sonic_fpaths)], []), 'Qm')
    fig = comp_plot.render(
        labels=sum([['', '{}Hz PRF'.format(si_format(PRF, space=' '))] for PRF in PRFs], []),
        lines=['-', '--'] * len(PRFs), colors=plt.get_cmap('Paired').colors[4:12], fs=8,
        patches=patches,
        xticks=[0, 250], yticks=[getPointNeuron(neuron).Vm0, 50],
        straightlegend=True, figsize=cm2inch(12.5, 5.8)
    )
    fig.axes[0].get_xaxis().set_label_coords(0.5, -0.05)
    fig.subplots_adjust(bottom=0.2, right=0.95, top=0.95)
    fig.canvas.set_window_title(figbase + 'e Qprofiles')
    return fig


def spikemetrics_vs_PRF(neuron, a, Fdrive, Adrive, tstim, toffset, PRFs, DC, inputdir):
    ''' Comparison of spiking metrics for PW stimuli at 5% duty cycle
        with different pulse repetition frequencies. '''
    xlabel = 'PRF (Hz)'
    subdir = os.path.join(inputdir, neuron)
    sonic_fpaths = getSims(subdir, neuron, a, NeuronalBilayerSonophore.simQueue(
        [Fdrive], [Adrive], [tstim], [toffset], PRFs, [DC], 'sonic'))
    full_fpaths = getSims(subdir, neuron, a, NeuronalBilayerSonophore.simQueue(
        [Fdrive], [Adrive], [tstim], [toffset], PRFs, [DC], 'full'))
    data_fpaths = {'full': full_fpaths, 'sonic': sonic_fpaths}
    metrics_files = {x: '{}_spikemetrics_vs_PRF_{}.csv'.format(neuron, x)
                     for x in ['full', 'sonic']}
    metrics_fpaths = {key: os.path.join(inputdir, value) for key, value in metrics_files.items()}
    metrics = getSpikingMetrics(
        subdir, neuron, PRFs, xlabel, data_fpaths, metrics_fpaths)
    fig = plotSpikingMetrics(PRFs, xlabel, {neuron: metrics}, spikeamp=False, logscale=True)
    fig.canvas.set_window_title(figbase + 'e spikemetrics')
    return fig


def main():

    parser = FigureParser(['a', 'b', 'c', 'd', 'e'])
    parser.addInputDir()
    args = parser.parse()
    logger.setLevel(args['loglevel'])
    figset = args['subset']
    inputdir = args['inputdir']
    logger.info('Generating panels {} of {}'.format(figset, figbase))

    # Parameters
    radii = np.array([16, 22.6, 32, 45.3, 64]) * 1e-9  # m
    a = 32e-9  # m
    tstim = 150e-3  # s
    toffset = 100e-3  # s
    freqs = np.array([20e3, 100e3, 500e3, 1e6, 2e6, 3e6, 4e6])  # Hz
    Fdrive = 500e3  # Hz
    amps = np.array([50, 100, 300, 600]) * 1e3  # Pa
    Adrive = 100e3  # Pa
    PRFs_sparse = np.array([1e1, 1e2, 1e3, 1e4])  # Hz
    PRFs_dense = sum([[x, 2 * x, 5 * x] for x in PRFs_sparse[:-1]], []) + [PRFs_sparse[-1]]  # Hz
    PRF = 100  # Hz
    DCs = np.array([5, 10, 25, 50, 75, 100]) * 1e-2
    DC = 0.05

    # Get threshold amplitudes if needed
    if 'a' in figset or 'b' in figset:
        CW_Athr_vs_Fdrive = getCWtitrations_vs_Fdrive(
            ['RS'], a, freqs, tstim, toffset, os.path.join(inputdir, 'CW_Athrs_vs_freqs.csv'))
    if 'c' in figset:
        CW_Athr_vs_radius = getCWtitrations_vs_radius(
            ['RS'], radii, Fdrive, tstim, toffset, os.path.join(inputdir, 'CW_Athrs_vs_radius.csv'))

    # Generate figures
    figs = []
    if 'a' in figset:
        figs.append(Qprofiles_vs_amp('RS', a, Fdrive, CW_Athr_vs_Fdrive, tstim, toffset, inputdir))
        figs.append(spikemetrics_vs_amp('RS', a, Fdrive, amps, tstim, toffset, inputdir))
    if 'b' in figset:
        figs.append(Qprofiles_vs_freq(
            'RS', a, [freqs.min(), freqs.max()], CW_Athr_vs_Fdrive, tstim, toffset, inputdir))
        figs.append(spikemetrics_vs_freq(
            'RS', a, freqs, CW_Athr_vs_Fdrive, tstim, toffset, inputdir))
    if 'c' in figset:
        figs.append(Qprofiles_vs_radius(
            'RS', [radii.min(), radii.max()], Fdrive, CW_Athr_vs_radius, tstim, toffset, inputdir))
        figs.append(spikemetrics_vs_radius(
            'RS', radii, Fdrive, CW_Athr_vs_radius, tstim, toffset, inputdir))
    if 'd' in figset:
        figs.append(Qprofiles_vs_DC(
            ['RS', 'LTS'], a, Fdrive, Adrive, tstim, toffset, PRF, DC, inputdir))
        figs.append(spikemetrics_vs_DC(
            ['RS', 'LTS'], a, Fdrive, Adrive, tstim, toffset, PRF, DCs, inputdir))
    if 'e' in figset:
        figs.append(Qprofiles_vs_PRF(
            'LTS', a, Fdrive, Adrive, tstim, toffset, PRFs_sparse, DC, inputdir))
        figs.append(spikemetrics_vs_PRF(
            'LTS', a, Fdrive, Adrive, tstim, toffset, PRFs_dense, DC, inputdir))

    if args['save']:
        for fig in figs:
            figname = '{}.pdf'.format(fig.canvas.get_window_title())
            fig.savefig(os.path.join(args['outputdir'], figname), transparent=True)
    else:
        plt.show()


if __name__ == '__main__':
    main()
