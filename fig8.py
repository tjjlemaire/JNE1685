# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2018-11-27 17:57:45
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-06-17 22:05:08

''' Sub-panels of threshold curves for various sonophore radii and US frequencies. '''

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from PySONIC.utils import logger, si_format
from PySONIC.plt import cm2inch
from PySONIC.parsers import FigureParser

# Plot parameters
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'arial'

# Figure basename
figbase = os.path.splitext(__file__)[0]


def getThresholdAmplitudes(root, neuron, a, Fdrive, tstim, PRF):
    subfolder = '{} {:.0f}nm {}Hz PRF{}Hz {}s'.format(
        neuron, a * 1e9,
        *si_format([Fdrive, PRF, tstim], 0, space='')
    )

    fname = 'log_ASTIM.xlsx'
    fpath = os.path.join(root, subfolder, fname)

    df = pd.read_excel(fpath, sheet_name='Data')
    DCs = df['Duty factor'].values
    Athrs = df['Adrive (kPa)'].values

    iDCs = np.argsort(DCs)
    DCs = DCs[iDCs]
    Athrs = Athrs[iDCs]

    return DCs, Athrs


def plotThresholdAmps(root, neurons, radii, freqs, PRF, tstim, fs=10, colors=None, figsize=None):
    ''' Plot threshold excitation amplitudes of several neurons determined by titration procedures,
        as a function of duty cycle, for various combinations of sonophore radius and US frequency.

        :param neurons: list of neuron names
        :param radii: list of sonophore radii (m)
        :param freqs: list US frequencies (Hz)
        :param PRF: pulse repetition frequency used for titration procedures (Hz)
        :param tstim: stimulus duration used for titration procedures
        :return: figure handle
    '''
    if figsize is None:
        figsize = cm2inch(8, 7)
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
    linestyles = ['-', '--']
    for neuron, ls in zip(neurons, linestyles):
        icolor = 0
        for i, a in enumerate(radii):
            for j, Fdrive in enumerate(freqs):
                if colors is None:
                    color = 'C{}'.format(icolor)
                else:
                    color = colors[icolor]
                DCs, Athrs = getThresholdAmplitudes(root, neuron, a, Fdrive, tstim, PRF)
                lbl = '{} neuron, {:.0f} nm, {}Hz, {}Hz PRF'.format(
                    neuron, a * 1e9, *si_format([Fdrive, PRF], 0, space=' '))
                ax.plot(DCs * 1e2, Athrs, ls, c=color, label=lbl)
                icolor += 1
    ax.legend(fontsize=fs - 5, frameon=False)
    fig.tight_layout()
    return fig


def main():
    parser = FigureParser(['a', 'b'])
    parser.addInputDir()
    args = parser.parse()
    logger.setLevel(args['loglevel'])
    figset = args['subset']
    inputdir = args['inputdir']
    logger.info('Generating panels {} of {}'.format(figset, figbase))

    # Parameters
    neurons = ['RS', 'LTS']
    radii = np.array([16, 32, 64]) * 1e-9  # m
    a = radii[1]
    freqs = np.array([20, 500, 4000]) * 1e3  # Hz
    Fdrive = freqs[1]
    PRFs = np.array([1e1, 1e2, 1e3])  # Hz
    PRF = PRFs[1]
    tstim = 1  # s

    colors = plt.get_cmap('tab20c').colors

    # Generate figures
    figs = []
    if 'a' in figset:
        fig = plotThresholdAmps(inputdir, neurons, radii, [Fdrive], PRF, tstim,
                                fs=12, colors=colors[:3][::-1])
        fig.canvas.set_window_title(figbase + 'a')
        figs.append(fig)
    if 'b' in figset:
        fig = plotThresholdAmps(inputdir, neurons, [a], freqs, PRF, tstim,
                                fs=12, colors=colors[8:11][::-1])
        fig.canvas.set_window_title(figbase + 'b')
        figs.append(fig)

    if args['save']:
        for fig in figs:
            figname = '{}.pdf'.format(fig.canvas.get_window_title())
            fig.savefig(os.path.join(args['outpudir'], figname), transparent=True)
    else:
        plt.show()


if __name__ == '__main__':
    main()
