# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2018-09-28 16:13:34
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-06-17 22:08:28

''' Subpanels of the QSS approximation figure. '''

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
from argparse import ArgumentParser

from PySONIC.core import NeuronalBilayerSonophore
from PySONIC.utils import logger, selectDirDialog
from PySONIC.neurons import getPointNeuron
from PySONIC.parsers import FigureParser


# Plot parameters
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'arial'

# Figure basename
figbase = os.path.splitext(__file__)[0]


def plotQSSvars_vs_Adrive(neuron, a, Fdrive, PRF, DC, fs=8, markers=['-', '--', '.-'], title=None):

    pneuron = getPointNeuron(neuron)

    # Determine spiking threshold
    Vthr = pneuron.VT  # mV
    Qthr = pneuron.Cm0 * Vthr * 1e-3  # C/m2

    # Get QSS variables for each amplitude at threshold charge
    nbls = NeuronalBilayerSonophore(a, pneuron, Fdrive)
    Aref, _, Vmeff, QS_states = nbls.quasiSteadyStates(Fdrive, charges=Qthr, DCs=DC)

    # Compute US-ON and US-OFF ionic currents
    currents_on = pneuron.currents(Vmeff, QS_states)
    currents_off = pneuron.currents(pneuron.VT, QS_states)
    iNet_on = sum(currents_on.values())
    iNet_off = sum(currents_off.values())

    # Retrieve list of ionic currents names, with iLeak first
    ckeys = list(currents_on.keys())
    ckeys.insert(0, ckeys.pop(ckeys.index('iLeak')))

    # Compute quasi-steady ON, OFF and net charge variations, and threshold amplitude
    dQ_on = -iNet_on * DC / PRF
    dQ_off = -iNet_off * (1 - DC) / PRF
    dQ_net = dQ_on + dQ_off
    Athr = np.interp(0, dQ_net, Aref, left=0., right=np.nan)

    # Create figure
    fig, axes = plt.subplots(4, 1, figsize=(4, 6))
    axes[-1].set_xlabel('Amplitude (kPa)', fontsize=fs)
    for ax in axes:
        for skey in ['top', 'right']:
            ax.spines[skey].set_visible(False)
            ax.set_xscale('log')
        ax.set_xlim(1e1, 1e2)
        ax.set_xticks([1e1, 1e2])
        for item in ax.get_xticklabels() + ax.get_yticklabels():
            item.set_fontsize(fs)
        for item in ax.get_xticklabels(minor=True):
            item.set_visible(False)
    figname = '{} neuron thr dynamics {:.1f}nC_cm2 {:.0f}% DC'.format(
        pneuron.name, Qthr * 1e5, DC * 1e2)
    fig.suptitle(figname, fontsize=fs)

    # Subplot 1: Vmeff
    ax = axes[0]
    ax.set_ylabel('Effective potential (mV)', fontsize=fs)
    Vbounds = (-120, -40)
    ax.set_ylim(Vbounds)
    ax.set_yticks([Vbounds[0], pneuron.Vm0, Vbounds[1]])
    ax.set_yticklabels(['{:.0f}'.format(Vbounds[0]), '$V_{m0}$', '{:.0f}'.format(Vbounds[1])])
    ax.plot(Aref * 1e-3, Vmeff, '--', color='C0', label='ON')
    ax.plot(Aref * 1e-3, pneuron.VT * np.ones(Aref.size), ':', color='C0', label='OFF')
    ax.axhline(pneuron.Vm0, linewidth=0.5, color='k')

    # Subplot 2: quasi-steady states
    ax = axes[1]
    ax.set_ylabel('Quasi-steady states', fontsize=fs)
    ax.set_yticks([0, 0.5, 0.6])
    ax.set_yticklabels(['0', '0.5', '1'])
    ax.set_ylim([-0.05, 0.65])
    d = .01
    f = 1.03
    xcut = ax.get_xlim()[0]
    for ycut in [0.54, 0.56]:
        ax.plot([xcut / f, xcut * f], [ycut - d, ycut + d], color='k', clip_on=False)
    for label, QS_state in zip(pneuron.states, QS_states):
        if label == 'h':
            QS_state -= 0.4
        ax.plot(Aref * 1e-3, QS_state, label=label)

    # Subplot 3: currents
    ax = axes[2]
    ax.set_ylabel('QSS Currents (mA/m2)', fontsize=fs)
    Ibounds = (-10, 10)
    ax.set_ylim(Ibounds)
    ax.set_yticks([Ibounds[0], 0.0, Ibounds[1]])
    for i, key in enumerate(ckeys):
        c = 'C{}'.format(i)
        if isinstance(currents_off[key], float):
            currents_off[key] = np.ones(Aref.size) * currents_off[key]
        ax.plot(Aref * 1e-3, currents_on[key], '--', label=key, c=c)
        ax.plot(Aref * 1e-3, currents_off[key], ':', c=c)
    ax.plot(Aref * 1e-3, iNet_on, '--', color='k', label='iNet')
    ax.plot(Aref * 1e-3, iNet_off, ':', color='k')
    ax.axhline(0, color='k', linewidth=0.5)

    # Subplot 4: charge variations and activation threshold
    ax = axes[3]
    ax.set_ylabel('$\\rm \Delta Q_{QS}\ (nC/cm^2)$', fontsize=fs)
    dQbounds = (-0.06, 0.1)
    ax.set_ylim(dQbounds)
    ax.set_yticks([dQbounds[0], 0.0, dQbounds[1]])
    ax.plot(Aref * 1e-3, dQ_on, '--', color='C0', label='ON')
    ax.plot(Aref * 1e-3, dQ_off, ':', color='C0', label='OFF')
    ax.plot(Aref * 1e-3, dQ_net, color='C0', label='Net')
    ax.plot([Athr * 1e-3] * 2, [ax.get_ylim()[0], 0], linestyle='--', color='k')
    ax.plot([Athr * 1e-3], [0], 'o', c='k')
    ax.axhline(0, color='k', linewidth=0.5)

    fig.tight_layout()
    fig.subplots_adjust(right=0.8)
    for ax in axes:
        ax.legend(loc='center right', fontsize=fs, frameon=False, bbox_to_anchor=(1.3, 0.5))

    if title is not None:
        fig.canvas.set_window_title(title)
    return fig


def plotQSSdQ_vs_Adrive(neuron, a, Fdrive, PRF, DCs, fs=8, title=None):

    pneuron = getPointNeuron(neuron)

    # Determine spiking threshold
    Vthr = pneuron.VT  # mV
    Qthr = pneuron.Cm0 * Vthr * 1e-3  # C/m2

    # Get QSS variables for each amplitude and DC at threshold charge
    nbls = NeuronalBilayerSonophore(a, pneuron, Fdrive)
    Aref, _, Vmeff, QS_states = nbls.quasiSteadyStates(Fdrive, charges=Qthr, DCs=DCs)

    dQnet = np.empty((DCs.size, Aref.size))
    Athr = np.empty(DCs.size)
    for i, DC in enumerate(DCs):
        # Compute US-ON and US-OFF net membrane current from QSS variables
        iNet_on = pneuron.iNet(Vmeff, QS_states[:, :, i])
        iNet_off = pneuron.iNet(Vthr, QS_states[:, :, i])

        # Compute the pulse average net current along the amplitude space
        iNet_avg = iNet_on * DC + iNet_off * (1 - DC)
        dQnet[i, :] = -iNet_avg / PRF

        # Find the threshold amplitude that cancels the pulse average net current
        Athr[i] = np.interp(0, -iNet_avg, Aref, left=0., right=np.nan)

    # Create figure
    fig, ax = plt.subplots(figsize=(4, 2))
    figname = '{} neuron thr vs DC'.format(pneuron.name, Qthr * 1e5)
    fig.suptitle(figname, fontsize=fs)
    for key in ['top', 'right']:
        ax.spines[key].set_visible(False)
    ax.set_xscale('log')
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(fs)
    for item in ax.get_xticklabels(minor=True):
            item.set_visible(False)
    ax.set_xlabel('Amplitude (kPa)', fontsize=fs)
    ax.set_ylabel('$\\rm \Delta Q_{QS}\ (nC/cm^2)$', fontsize=fs)
    ax.set_xlim(1e1, 1e2)
    ax.axhline(0., linewidth=0.5, color='k')
    ax.set_ylim(-0.06, 0.12)
    ax.set_yticks([-0.05, 0.0, 0.10])
    ax.set_yticklabels(['-0.05', '0', '0.10'])

    norm = matplotlib.colors.LogNorm(DCs.min(), DCs.max())
    sm = cm.ScalarMappable(norm=norm, cmap='viridis')
    sm._A = []
    for i, DC in enumerate(DCs):
        ax.plot(Aref * 1e-3, dQnet[i, :], c=sm.to_rgba(DC), label='{:.0f}% DC'.format(DC * 1e2))
        ax.plot([Athr[i] * 1e-3] * 2, [ax.get_ylim()[0], 0], linestyle='--', c=sm.to_rgba(DC))
        ax.plot([Athr[i] * 1e-3], [0], 'o', c=sm.to_rgba(DC))

    fig.tight_layout()
    fig.subplots_adjust(right=0.8)
    ax.legend(loc='center right', fontsize=fs, frameon=False, bbox_to_anchor=(1.3, 0.5))

    if title is not None:
        fig.canvas.set_window_title(title)

    return fig


def plotQSSAthr_vs_DC(neurons, a, Fdrive, DCs_dense, DCs_sparse, fs=8, title=None):

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.set_title('Rheobase amplitudes', fontsize=fs)
    ax.set_xlabel('Duty cycle (%)', fontsize=fs)
    ax.set_ylabel('$\\rm A_T\ (kPa)$', fontsize=fs)
    for key in ['top', 'right']:
        ax.spines[key].set_visible(False)
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(fs)
    ax.set_xticks([25, 50, 75, 100])
    ax.set_yscale('log')
    ax.set_ylim([10, 600])
    norm = matplotlib.colors.LogNorm(DCs_sparse.min(), DCs_sparse.max())
    sm = cm.ScalarMappable(norm=norm, cmap='viridis')
    sm._A = []
    for i, neuron in enumerate(neurons):
        pneuron = getPointNeuron(neuron)
        nbls = NeuronalBilayerSonophore(a, pneuron)
        Athrs_dense = nbls.findRheobaseAmps(DCs_dense, Fdrive, pneuron.VT)[0] * 1e-3  # kPa
        Athrs_sparse = nbls.findRheobaseAmps(DCs_sparse, Fdrive, pneuron.VT)[0] * 1e-3  # kPa
        ax.plot(DCs_dense * 1e2, Athrs_dense, label='{} neuron'.format(pneuron.name))
        for DC, Athr in zip(DCs_sparse, Athrs_sparse):
            ax.plot(DC * 1e2, Athr, 'o',
                    label='{:.0f}% DC'.format(DC * 1e2) if i == len(neurons) - 1 else None,
                    c=sm.to_rgba(DC))
    ax.legend(fontsize=fs, frameon=False)
    fig.tight_layout()
    if title is not None:
        fig.canvas.set_window_title(title)
    return fig


def main():

    parser = FigureParser(['a', 'b', 'c', 'd', 'e'])
    args = parser.parse()
    logger.setLevel(args['loglevel'])
    figset = args['subset']
    logger.info('Generating panels {} of {}'.format(figset, figbase))

    # Parameters
    a = 32e-9  # m
    Fdrive = 500e3  # Hz
    PRF = 100.0  # Hz
    DC = 0.5
    DCs_sparse = np.array([5, 15, 50, 75, 95]) / 1e2
    DCs_dense = np.arange(1, 101) / 1e2

    # Figures
    figs = []
    if 'a' in figset:
        figs += [
            plotQSSvars_vs_Adrive('RS', a, Fdrive, PRF, DC, title=figbase + 'a RS'),
            plotQSSvars_vs_Adrive('LTS', a, Fdrive, PRF, DC, title=figbase + 'a LTS')
        ]
    if 'b' in figset:
        figs += [
            plotQSSdQ_vs_Adrive('RS', a, Fdrive, PRF, DCs_sparse, title=figbase + 'b RS'),
            plotQSSdQ_vs_Adrive('LTS', a, Fdrive, PRF, DCs_sparse, title=figbase + 'b LTS')
        ]
    if 'c' in figset:
        figs.append(plotQSSAthr_vs_DC(['RS', 'LTS'], a, Fdrive, DCs_dense, DCs_sparse,
                                      title=figbase + 'c'))

    if args['save']:
        for fig in figs:
            figname = '{}.pdf'.format(fig.canvas.get_window_title())
            fig.savefig(os.path.join(args['outputdir'], figname), transparent=True)
    else:
        plt.show()


if __name__ == '__main__':
    main()
