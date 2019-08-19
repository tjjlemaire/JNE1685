# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2018-06-06 18:38:04
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-06-29 13:56:42

''' Sub-panels of the model optimization figure. '''


import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.patches import Rectangle

from PySONIC.utils import logger, rescale, si_format
from PySONIC.plt import GroupedTimeSeries, cm2inch
from PySONIC.constants import NPC_DENSE
from PySONIC.neurons import getPointNeuron
from PySONIC.core import BilayerSonophore, NeuronalBilayerSonophore
from PySONIC.parsers import FigureParser


# Plot parameters
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'arial'

# Figure basename
figbase = os.path.splitext(__file__)[0]


def PmApprox(bls, Z, fs=12, lw=2):
    fig, ax = plt.subplots(figsize=cm2inch(7, 7))
    for key in ['right', 'top']:
        ax.spines[key].set_visible(False)
    for key in ['bottom', 'left']:
        ax.spines[key].set_linewidth(2)
    ax.spines['bottom'].set_position('zero')
    ax.set_xlabel('Z (nm)', fontsize=fs)
    ax.set_ylabel('Pressure (kPa)', fontsize=fs, labelpad=-10)
    ax.set_xticks([0, bls.a * 1e9])
    ax.set_xticklabels(['0', 'a'])
    ax.tick_params(axis='x', which='major', length=25, pad=5)
    ax.set_yticks([0])
    ax.set_ylim([-10, 50])
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(fs)
    ax.plot(Z * 1e9, bls.v_PMavg(Z, bls.v_curvrad(Z), bls.surface(Z)) * 1e-3, c='g', label='$P_m$')
    ax.plot(Z * 1e9, bls.PMavgpred(Z) * 1e-3, '--', c='r', label='$\~P_m$')
    ax.axhline(y=0, color='k')
    ax.legend(fontsize=fs, frameon=False)
    fig.tight_layout()
    fig.canvas.set_window_title(figbase + 'a')
    return fig


def recasting(nbls, Fdrive, Adrive, fs=12, lw=2, ps=15):

    # Run effective simulation
    data, _ = nbls.simulate(Fdrive, Adrive, 5 / Fdrive, 0., method='full')
    t, Qm, Vm = [data[key].values for key in ['t', 'Qm', 'Vm']]
    t *= 1e6  # us
    Qm *= 1e5  # nC/cm2
    Qrange = (Qm.min(), Qm.max())
    dQ = Qrange[1] - Qrange[0]

    # Create figure and axes
    fig, axes = plt.subplots(1, 2, figsize=cm2inch(17, 5))
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    # Plot Q-trace and V-trace
    ax = axes[0]
    for key in ['top', 'right']:
        ax.spines[key].set_visible(False)
    for key in ['bottom', 'left']:
        ax.spines[key].set_position(('axes', -0.03))
        ax.spines[key].set_linewidth(2)
    ax.plot(t, Vm, label='Vm', c='dimgrey', linewidth=lw)
    ax.plot(t, Qm, label='Qm', c='k', linewidth=lw)
    ax.add_patch(Rectangle(
        (t[0], Qrange[0] - 5), t[-1], dQ + 10,
        fill=False, edgecolor='k', linestyle='--', linewidth=1
    ))
    ax.yaxis.set_tick_params(width=2)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    # ax.set_xlim((t.min(), t.max()))
    ax.set_xticks([])
    ax.set_xlabel('{}s'.format(si_format((t.max()), space=' ')), fontsize=fs)
    ax.set_ylabel('$\\rm nC/cm^2$ - mV', fontsize=fs, labelpad=-15)
    ax.set_yticks(ax.get_ylim())
    for item in ax.get_yticklabels():
        item.set_fontsize(fs)

    # Plot inset on Q-trace
    ax = axes[1]
    for key in ['top', 'right', 'bottom', 'left']:
        ax.spines[key].set_linewidth(1)
        ax.spines[key].set_linestyle('--')
    ax.plot(t, Vm, label='Vm', c='dimgrey', linewidth=lw)
    ax.plot(t, Qm, label='Qm', c='k', linewidth=lw)
    ax.set_xlim((t.min(), t.max()))
    ax.set_xticks([])
    ax.set_yticks([])
    delta = 0.05
    ax.set_ylim(Qrange[0] - delta * dQ, Qrange[1] + delta * dQ)

    fig.canvas.set_window_title(figbase + 'b')
    return fig


def mechSim(bls, Fdrive, Adrive, Qm, fs=12, lw=2, ps=15):

    # Run mechanical simulation
    data, _ = bls.simulate(Fdrive, Adrive, Qm)
    t, Z, ng = [data[key].values for key in ['t', 'Z', 'ng']]

    # Create figure
    fig, ax = plt.subplots(figsize=cm2inch(7, 7))
    fig.suptitle('Mechanical simulation', fontsize=12)
    for skey in ['bottom', 'left', 'right', 'top']:
        ax.spines[skey].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # Plot variables and labels
    t_plot = np.insert(t, 0, -1e-6) * 1e6
    Pac = Adrive * np.sin(2 * np.pi * Fdrive * t + np.pi)  # Pa
    yvars = {'P_A': Pac * 1e-3, 'Z': Z * 1e9, 'n_g': ng * 1e22}
    colors = {'P_A': 'k', 'Z': 'C0', 'n_g': 'C5'}
    dy = 1.2
    for i, ykey in enumerate(yvars.keys()):
        y = yvars[ykey]
        y_plot = rescale(np.insert(y, 0, y[0])) - dy * i
        ax.plot(t_plot, y_plot, color=colors[ykey], linewidth=lw)
        ax.text(t_plot[0] - 0.1, y_plot[0], '$\mathregular{{{}}}$'.format(ykey), fontsize=fs,
                horizontalalignment='right', verticalalignment='center', color=colors[ykey])

    # Acoustic pressure annotations
    ax.annotate(s='', xy=(1.5, 1.1), xytext=(3.5, 1.1),
                arrowprops=dict(arrowstyle='<|-|>', color='k'))
    ax.text(2.5, 1.12, '1/f', fontsize=fs, color='k',
            horizontalalignment='center', verticalalignment='bottom')
    ax.annotate(s='', xy=(1.5, -0.1), xytext=(1.5, 1),
                arrowprops=dict(arrowstyle='<|-|>', color='k'))
    ax.text(1.55, 0.4, '2A', fontsize=fs, color='k',
            horizontalalignment='left', verticalalignment='center')

    # Periodic stabilization patch
    ax.add_patch(Rectangle((2, -2 * dy - 0.1), 2, 2 * dy, color='dimgrey', alpha=0.3))
    ax.text(3, -2 * dy - 0.2, 'limit cycle', fontsize=fs, color='dimgrey',
            horizontalalignment='center', verticalalignment='top')
    # Z_last patch
    ax.add_patch(Rectangle((2, -dy - 0.1), 2, dy, edgecolor='k', facecolor='none', linestyle='--'))

    # ngeff annotations
    c = plt.get_cmap('tab20').colors[11]
    ax.text(t_plot[-1] + 0.1, y_plot[-1], '$\mathregular{n_{g,eff}}$', fontsize=fs, color=c,
            horizontalalignment='left', verticalalignment='center')
    ax.scatter([t_plot[-1]], [y_plot[-1]], color=c, s=ps)

    fig.canvas.set_window_title(figbase + 'c mechsim')
    return fig


def cycleAveraging(bls, pneuron, Fdrive, Adrive, Qm, fs=12, lw=2, ps=15):

    # Run mechanical simulation
    data, _ = bls.simulate(Fdrive, Adrive, Qm)
    t, Z, ng = [data[key].values for key in ['t', 'Z', 'ng']]

    # Compute variables evolution over last acoustic cycle
    t_last = t[-NPC_DENSE:] * 1e6  # us
    Z_last = Z[-NPC_DENSE:]  # m
    Cm = bls.v_capacitance(Z_last) * 1e2  # uF/m2
    Vm = Qm / Cm * 1e5  # mV
    yvars = {
        'C_m': Cm,  # uF/cm2
        'V_m': Vm,  # mV
        '\\alpha_m': pneuron.alpham(Vm) * 1e3,  # ms-1
        '\\beta_m': pneuron.betam(Vm) * 1e3,  # ms-1
        'p_\\infty / \\tau_p': pneuron.pinf(Vm) / pneuron.taup(Vm) * 1e3,  # ms-1
        '(1-p_\\infty) / \\tau_p': (1 - pneuron.pinf(Vm)) / pneuron.taup(Vm) * 1e3  # ms-1
    }

    # Determine colors
    violets = plt.get_cmap('Paired').colors[8:10][::-1]
    oranges = plt.get_cmap('Paired').colors[6:8][::-1]
    colors = {
        'C_m': ['k', 'dimgrey'],
        'V_m': plt.get_cmap('tab20').colors[14:16],
        '\\alpha_m': violets,
        '\\beta_m': oranges,
        'p_\\infty / \\tau_p': violets,
        '(1-p_\\infty) / \\tau_p': oranges
    }

    # Create figure and axes
    fig, axes = plt.subplots(6, 1, figsize=cm2inch(4, 15))
    fig.suptitle('Cycle-averaging', fontsize=fs)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for skey in ['bottom', 'left', 'right', 'top']:
            ax.spines[skey].set_visible(False)

    # Plot variables
    for ax, ykey in zip(axes, yvars.keys()):
        ax.set_xticks([])
        ax.set_yticks([])
        for skey in ['bottom', 'left', 'right', 'top']:
            ax.spines[skey].set_visible(False)
        y = yvars[ykey]
        ax.plot(t_last, y, color=colors[ykey][0], linewidth=lw)
        ax.plot([t_last[0], t_last[-1]], [np.mean(y)] * 2, '--', color=colors[ykey][1])
        ax.scatter([t_last[-1]], [np.mean(y)], s=ps, color=colors[ykey][1])
        ax.text(t_last[0] - 0.1, y[0], '$\mathregular{{{}}}$'.format(ykey), fontsize=fs,
                horizontalalignment='right', verticalalignment='center', color=colors[ykey][0])

    fig.canvas.set_window_title(figbase + 'c cycleavg')
    return fig


def Qsolution(nbls, Fdrive, Adrive, tstim, toffset, PRF, DC, fs=12, lw=2, ps=15):

    # Run effective simulation
    data, _ = nbls.simulate(Fdrive, Adrive, tstim, toffset, PRF, DC, method='sonic')
    t, Qm, states = [data[key].values for key in ['t', 'Qm', 'stimstate']]
    t *= 1e3  # ms
    Qm *= 1e5  # nC/cm2
    tpulse_on, tpulse_off = GroupedTimeSeries.getStimPulses(_, t, states)

    # Add small onset
    t = np.insert(t, 0, -5.0)
    Qm = np.insert(Qm, 0, Qm[0])

    # Create figure and axes
    fig, ax = plt.subplots(figsize=cm2inch(12, 6))
    ax.set_xticks([])
    ax.set_yticks([])
    for key in ['top', 'right']:
        ax.spines[key].set_visible(False)
    for key in ['bottom', 'left']:
        ax.spines[key].set_position(('axes', -0.03))
        ax.spines[key].set_linewidth(2)

    # Plot Q-trace and stimulation pulses
    ax.plot(t, Qm, label='Qm', c='k', linewidth=lw)
    for ton, toff in zip(tpulse_on, tpulse_off):
        ax.axvspan(ton, toff, edgecolor='none', facecolor='#8A8A8A', alpha=0.2)
    ax.yaxis.set_tick_params(width=2)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax.set_xlim((t.min(), t.max()))
    ax.set_xticks([])
    ax.set_xlabel('{}s'.format(si_format((t.max()) * 1e-3, space=' ')), fontsize=fs)
    ax.set_ylabel('$\\rm nC/cm^2$', fontsize=fs, labelpad=-15)
    ax.set_yticks(ax.get_ylim())
    for item in ax.get_yticklabels():
        item.set_fontsize(fs)
    ax.legend(fontsize=fs, frameon=False)

    fig.canvas.set_window_title(figbase + 'e Qtrace')
    return fig


def main():
    parser = FigureParser(['a', 'b', 'c', 'e'])
    args = parser.parse()
    logger.setLevel(args['loglevel'])
    figset = args['subset']
    logger.info('Generating panels {} of {}'.format(figset, figbase))

    # Parameters
    pneuron = getPointNeuron('RS')
    a = 32e-9  # m
    Fdrive = 500e3  # Hz
    Adrive = 100e3  # Pa
    PRF = 100.  # Hz
    DC = 0.5
    tstim = 150e-3  # s
    toffset = 100e-3  # s
    Qm = -71.9e-5  # C/cm2
    bls = BilayerSonophore(a, pneuron.Cm0, pneuron.Qm0())
    nbls = NeuronalBilayerSonophore(a, pneuron)

    # Figures
    figs = []
    if 'a' in figset:
        figs.append(PmApprox(bls, np.linspace(-0.4 * bls.Delta_, bls.a, 1000)))
    if 'b' in figset:
        figs.append(recasting(nbls, Fdrive, Adrive))
    if 'c' in figset:
        figs += [
            mechSim(bls, Fdrive, Adrive, Qm),
            cycleAveraging(bls, pneuron, Fdrive, Adrive, Qm)
        ]
    if 'e' in figset:
        figs.append(Qsolution(nbls, Fdrive, Adrive, tstim, toffset, PRF, DC))

    if args['save']:
        for fig in figs:
            figname = '{}.pdf'.format(fig.canvas.get_window_title())
            fig.savefig(os.path.join(args['outputdir'], figname), transparent=True)
    else:
        plt.show()


if __name__ == '__main__':
    main()
