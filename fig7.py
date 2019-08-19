# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2018-09-26 09:51:43
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-06-17 22:03:38

''' Sub-panels of (duty-cycle x amplitude) US activation maps and related Q-V traces. '''

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from PySONIC.core import NeuronalBilayerSonophore
from PySONIC.utils import logger, si_format
from PySONIC.plt import ActivationMap
from PySONIC.neurons import getPointNeuron
from PySONIC.parsers import FigureParser

# Plot parameters
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'arial'

# Figure basename
figbase = os.path.splitext(__file__)[0]


def plotMapAndTraces(inputdir, pneuron, a, Fdrive, tstim, amps, PRF, DCs, FRbounds,
                     insets, tbounds, Vbounds, prefix):
    # Activation map
    mapcode = '{} {}Hz PRF{}Hz 1s'.format(pneuron.name, *si_format([Fdrive, PRF, tstim], space=''))
    subdir = os.path.join(inputdir, mapcode)
    actmap = ActivationMap(subdir, pneuron, a, Fdrive, tstim, PRF, amps, DCs)
    mapfig = actmap.render(FRbounds=FRbounds, thresholds=True)
    mapfig.canvas.set_window_title('{} map {}'.format(prefix, mapcode))
    ax = mapfig.axes[0]
    DC_insets, A_insets = zip(*insets)
    ax.scatter(DC_insets, A_insets, s=80, facecolors='none', edgecolors='k', linestyle='--')

    # Related inset traces
    tracefigs = []
    nbls = NeuronalBilayerSonophore(a, pneuron)
    for inset in insets:
        DC = inset[0] * 1e-2
        Adrive = inset[1] * 1e3
        fname = '{}.pkl'.format(nbls.filecode(
            Fdrive, actmap.correctAmp(Adrive), tstim, 0., PRF, DC, 'sonic'))
        fpath = os.path.join(subdir, fname)
        tracefig = actmap.plotQVeff(fpath, tbounds=tbounds, ybounds=Vbounds)
        figcode = '{} VQ trace {} {:.1f}kPa {:.0f}%DC'.format(
            prefix, pneuron.name, Adrive * 1e-3, DC * 1e2)
        tracefig.canvas.set_window_title(figcode)
        tracefigs.append(tracefig)

    return mapfig, tracefigs


def panel(inputdir, pneurons, a, tstim, PRF, amps, DCs, FRbounds, tbounds, Vbounds, insets, prefix):

    mapfigs, tracefigs = [], []
    for pn in pneurons:
        out = plotMapAndTraces(
            inputdir, pn, a, 500e3, tstim, amps, PRF, DCs,
            FRbounds, insets[pn.name], tbounds, Vbounds, prefix)
        mapfigs.append(out[0])
        tracefigs += out[1]

    return mapfigs + tracefigs


def main():
    parser = FigureParser(['a', 'b', 'c'])
    parser.addInputDir()
    args = parser.parse()
    logger.setLevel(args['loglevel'])
    figset = args['subset']
    inputdir = args['inputdir']
    logger.info('Generating panels {} of {}'.format(figset, figbase))

    # Parameters
    pneurons = [getPointNeuron(n) for n in ['RS', 'LTS']]
    a = 32e-9  # m
    tstim = 1.0  # s
    amps = np.logspace(np.log10(10), np.log10(600), num=30) * 1e3  # Pa
    DCs = np.arange(1, 101) * 1e-2
    FRbounds = (1e0, 1e3)  # Hz
    tbounds = (0, 240e-3)  # s
    Vbounds = -150, 50  # mV

    # Generate figures
    try:
        figs = []
        if 'a' in figset:
            PRF = 1e1
            insets = {
                'RS': [(28, 127.0), (37, 168.4)],
                'LTS': [(8, 47.3), (30, 146.2)]
            }
            figs += panel(inputdir, pneurons, a, tstim, PRF, amps, DCs, FRbounds, tbounds, Vbounds,
                          insets, figbase + 'a')
        if 'b' in figset:
            PRF = 1e2
            insets = {
                'RS': [(51, 452.4), (56, 452.4)],
                'LTS': [(13, 193.9), (43, 257.2)]
            }
            figs += panel(inputdir, pneurons, a, tstim, PRF, amps, DCs, FRbounds, tbounds, Vbounds,
                          insets, figbase + 'b')
        if 'c' in figset:
            PRF = 1e3
            insets = {
                'RS': [(40, 110.2), (64, 193.9)],
                'LTS': [(10, 47.3), (53, 168.4)]
            }
            figs += panel(inputdir, pneurons, a, tstim, PRF, amps, DCs, FRbounds, tbounds, Vbounds,
                          insets, figbase + 'c')

    except Exception as e:
        logger.error(e)
        return

    if args['save']:
        for fig in figs:
            figname = '{}.pdf'.format(fig.canvas.get_window_title())
            fig.savefig(os.path.join(args['outputdir'], figname), transparent=True)
    else:
        plt.show()


if __name__ == '__main__':
    main()
