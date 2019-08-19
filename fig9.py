# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2018-12-09 12:06:01
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-06-17 22:06:53

''' Sub-panels of SONIC model validation on an STN neuron (response to CW sonication). '''

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from PySONIC.core import NeuronalBilayerSonophore
from PySONIC.neurons import getPointNeuron
from PySONIC.utils import logger, Intensity2Pressure
from PySONIC.plt import CompTimeSeries, GroupedTimeSeries
from PySONIC.parsers import FigureParser

# Plot parameters
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'arial'

# Figure basename
figbase = os.path.splitext(__file__)[0]


def main():
    parser = FigureParser(['a', 'b'])
    parser.addInputDir()
    args = parser.parse()
    logger.setLevel(args['loglevel'])
    figset = args['subset']
    inputdir = args['inputdir']
    logger.info('Generating panels {} of {}'.format(figset, figbase))

    # Parameters
    pneuron = getPointNeuron('STN')
    a = 32e-9  # m
    Fdrive = 500e3  # Hz
    tstim = 1  # s
    toffset = 0.  # s
    PRF = 1e2
    DC = 1.
    nbls = NeuronalBilayerSonophore(a, pneuron)

    # Range of intensities
    intensities = pneuron.getLowIntensities()  # W/m2

    # Levels depicted with individual traces
    subset_intensities = [112, 114, 123]  # W/m2

    # convert to amplitudes and get filepaths
    amplitudes = Intensity2Pressure(intensities)  # Pa
    fnames = ['{}.pkl'.format(nbls.filecode(Fdrive, A, tstim, toffset, PRF, DC, 'sonic'))
              for A in amplitudes]
    fpaths = [os.path.join(inputdir, 'STN', fn) for fn in fnames]

    # Generate figures
    figs = []
    if 'a' in figset:
        comp_plot = CompTimeSeries(fpaths, 'FR')
        fig = comp_plot.render(
            patches='none',
            cmap='Oranges',
            tbounds=(0, tstim)
        )
        fig.canvas.set_window_title(figbase + 'a')
        figs.append(fig)
    if 'b' in figset:
        isubset = [np.argwhere(intensities == x)[0][0] for x in subset_intensities]
        subset_amplitudes = amplitudes[isubset]
        titles = ['{:.2f} kPa ({:.0f} W/m2)'.format(A * 1e-3, I)
                  for A, I in zip(subset_amplitudes, subset_intensities)]
        figtraces = GroupedTimeSeries([fpaths[i] for i in isubset], pltscheme={'Q_m': ['Qm']})()
        for fig, title in zip(figtraces, titles):
            fig.axes[0].set_title(title)
            fig.canvas.set_window_title(figbase + 'b {}'.format(title))
            figs.append(fig)

    if args['save']:
        for fig in figs:
            s = fig.canvas.get_window_title()
            s = s.replace('(', '- ').replace('/', '_').replace(')', '')
            figname = '{}.pdf'.format(s)
            fig.savefig(os.path.join(args['outputdir'], figname), transparent=True)
    else:
        plt.show()


if __name__ == '__main__':
    main()
