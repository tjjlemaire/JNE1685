# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2017-02-15 15:59:37
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-06-17 21:55:47

''' Sub-panels of the effective variables figure. '''

import os
import matplotlib
import matplotlib.pyplot as plt

from PySONIC.plt import plotEffectiveVariables
from PySONIC.utils import logger
from PySONIC.neurons import getPointNeuron
from PySONIC.parsers import FigureParser

# Plot parameters
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'arial'

# Figure basename
figbase = os.path.splitext(__file__)[0]


def main():
    parser = FigureParser(['a', 'b', 'c'])
    args = parser.parse()
    logger.setLevel(args['loglevel'])
    figset = args['subset']
    logger.info('Generating panels {} of {}'.format(figset, figbase))

    # Parameters
    pneuron = getPointNeuron('RS')
    a = 32e-9  # m
    Fdrive = 500e3  # Hz
    Adrive = 50e3  # Pa

    # Generate figures
    figs = []
    if 'a' in figset:
        fig = plotEffectiveVariables(pneuron, a=a, Fdrive=Fdrive, cmap='Oranges', zscale='log')
        fig.canvas.set_window_title(figbase + 'a')
        figs.append(fig)
    if 'b' in figset:
        fig = plotEffectiveVariables(pneuron, a=a, Adrive=Adrive, cmap='Greens', zscale='log')
        fig.canvas.set_window_title(figbase + 'b')
        figs.append(fig)
    if 'c' in figset:
        fig = plotEffectiveVariables(
            pneuron, Fdrive=Fdrive, Adrive=Adrive, cmap='Blues', zscale='log')
        fig.canvas.set_window_title(figbase + 'c')
        figs.append(fig)

    if args['save']:
        for fig in figs:
            figname = '{}.pdf'.format(fig.canvas.get_window_title())
            fig.savefig(os.path.join(args['outputdir'], figname), transparent=True)
    else:
        plt.show()


if __name__ == '__main__':
    main()
