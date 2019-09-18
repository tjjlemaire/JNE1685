# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2018-06-06 18:38:04
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-09-18 18:10:22

''' Generate the data necessary to produce the paper figures. '''

import os
import logging
import numpy as np
import pandas as pd

from PySONIC.core import NeuronalBilayerSonophore
from PySONIC.utils import *
from PySONIC.neurons import getPointNeuron
from PySONIC.core import Batch


def getQueue(a, pneuron, *sim_args, outputdir=None, overwrite=False):
    sim_args = list(sim_args)
    for i, arg in enumerate(sim_args):
        if arg is not None and not isIterable(arg):
            sim_args[i] = [arg]
    nbls = NeuronalBilayerSonophore(a, pneuron)
    queue = nbls.simQueue(*sim_args, outputdir=outputdir, overwrite=overwrite)
    return [([NeuronalBilayerSonophore, a, pneuron] + params[0], params[1]) for params in queue]


def init_sim_save(cls, a, pneuron, *args, **kwargs):
    return cls(a, pneuron).simAndSave(*args, **kwargs)


def codes(a, pneuron, Fdrive, PRF, tstim):
    return [
        pneuron.name,
        f'{si_format(a, space="")}m',
        f'{si_format(Fdrive, space="")}Hz',
        f'PRF{si_format(PRF, space="")}Hz',
        f'{si_format(tstim, space="")}s'
    ]


def comparisons(outdir, overwrite):
    ''' Comparison of predictions of the Full NICE vs coarse-grained SONIC models
        of different neurons across the LIFUS parameter range.
    '''

    # Parameters
    a = 32e-9  # m
    Fdrive = 500e3  # Hz
    Adrive = 100e3  # Pa
    tstim = 150e-3  # s
    toffset = 100e-3  # s
    PRF = 100  # Hz
    DC = 1.
    fs = 1.
    methods = ['sonic', 'full']
    freqs = np.array([20e3, 100e3, 500e3, 1e6, 2e6, 3e6, 4e6])  # Hz
    radii = np.round(np.logspace(np.log10(0.5 * a), np.log10(2 * a), 5), 10)  # m

    # Point-neuron objects
    RS = getPointNeuron('RS')
    LTS = getPointNeuron('LTS')

    # Get RS neuron threshold amplitudes as as function of US frequency and sonophore radius
    pneuron = RS
    nbls = NeuronalBilayerSonophore(a, pneuron)
    Akey = 'Athr (kPa)'
    RS_Athr_vs_freq = pd.DataFrame(
        data={Akey: np.array([nbls.titrate(x, tstim, toffset) * 1e-3 for x in freqs])},
        index=freqs * 1e-3)
    RS_Athr_vs_radius = pd.DataFrame(
        data={Akey: np.array([NeuronalBilayerSonophore(x, pneuron).titrate(Fdrive, tstim, toffset) * 1e-3 for x in radii])},
        index=radii * 1e9)
    RS_Athr_vs_freq.to_csv(
        os.path.join(outdir, f'{pneuron.name}_Athr_vs_freq.csv'), index_label='Fdrive (kHz)')
    RS_Athr_vs_radius.to_csv(
        os.path.join(outdir, f'{pneuron.name}_Athr_vs_radius.csv'), index_label='radius (nm)')

    # Initialize queue
    queue = []

    # CW simulations: span US amplitude, US frequency and sonophore radius ranges with RS neuron
    pneuron = RS
    Athr = RS_Athr_vs_freq[Akey].loc[Fdrive * 1e-3]  # kPa
    amps = np.array([Athr - 5., Athr, Athr + 20., 50, 100, 300, 600]) * 1e3  # Pa
    queue += getQueue(a, pneuron, Fdrive, amps, tstim, toffset, PRF, DC, fs, methods,
                      outputdir=outdir, overwrite=overwrite)
    for x in freqs:
        Athr = RS_Athr_vs_freq[Akey].loc[x * 1e-3]  # kPa
        queue += getQueue(a, pneuron, x, (Athr + 20.) * 1e3, tstim, toffset, PRF, DC, fs, methods,
                          outputdir=outdir, overwrite=overwrite)
    for x in filter(lambda x: x != a, radii):
        Athr = RS_Athr_vs_radius[Akey].loc[x * 1e9]  # kPa
        queue += getQueue(x, pneuron, Fdrive, (Athr + 20.) * 1e3, tstim, toffset, PRF, DC, fs, methods,
                          outputdir=outdir, overwrite=overwrite)

    # PW simulations: span DC range with RS and LTS neurons
    DCs = np.array([5, 10, 25, 50, 75, 100]) * 1e-2
    for pneuron in [RS, LTS]:
        queue += getQueue(a, pneuron, Fdrive, Adrive, tstim, toffset, PRF, DCs, fs, methods,
                          outputdir=outdir, overwrite=overwrite)

    # PW simulations: span PRF range with LTS neuron
    pneuron = LTS
    PRFs_orders = np.logspace(1, 4, 4)  # Hz
    PRFs = sum([[x, 2 * x, 5 * x] for x in PRFs_orders[:-1]], []) + [PRFs_orders[-1]]  # Hz
    DC = 0.05
    queue += getQueue(a, pneuron, Fdrive, Adrive, tstim, toffset, PRFs, DC, fs, methods,
                      outputdir=outdir, overwrite=overwrite)

    # Return batch object
    return Batch(init_sim_save, queue)


def maps(outdir, overwrite):
    ''' Duty cycle x amplitude maps of the neural firing rates of various neurons at various
        pulse-repetition frequencies, predicted by the SONIC model.
    '''

    # Parameters
    neurons = ['RS', 'LTS']
    a = 32e-9  # m
    Fdrive = 500e3  # Hz
    tstim = 1.0  # s
    toffset = 0  # s
    amps = np.logspace(np.log10(10), np.log10(600), num=30) * 1e3  # Pa
    DCs = np.arange(1, 101) * 1e-2
    PRFs = np.logspace(1, 3, 3)  # Hz
    fs = 1.

    # Span DC x Amplitude space for each neuron and PRF
    queue = []
    for neuron in neurons:
        pneuron = getPointNeuron(neuron)
        for PRF in PRFs:
            suboutdir = os.path.join(outdir, ' '.join(codes(a, pneuron, Fdrive, PRF, tstim)))
            if not os.path.isdir(suboutdir):
                os.mkdir(suboutdir)
            queue += getQueue(a, pneuron, Fdrive, amps, tstim, toffset, PRF, DCs, fs, 'sonic',
                              outputdir=suboutdir, overwrite=overwrite)

    # Return batch object
    return Batch(init_sim_save, queue)


def thresholds(outdir, overwrite):
    ''' Duty-cycle-dpendent excitation threshold amplitudes of several neurons for various
        US frequencies and sonophore radii.
    '''

    # Parameters
    neurons = ['RS', 'LTS']
    radii = np.array([16, 32, 64]) * 1e-9  # m
    freqs = np.array([20, 500, 4000]) * 1e3  # Hz
    a = radii[1]
    Fdrive = freqs[1]
    tstim = 1.  # s
    toffset = 0.  # s
    PRF = 100.  # Hz
    fs = 1.
    DCs = np.arange(1, 101) * 1e-2

    # Titrate over DC range for different US frequencies and sonophore radii
    queue = []
    for neuron in neurons:
        pneuron = getPointNeuron(neuron)
        for x in freqs:
            suboutdir = os.path.join(outdir, ' '.join(codes(a, pneuron, x, PRF, tstim)))
            if not os.path.isdir(suboutdir):
                os.mkdir(suboutdir)
            queue += getQueue(a, pneuron, x, None, tstim, toffset, PRF, DCs, fs, 'sonic',
                              outputdir=suboutdir, overwrite=overwrite)
        for x in radii:
            suboutdir = os.path.join(outdir, ' '.join(codes(x, pneuron, Fdrive, PRF, tstim)))
            if not os.path.isdir(suboutdir):
                os.mkdir(suboutdir)
            queue += getQueue(x, pneuron, Fdrive, None, tstim, toffset, PRF, DCs, fs, 'sonic',
                              outputdir=suboutdir, overwrite=overwrite)

    # Return batch object
    return Batch(init_sim_save, queue)


def STN(outdir, overwrite):
    ''' Behavior of the STN neuron under CW sonication for various amplitudes. '''

    # Parameters
    pneuron = getPointNeuron('STN')
    a = 32e-9  # m
    Fdrive = 500e3  # Hz
    tstim = 1  # s
    toffset = 0.  # s
    PRF = 1e2
    DC = 1.
    fs = 1.
    intensities = np.hstack((
        np.arange(10, 101, 10),
        np.arange(101, 131, 1),
        np.array([140])
    ))  # W/m2
    amps = Intensity2Pressure(intensities)  # Pa

    # Span low-intensity range
    queue = getQueue(a, pneuron, Fdrive, amps, tstim, toffset, PRF, DC, fs, 'sonic',
                     outputdir=outdir, overwrite=overwrite)

    # Return batch object
    return Batch(init_sim_save, queue)


def fill(text, char='-', totlength=100):
    ndashes = totlength - len(text) - 2
    nside = ndashes // 2
    nleft, nright = nside, nside
    if ndashes % 2 == 1:
        nright += 1
    return f'{char * nleft} {text} {char * nright}'


if __name__ == '__main__':

    # Global parameters
    loglevel = logging.INFO
    logger.setLevel(loglevel)
    funcs = [comparisons, maps, thresholds, STN]
    mpi = True
    overwrite = False
    data_root = selectDirDialog(title='Select data root directory')

    logger.info('Creating output sub-directories')
    outdirs = [os.path.join(data_root, func.__name__) for func in funcs]
    for outdir in outdirs:
        if not os.path.isdir(outdir):
            os.mkdir(outdir)

    logger.info(f'Defining job batches')
    batches = {}
    for func, outdir in zip(funcs, outdirs):
        batches[func.__name__] = func(outdir, overwrite)

    logger.info(f'Running batches to generate data')
    output = []
    for name, batch in batches.items():
        logger.info(fill(f'{name} ({len(batch.queue)} jobs)'))
        output += batch.run(mpi=mpi, loglevel=loglevel)

    logger.info('All batches completed')
