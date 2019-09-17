# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2018-06-06 18:38:04
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-09-17 17:29:33

''' Generate the data necessary to produce the paper figures. '''

import os
import logging
import numpy as np
import pandas as pd

from PySONIC.core import NeuronalBilayerSonophore
from PySONIC.utils import *
from PySONIC.neurons import getPointNeuron
from PySONIC.core import Batch


def comparisons(outdir, mpi, loglevel, overwrite):
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
    radii = np.logspace(np.log10(16), np.log10(64), 5) * 1e-9  # m

    output = []

    # Get RS neuron threshold amplitudes as as function of US frequency and sonophore radius
    pneuron = getPointNeuron('RS')
    nbls = NeuronalBilayerSonophore(a, pneuron)
    fkey = 'Fdrive (kHz)'
    akey = 'radius (nm)'
    Akey = 'Athr (kPa)'
    RS_Athr_vs_freq = pd.DataFrame(
        data={Akey: np.array([nbls.titrate(x, tstim, toffset) * 1e-3 for x in freqs])},
        index=freqs * 1e-3)
    RS_Athr_vs_radius = pd.DataFrame(
        data={Akey: np.array([
            NeuronalBilayerSonophore(x, pneuron).titrate(Fdrive, tstim, toffset) * 1e-3 for x in radii])},
        index=radii * 1e9)

    # Save intermediate results in sub-directory
    RS_Athr_vs_freq.to_csv(os.path.join(outdir, 'RS_Athr_vs_freq.csv'), index_label=fkey)
    RS_Athr_vs_radius.to_csv(os.path.join(outdir, 'RS_Athr_vs_radius.csv'), index_label=akey)

    # CW simulations with RS neuron
    pneuron = getPointNeuron('RS')
    nbls = NeuronalBilayerSonophore(a, pneuron)

    # Span US amplitude and US frequency ranges
    Athr = RS_Athr_vs_freq[Akey].loc[Fdrive * 1e-3]  # kPa
    amps = np.array([Athr - 5., Athr, Athr + 20., 50, 100, 300, 600]) * 1e3  # Pa
    queue = nbls.simQueue([Fdrive], amps, [tstim], [toffset], [PRF], [DC], [fs], methods,
                          outputdir=outdir, overwrite=overwrite)
    for x in freqs:
        Athr = RS_Athr_vs_freq[Akey].loc[x * 1e-3]  # kPa
        Adrive = (Athr + 20.) * 1e3  # Pa
        queue += nbls.simQueue([x], [Adrive], [tstim], [toffset], [PRF], [DC], [fs], methods,
                               outputdir=outdir, overwrite=overwrite)
    output += Batch(nbls.simAndSave, queue).run(mpi=mpi, loglevel=loglevel)

    # Span sonophore radius frequency range
    for x in radii:
        Athr = RS_Athr_vs_radius[Akey].loc[x * 1e9]  # kPa
        Adrive = (Athr + 20.) * 1e3  # Pa
        nbls = NeuronalBilayerSonophore(x, pneuron)
        queue = nbls.simQueue([Fdrive], [Adrive], [tstim], [toffset], [PRF], [DC], [fs], methods,
                              outputdir=outdir, overwrite=overwrite)
        output += Batch(nbls.simAndSave, queue).run(mpi=mpi, loglevel=loglevel)

    # Span DC range with RS and LTS neurons
    neurons = ['RS', 'LTS']
    DCs = np.array([5, 10, 25, 50, 75, 100]) * 1e-2
    for neuron in neurons:
        pneuron = getPointNeuron(neuron)
        nbls = NeuronalBilayerSonophore(a, pneuron)
        queue = nbls.simQueue([Fdrive], [Adrive], [tstim], [toffset], [PRF], DCs, [fs], methods,
                              outputdir=outdir, overwrite=overwrite)
        output += Batch(nbls.simAndSave, queue).run(mpi=mpi, loglevel=loglevel)

    # Span PRF range with LTS neuron
    neuron = 'LTS'
    pneuron = getPointNeuron(neuron)
    nbls = NeuronalBilayerSonophore(a, pneuron)
    PRFs_orders = np.logspace(1, 4, 4)  # Hz
    PRFs = sum([[x, 2 * x, 5 * x] for x in PRFs_orders[:-1]], []) + [PRFs_orders[-1]]  # Hz
    DC = 0.05
    queue = nbls.simQueue([Fdrive], [Adrive], [tstim], [toffset], PRFs, [DC], [fs], methods,
                          outputdir=outdir, overwrite=overwrite)
    output += Batch(nbls.simAndSave, queue).run(mpi=mpi, loglevel=loglevel)

    return output


def maps(outdir, mpi, loglevel, overwrite):
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

    output = []

    # Span DC x Amplitude space for each neuron and PRF
    for neuron in neurons:
        pneuron = getPointNeuron(neuron)
        nbls = NeuronalBilayerSonophore(a, pneuron)
        for PRF in PRFs:
            code = '{} {}Hz PRF{}Hz {}s'.format(
                pneuron.name, *si_format([Fdrive, PRF, tstim], space=''))
            suboutdir = os.path.join(outdir, code)
            queue = nbls.simQueue([Fdrive], [Adrive], [tstim], [toffset], [PRF], [DC], [fs], ['sonic'],
                                  outputdir=suboutdir, overwrite=overwrite)
            output += Batch(nbls.simAndSave, queue).run(mpi=mpi, loglevel=loglevel)

    return output


def getThresholdAmplitudes(outdir, neuron, a, Fdrive, tstim, toffset, PRF, DCs):
    pneuron = getPointNeuron(neuron)
    nbls = NeuronalBilayerSonophore(a, pneuron)
    fcode = 'Athrs_{}_{}m_{}Hz_PRF{}Hz_{}s'.format(
        pneuron.name, *si_format([a, Fdrive, PRF, tstim], space=''))
    queue = nbls.simQueue([Fdrive], [None], [tstim], [toffset], [PRF], DCs, [fs], ['sonic'])
    outputs = Batch(nbls.simulate, queue).run(mpi=mpi, loglevel=loglevel)
    Athrs = np.array([item[0]['Adrive'] for item in outputs])
    df = pd.DataFrame({'DC': DCs, 'Athr': Athrs})
    df.to_csv(os.path.join(outdir, f'{fcode}.csv'), index=False)
    return df


def thresholds(outdir, mpi, loglevel, overwrite):
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
    for neuron in neurons:
        for x in freqs:
            getThresholdAmplitudes(outdir, neuron, a, x, tstim, toffset, PRF, DCs)
        for x in radii:
            getThresholdAmplitudes(outdir, neuron, x, Fdrive, tstim, toffset, PRF, DCs)


def STN(outdir, mpi, loglevel, overwrite):
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
    nbls = NeuronalBilayerSonophore(a, pneuron)
    queue = nbls.simQueue([Fdrive], amps, [tstim], [toffset], [PRF], [DC], [fs], ['sonic'],
                          outputdir=outdir, overwrite=overwrite)
    output = Batch(nbls.simAndSave, queue).run(mpi=mpi, loglevel=loglevel)

    return output


if __name__ == '__main__':

    # Global parameters
    loglevel = logging.INFO
    logger.setLevel(loglevel)
    mpi = True
    overwrite = False
    data_root = selectDirDialog(title='Select data root directory')

    # Run batches to generate data
    logger.info('Generating SONIC paper data')
    dash = '---------------------------------------------------'
    funcs = [comparisons, maps, thresholds, STN]
    outdirs = [os.path.join(data_root, func.__name__) for func in funcs]
    for outdir in outdirs:
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
    for func, outdir in zip(funcs, outdirs):
        print(f'{dash} {func.__name__} {dash}')
        func(outdir, mpi, loglevel, overwrite)
        print(f'{dash}{dash}')
