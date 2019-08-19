# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2018-10-01 20:45:29
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-07-01 09:38:52

import os
import numpy as np
import pandas as pd

from PySONIC.utils import *
from PySONIC.core import NeuronalBilayerSonophore
from PySONIC.neurons import *
from PySONIC.postpro import computeSpikingMetrics


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
                df.loc[Fdrive * 1e-3, neuron] = np.ceil(Athr * 1e-2) / 10
    df.sort_index(inplace=True)
    df.to_csv(fpath, sep=',', index_label=fkey)
    return df


def getCWtitrations_vs_radius(neurons, radii, Fdrive, tstim, toffset, fpath):
    akey = 'radius (nm)'
    radii = np.array(radii)
    if os.path.isfile(fpath):
        df = pd.read_csv(fpath, sep=',', index_col=akey)
    else:
        df = pd.DataFrame(index=radii * 1e9)
    for neuron in neurons:
        if neuron not in df:
            pneuron = getPointNeuron(neuron)
            for a in radii:
                nbls = NeuronalBilayerSonophore(a, pneuron)
                logger.info(
                    'Running CW titration for %s neuron @ %sHz (%.2f nm sonophore radius)',
                    neuron, si_format(Fdrive), a * 1e9)
                Athr = nbls.titrate(Fdrive, tstim, toffset)  # Pa
                df.loc[a * 1e9, neuron] = np.ceil(Athr * 1e-2) / 10
    df.sort_index(inplace=True)
    df.to_csv(fpath, sep=',', index_label=akey)
    return df


def getSims(outdir, neuron, a, queue):
    fpaths = []
    updated_queue = []
    pneuron = getPointNeurons(neuron)
    nbls = NeuronalBilayerSonophore(a, pneuron)
    for i, item in enumerate(queue):
        Fdrive, tstim, toffset, PRF, DC, Adrive, method = item
        fcode = nbls.filecode(Fdrive, Adrive, tstim, toffset, PRF, DC, method)
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


def getSpikingMetrics(outdir, neuron, xvar, xkey, data_fpaths, metrics_fpaths):
    metrics = {}
    for stype in data_fpaths.keys():
        if os.path.isfile(metrics_fpaths[stype]):
            logger.info('loading spiking metrics from file: "%s"', metrics_fpaths[stype])
            metrics[stype] = pd.read_csv(metrics_fpaths[stype], sep=',')
        else:
            logger.warning('computing %s spiking metrics vs. %s for %s neuron', stype, xkey, neuron)
            metrics[stype] = computeSpikingMetrics(data_fpaths[stype])
            metrics[stype][xkey] = pd.Series(xvar, index=metrics[stype].index)
            metrics[stype].to_csv(metrics_fpaths[stype], sep=',', index=False)
    return metrics


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
