# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-04-28 21:35:27
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-04-29 13:22:58

''' Run all the notebooks to produce the data and figures. '''

import logging
import re
import glob
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from argparse import ArgumentParser
from PySONIC.utils import logger

# Set logging level
logger.setLevel(logging.INFO)

# Create notebook processor object
ep = ExecutePreprocessor(timeout=-1, kernel_name='python3')

# Gather all the notebooks in the directory
notebooks = sorted(list(glob.glob('*.ipynb')))

# Associate each notebook for a figure index
notebook_pattern = re.compile(f'^figure_([0-9]+).ipynb$')
notebooks = {int(notebook_pattern.match(n).group(1)): n for n in notebooks}
valid_indexes = list(notebooks.keys())


def executeNotebook(fname):
    ''' Function to open, execute and save "in-place" a notebook. '''
    with open(fname) as f:
        nb = nbformat.read(f, as_version=4)
    ep.preprocess(nb, {})
    with open(fname, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)


def main():
    # Parse command line arguments
    ap = ArgumentParser()
    ap.add_argument(
        '-f', '--figure', type=str, nargs='+', default='all', help='Figure index')
    figindexes = ap.parse_args().figure

    try:
        # Check validity of provided figure indexes
        if figindexes == 'all' or figindexes == ['all']:
            figindexes = valid_indexes
        else:
            if 'all' in figindexes:
                raise ValueError('"all" cannot be provided with other figure indexes.')
            figindexes = sorted([int(s) for s in figindexes])
            for i in figindexes:
                if i not in valid_indexes:
                    raise ValueError(
                        f'{i} is not a valid figure index. Options are {valid_indexes}.')
    except Exception as e:
        logger.error(e)
        quit()

    # Construct list of selected notebooks and execute them sequentially
    logger.info(f'selected figures: {figindexes}')
    selected_notebooks = [notebooks[i] for i in figindexes]
    for notebook in selected_notebooks:
        logger.info(f'running {notebook} notebook')
        executeNotebook(notebook)


if __name__ == '__main__':
    main()
