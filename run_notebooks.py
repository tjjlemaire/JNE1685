# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-04-28 21:35:27
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-05-02 14:28:47

''' Run all the notebooks to produce the data and figures. '''

import logging
import re
import glob
from argparse import ArgumentParser
from PySONIC.utils import logger

from notebook_runner import runNotebook

# Set logging level
logger.setLevel(logging.INFO)

# Gather all the notebooks in the directory
notebooks = sorted(list(glob.glob('*.ipynb')))

# Associate each notebook for a figure index
notebook_pattern = re.compile(f'^figure_([0-9]+).ipynb$')
notebooks = {int(notebook_pattern.match(n).group(1)): n for n in notebooks}
valid_indexes = list(notebooks.keys())


def main():
    # Parse command line arguments
    ap = ArgumentParser()
    ap.add_argument(
        '-f', '--figure', type=str, nargs='+', default='all', help='Figure index')
    ap.add_argument(
        '-s', '--save', default=False, action='store_true', help='Save in-place')
    args = ap.parse_args()
    figindexes = args.figure
    save = args.save
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
        runNotebook(notebook, save=save)


if __name__ == '__main__':
    main()
