# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-04-28 21:35:27
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-04-28 22:22:41

''' Run all the notebooks to produce the data and figures. '''

import glob
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

ep = ExecutePreprocessor(timeout=-1, kernel_name='python3')


def executeNotebook(fname):
    with open(fname) as f:
        nb = nbformat.read(f, as_version=4)
    ep.preprocess(nb, {})
    with open(fname, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)


# Gather all the notebooks in the directory
notebooks = list(glob.glob('*.ipynb'))

# Execute each notebook sequentially
for notebook in notebooks:
    print(f'running {notebook} notebook')
    executeNotebook(notebook)
