# Description

This repository contains the Python scripts allowing to generate the figures of the **multi-Scale Optimized Neuronal Cavitation (SONIC) model** paper [1].

## Content of repository

- `figure_**.ipynb`: notebooks used to generate the paper figures that result from model simulations (i.e. all except the schematic figures 1 and 3).
- `run_notebooks.py`: script used to run the notebooks required to generate the figures, from the command line.
- `utils.py`: module containing utilities functions used in the notebooks.

# Requirements

- Python 3.6+
- NEURON 7.x (https://neuron.yale.edu/neuron/download/)
- PySONIC package (https://c4science.ch/diffusion/4670/)
- ExSONIC package (https://c4science.ch/diffusion/7145/)
- nbconvert and nbformat python packages (utilities for jupyter notebooks)

# Installation

- Install a Python distribution
- Install a NEURON distribution
- Download the PySONIC AND ExSONIC code bases from their repositories, and follow the README instructions to install them as packages.
- Install the required python dependencies to run the notebooks: `pip install -r requirements.txt`

# Usage

## Create a data directory

First, you must create a directory on your machine to hold the generated data. Once this is done, open the `root.py` and specify the full path to your data directory (replacing `None`).

## Getting/generating the data

Given the cumbersome model simulations required to create the figures, it is advised to run the `run_notebooks.py` script in order to generate the required dataset before opening and running the notebooks. By default, that script generates the data for all the figures, but you can specify a subset of your choice using the `-f` option.

For instance, to generate data uniquely for figure 4:
```
python run_all.py -f 4
```

To generate data uniquely for figures 4, 5 & 6:
```
python run_all.py -f 4 5 6
```

To generate data for all figures:
```
python run_all.py -f all
```

Be aware that the **cumulated computation time required to run all simulations can easily exceed 1 week**, and that the **total size of entire dataset size is about 112 GB**. Therefore, it is highly advised that you run that script on a **high-performance, multi-core machine with enough disk space**.

The generated dataset should be split between 4 sub-folders in the indicated output directory:
- *comparisons*
- *maps*
- *STN*
- *coverage*

## Generating the figures

To generate a figure:

- start a *jupyter notebook* / *jupyter lab* session:

`jupyter lab` / `jupyter notebook`

- open the figure notebook
- select all the cells (`Ctrl` + `A`) and run them (`Ctrl` + `Enter`)
- wait for the complete notebook execution

Upon completion, the figures panels should appear in the notebook. Additionally, they will be saved as PDFs in a *figs* sub-folder.

# Authors

Code written and maintained by Theo Lemaire (theo.lemaire@epfl.ch).

# License

This project is licensed under the MIT License - see the LICENSE file for details.

# References

[1] Lemaire, T., Neufeld, E., Kuster, N., and Micera, S. (2019). Understanding ultrasound neuromodulation using a computationally efficient and interpretable model of intramembrane cavitation. J. Neural Eng.