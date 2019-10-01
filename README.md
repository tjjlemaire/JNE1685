# Description

This repository contains the Python scripts allowing to generate the figures of the **multi-Scale Optimized Neuronal Cavitation (SONIC) model** paper [1].

## Content of repository

- `figure_**.ipynb`: notebooks used to generate the paper figures that result from model simulations (i.e. all except the schematic figures 1 and 3).
- `get_data.py`: script used to run all model simulations required to generate the figures.
- `utils.py`: module containing utilities functions used in the notebooks and the `get_data.py` script.

# Requirements

- Python 3.6+
- NEURON 7.x (https://neuron.yale.edu/neuron/download/)
- PySONIC package (https://c4science.ch/diffusion/4670/)
- ExSONIC package (https://c4science.ch/diffusion/7145/)

# Installation

- Install a Python distribution
- Install a NEURON distribution
- Download the PySONIC AND ExSONIC packages from their repositories, and install them as packages using `pip install`.

Check out the PySONIC and ExSONIC repositories for more instructions on how to install the Python packages and their dependencies (including NEURON).

# Usage

## Getting/generating the data

Before generating the figures, one must obtain the required dataset resulting from cumbersome model simulations. 

In order to generate that data from scratch, you can use the `get_data.py` script, with the `mpi` option to enable multiprocessing: 

```
python get_data.py --mpi
```

Be aware that the **cumulated computation time required to run all simulations can easily exceed 1 week**, and that the **total size of entire dataset size is about 104 GB**. Therefore, it is highly advised that you run that script on a **high-performance, multi-core machine with enough disk space**.

If you don't want to generate the data from scratch, you can download it from the following link:

<span style="color:red">TODO: add link to data repository</span>

## Generating the figures

To generate a figure:

- start a *jupyter notebook*  / *jupyter lab* session:

`$ jupyter lab`

- open the figure notebook
- start running the notebook cell by cell (using Shift + Enter)
- if required, enter an input directory to inform about where the data is located (follow the indications)
- complete the notebook execution, using the command: *Run -> Run Selected Cell and All Below*

The figures panels should appear in the notebook. Additionally, they will be saved as PDFs in a *figs* sub-folder. 

# Authors

Code written and maintained by Theo Lemaire (theo.lemaire@epfl.ch).

# License

This project is licensed under the MIT License - see the LICENSE file for details.

# References

[1] Lemaire, T., Neufeld, E., Kuster, N., and Micera, S. (2019). Understanding ultrasound neuromodulation using a computationally efficient and interpretable model of intramembrane cavitation. J. Neural Eng.