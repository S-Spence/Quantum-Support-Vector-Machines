# Quantum-Support-Vector-Machines

The repository stores a comparison of two hybrid quantum support vector machines to classical support vector machines. The quantum models tested were the fidelity quantum kernel and the projected quantum kernel. The jupyter notebooks and evaluations directory show the experiment results. The classical support vector machines outperformed the quantum models for all experiments. See the paper for more details.

## Environment Setup

Using venv:

```
python3 -m venv .venv
source .venv/bin/activate # or .\venv\Scripts\activate in powershell
pip install -r environment.txt
```

Using anaconda:

```
conda env create -f environment.yaml
conda activate qsvm
```

## Quickstart

Run the jupyter notebooks to test the existing models or create a new notebook to tune model parameters or test different kernel configurations. 
