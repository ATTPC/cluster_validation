# cluster_validation

Methods for validating [Spyral](https://github.com/ATTPC/Spyral) clustering algorithms 
with [attpc_engine](https://github.com/ATTPC/attpc_engine) data. We implement a custom
Spyral phase for analyzing the accuracy of clustering.

## Installation

Download the repository as

```bash
git clone https://github.com/ATTPC/cluster_validation.git
```

Enter the repo and create a python environment and activate it

```bash
cd cluster_validation
python -m venv .venv
source .venv/bin/activate
```

Note you may need to  use a specific Python (i.e. replace `python` with `python3.11`)
and the activation step may change depending on your platform (specifically for Windows).
Now install the dependencies from the `requirements.txt` file

```bash
pip install -r requirements.txt
```

## Usage

Before using this repo, you will need to generate some [attpc_engine](https://github.com/ATTPC/attpc_engine)
data. See the [engine docs](https://attpc.github.io/attpc_engine) for details.

This repo comes with a couple of methods for usage. The first is a notebook for
visualizing the analysis and understanding the techniques used. To run the notebook,
run the command

```bash
jupyter-lab
```

and select the notebook `cluster_validation.ipynb`. You'll have to set some paths and
choose clustering parameters, which you can explore with the visualizations.

However, the notebook is not a good way to explore a large set of events. To run an
entire dataset through the analysis, we use the [Spyral](https://github.com/ATTPC/Spyral)
framework. Included in the repo is the file `cluster_validation.py`. This runs a 
customized Spyral phase that clusters and validates the data. Note that you will need to
edit `cluster_validation.py` to set paths and parameters. See the 
[Spyral docs](https://attpc.github.io/Spyral) for details on the various parameters. To 
run the analysis simply use

```bash
python cluster_validation.py
```

Once you've run the validation pipeline, you will have a Spyral workspace that has a
directory called `ClusterValidation`. Inside this directory will be a standard Spyral
cluster datafile (`run_#.h5`) and a parquet file (`run_#.parquet`) that contains the
validation data. This repo contains a simple tool which provides some aggregate 
validation results `evaluate_cluster_accuracy.py`. To run the aggregator use

```bash
python evaluate_cluster_accuracy.py <path/to/parquet.parquet> <truth_label>
```

This will evaluate clustering statistics for the specific truth label you specify; that 
is, it will evaluate the accuracy of clustering a specific nucleus.


## Modifying clustering

Generally, you will want to use this repo to evaluate the performance of different cluster
methods. You can modify the validation phase by editing the code in the `validation` 
directory. Normal rules for [custom Spyral phases](https://attpc.github.io/Spyral/user_guide/extending)
apply here.

## Requirements

Requires Python >= 3.10, < 3.13
