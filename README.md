# Deep protein representations enable recombinantprotein expression prediction
This repository contains the code developed for the paper "Deep protein representations enable recombinantprotein expression prediction".

## UniRep sequence representation
Most of the code is originally from the [UniRep repository](https://github.com/churchlab/UniRep/), but has been modified to suit this project. The script is [`src/unirep_formatter.py`](src/unirep_formatter.py).

## Training classifiers
Two scripts contains the code for training classifiers to predict protein expression:
* [`src/train.py`](src/train.py): Trains SVM, LR and RF classifiers
* [`src/train_ann.py`](src/train_ann.py): Train ANN classifiers

## Presenting results in tables and/or figures
The notebook [`Figures_and_tables.ipynb`](notebooks/Figures_and_tables.ipynb) show how to generate the figures and tables of the paper. Specifically Figure 2, Figure 3, Table 1 and supplementary material.