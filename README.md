# Who is driving the conversation?

This is the repository with all the publicly available* data and analysis scripts for the research paper: ["Who is driving the conversation? Studying the nodality of British MPs and journalists on social media"](https://arxiv.org/abs/2402.08765).

*: Twitter data is made available in accordance with Twitter’s terms of service.

## Index
- 1. [Installation](#1-installation)
- 2. [Data](#2-data)
- 3. [Classifier](#3-classifier)
- 4. [Network generator](#4-network_generator)
- 4. [Analysis](#4-analysis)
- 6. [Figures and Tables](#6-figures_and_tables)

## 1. Installation

The package is written in Python (version: 3.8). We recommend that the installation is made inside a virtual environment and to do this, one can use `conda` (recommended in order to control the Python version).

### Using conda

The tool `conda`, which comes bundled with Anaconda has the advantage that it lets us specify the version of Python that we want to use. Python=3.8 is required.

After locating in the local github folder, like `cd $PATH$` e.g. `Documents/Local_Github/nodality`, a new environment can be created with

```bash
$ conda env create -f environment.yaml
```

The environment's name will be `nodality`. The environment must be activated before using it with

```bash
$ conda activate nodality
```

## 2. Data

Following Twitter's terms and conditions, we can only share Tweet IDs. Our database consists of the activity of UK journalists and MPs from January 14, 2022, to January 13, 2023. 

The labels used are:
- 1 for the Russia-Ukraine War
- 2 for the COVID-19 pandemic
- 5 for the Cost of Living Crisis
- 6 for Brexit
- -1 for any other topic

## 3. Classifier

We use a weak supervision classifier based on [(Ratner et al, 2019)](https://ojs.aaai.org/index.php/AAAI/article/view/4403). The classifier can be found in `classifier.py`, while the accompanying labeling functions can be reviewed in `labeling_functions.py`. A confusion matrix of the classifier can be reviews in [`confusion_matrix.csv`]


## 4. Network generator

Given the classification of tweets, we generate the interaction networks where nodes are Twitter users and links represent the interactions they have between themselves. 

## 5. Analysis

The analysis of the networks are done in the different folders `linear_model/` and `pca_kmeans/`.

## 6. Figures and tables

The folder `figures_tables/` contains all the notebooks to create the Figures and Tables of the pre-print.
