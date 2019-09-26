# Custom loss functions for binary classification problems with highly imbalanced dataset using Extreme Gradient Boosted Trees

This repository contains presentation and R scripts that was used to obtains experiment results presented during a <i>Custom loss functions for binary classification problems with highly imbalanced dataset using Extreme Gradient Boosted Trees</i> talk at WhyR 2019? conference in Warsaw.

# Installation

R scripts used in experiments require some packages to be installed. To prepare your environment use following command.

```r
install.packages(c("dplyr", "mlr", "mlrMBO", "PRROC", "rgenoud", "xgboost"))
```

# Experiment

The matter of experiment was comparing two performance measures F1 score and AUCPR (area under precision-recall curve) of 4 xgboost models with different optimization functions:

* cross-entropy (standard optimization function)
* focal loss
* weighted cross-entropy
* bilinear function

The best value for both measures was found by performing hyper parameter search using MBO (Model-based Bayesian Optimization) approach with 5 fold cross validation.

# Training data 
This experiment requires special kind of imbalanced training set so we provided our own one from real-world application. It is a dataset consisting of 118 features (X0-X117) and target binary class (Class) representing claims of one of Insurance Company with label denoting if given case is fraudulent or not. Values of features are effect of performing normalization and PCA transformation on original data. The dataset can be download from following link.

* [Training dataset](<link-to-drive>)

In order to repeat the experiment download the CSV file from the link and put it in <i>data</i> directory of this repository.

# Running the experiment


Run this experiment using following command.

```bash
Rscript all_xgb_param_tuning.R
```

This script will produce a ``tune_results.Rdata`` file that contains result of mlr ``tuneParams`` function.

# References

[References](Presentation/references.bib)