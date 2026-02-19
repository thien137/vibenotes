---
topic: algorithmic-ml
title: Decision Trees
summary: "Tree-based models: splitting, pruning, and ensembles."
image: images/ml-trees.svg
---

# Decision Trees

Decision trees partition the feature space with axis-aligned splits. Each internal node tests a feature; leaves predict the output.

## Splitting

Choose splits that maximize **information gain** (or minimize impurity). Common criteria:

- **Gini impurity**: $1 - \sum_{k} p_{k}^2$
- **Entropy**: $-\sum_{k} p_{k} \log p_{k}$

## Pruning

Reduce overfitting by pruning branches. Methods: reduced-error pruning, cost-complexity pruning.

## Ensembles

- **Random Forest**: Many trees on bootstrap samples + random feature subsets
- **Gradient Boosting**: Sequentially fit trees to residuals (e.g., XGBoost, LightGBM)
