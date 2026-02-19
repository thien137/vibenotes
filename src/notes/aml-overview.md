---
topic: algorithmic-ml
title: Algorithmic ML Overview
summary: "Core concepts: supervised vs unsupervised, bias-variance tradeoff."
image: images/ml-overview.svg
---

# Algorithmic Machine Learning Overview

Classical machine learning focuses on algorithms that learn patterns from data without being explicitly programmed.

## Supervised vs Unsupervised

- **Supervised**: Learn from labeled examples $(x, y)$. Tasks: classification, regression.
- **Unsupervised**: Learn from unlabeled data $x$. Tasks: clustering, dimensionality reduction, density estimation.

## Bias-Variance Tradeoff

Total error decomposes as:

$$
\text{Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}
$$

- **High bias**: Underfitting—model too simple
- **High variance**: Overfitting—model too complex, memorizes noise

## Cross-Validation

Use $k$-fold cross-validation to estimate generalization performance and tune hyperparameters.
