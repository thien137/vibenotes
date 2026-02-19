---
topic: algorithmic-ml
title: Support Vector Machines
summary: "Maximum margin classifiers and the kernel trick."
image: images/ml-svm.svg
---

# Support Vector Machines (SVMs)

SVMs find the **maximum margin** hyperplane that separates classes. Support vectors are the training points closest to the decision boundary.

## Linear SVM

For linearly separable data, the optimization is:

$$
\min_{w,b} \frac{1}{2}\|w\|^{2} \quad \text{s.t.} \quad y_{i}(w \cdot x_{i} + b) \geq 1
$$

The decision function is $f(x) = \text{sign}(w \cdot x + b)$.

## Soft Margin

For non-separable data, introduce slack variables $\xi_{i}$:

$$
\min_{w,b,\xi} \frac{1}{2}\|w\|^{2} + C \sum_{i} \xi_{i}
$$

$C$ controls the tradeoff between margin size and classification errors.

## Kernel Trick

Map data to a higher-dimensional space via $\phi(x)$ without computing $\phi$ explicitly. Use kernel $K(x,x') = \phi(x) \cdot \phi(x')$. Common kernels: RBF, polynomial.
