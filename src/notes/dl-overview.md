---
topic: deep-learning
title: Deep Learning Overview
summary: "From perceptrons to deep nets. Backpropagation and optimization."
image: images/dl-overview.svg
---

# Deep Learning Overview

Deep learning uses **neural networks** with many layers to learn hierarchical representations from data.

## Perceptron

A single neuron: $y = \sigma(w \cdot x + b)$ where $\sigma$ is an activation function (e.g., sigmoid, ReLU).

## Multi-Layer Networks

Stack layers: $h^{(l+1)} = \sigma(W^{(l)} h^{(l)} + b^{(l)})$

## Backpropagation

Efficiently compute gradients via the chain rule. For loss $L$:

$$
\frac{\partial L}{\partial w_{ij}^{(l)}} = \frac{\partial L}{\partial a_j^{(l)}} \frac{\partial a_j^{(l)}}{\partial w_{ij}^{(l)}}
$$

## Optimization

Stochastic gradient descent (SGD) and variants: Adam, RMSprop. Regularization: dropout, weight decay.
