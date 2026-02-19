---
topic: deep-learning
title: Convolutional Networks
summary: "CNNs for images: convolutions, pooling, and architectures."
image: images/dl-cnn.svg
---

# Convolutional Neural Networks (CNNs)

CNNs are designed for grid-like data (images) using **convolutional** and **pooling** layers.

## Convolution

Apply learnable filters (kernels) that slide across the input:

$$
(y)_{i,j} = \sum_m \sum_n (x)_{i+m, j+n} \cdot (k)_{m,n}
$$

Captures local patterns (edges, textures) and shares weights across spatial positions.

## Pooling

Downsample feature maps (e.g., max pooling) to reduce dimensionality and add translation invariance.

## Common Architectures

- **LeNet**, **AlexNet**: Early CNNs
- **VGG**: Deeper stacks of 3×3 convs
- **ResNet**: Residual connections for very deep networks
