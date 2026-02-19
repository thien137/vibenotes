---
topic: reinforcement-learning
title: Policy Gradient
summary: "Directly optimizing the policy using gradient ascent on expected return."
image: images/rl-policy.svg
---

# Policy Gradient Methods

Instead of learning value functions and deriving a policy, **policy gradient** methods directly parameterize and optimize the policy $\pi_\theta(a|s)$.

## Objective

Maximize expected return:

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[ G(\tau) \right]
$$

## Policy Gradient Theorem

The gradient of the objective is:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[ \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \, G_t \right]
$$

## REINFORCE

A simple Monte Carlo policy gradient algorithm:

1. Sample trajectory $\tau$ from $\pi_\theta$
2. For each step $t$, update: $\theta \leftarrow \theta + \alpha G_t \nabla_\theta \log \pi_\theta(a_t|s_t)$

## Variance Reduction

High variance in $G_t$ leads to slow learning. Common tricks: baseline subtraction, advantage actor-critic (A2C).
