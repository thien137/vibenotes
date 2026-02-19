---
topic: reinforcement-learning
title: Reinforcement Learning Overview
summary: "Introduction to RL: agents, environments, rewards, and policies."
image: images/rl-overview.svg
---

# Overview

Reinforcement learning (RL) is a type of machine learning where an **agent** learns to make decisions by interacting with an **environment**. The agent receives **rewards** (or penalties) based on its actions and aims to maximize cumulative reward over time.

## Key Components

- **Agent**: The learner or decision-maker
- **Environment**: Everything the agent interacts with
- **State** $s$: A representation of the current situation
- **Action** $a$: What the agent can do
- **Reward** $r$: Feedback from the environment

## The Reward Hypothesis

The goal can be stated as maximizing expected cumulative reward:

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}
$$

where $\gamma \in [0, 1]$ is the discount factor.

## Policy

A **policy** $\pi(a|s)$ is a mapping from states to probability distributions over actions. The agent uses its policy to select actions.
