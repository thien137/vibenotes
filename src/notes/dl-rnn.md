---
topic: deep-learning
title: Recurrent Networks
summary: "RNNs and LSTMs for sequential data."
image: images/dl-rnn.svg
---

# Recurrent Neural Networks (RNNs)

RNNs process **sequences** by maintaining a hidden state that carries information across time steps.

## Recurrence

At each step $t$:

$$
h_{t} = \sigma(W_{hh} h_{t-1} + W_{xh} x_{t} + b)
$$

The hidden state $h_{t}$ encodes the history of the sequence.

## Vanishing Gradient

Standard RNNs suffer from vanishing/exploding gradients when sequences are long. Training becomes difficult.

## Long Short-Term Memory (LSTM)

LSTMs use **gates** (forget, input, output) to control information flow:

- Forget gate: what to discard from cell state
- Input gate: what new info to store
- Output gate: what to output

This helps preserve gradients over long sequences.
