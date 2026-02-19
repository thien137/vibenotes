---
topic: reinforcement-learning
title: Q-Learning
summary: "Value-based method for learning optimal policies through Q-value updates."
image: images/rl-q.svg
---

# Q-Learning

Q-Learning is a **model-free**, **off-policy** value-based RL algorithm. It learns the optimal action-value function $Q^{\ast}(s,a)$ directly. It was introduced by Chris Watkins in 1989 and remains one of the most influential algorithms in reinforcement learning.

## Q-Value

<blockquote class="callout-definition">
<p><strong>Definition.</strong> The Q-value $Q(s,a)$ represents the expected cumulative reward of taking action $a$ in state $s$ and then following the optimal policy.</p>
</blockquote>

Equivalently:

$$
Q(s,a) = \mathbb{E}\left[ G_t \mid S_t = s, A_t = a \right]
$$

where $G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots$ is the return. The Q-value tells us how good it is to take action $a$ in state $s$ if we behave optimally afterwards.

### Intuition

Think of the Q-value as a lookup table: for each state-action pair, we store the expected total reward. The optimal policy is then to pick the action with the highest Q-value in each state: $\pi^{\ast}(s) = \arg\max_{a} Q(s,a)$.

### Key Properties

- Q-values satisfy the **Bellman optimality equation**: $Q^{\ast}(s,a) = \mathbb{E}[R + \gamma \max_{a'} Q^{\ast}(S', a')]$
- They can be learned from experience without knowing the environment dynamics
- The optimal Q-function is unique for finite MDPs

## Update Rule

The Q-learning update is:

$$
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma \max_{a} Q(S_{t+1}, a) - Q(S_t, A_t) \right]
$$

where $\alpha$ is the learning rate and $\gamma$ is the discount factor. Pseudocode:

```python
def q_learning_update(Q, s, a, r, s_next, alpha, gamma):
    Q[s, a] += alpha * (r + gamma * Q[s_next].max() - Q[s, a])
```

### Interpretation

The term in brackets is the **TD error**: the difference between the target $R_{t+1} + \gamma \max_{a} Q(S_{t+1}, a)$ and the current estimate $Q(S_t, A_t)$. We move our estimate a step ($\alpha$) toward the target.

### Convergence

Under standard conditions (all state-action pairs visited infinitely often, learning rate satisfies Robbins-Monro), Q-learning converges to $Q^{\ast}$ with probability 1.

## Properties

- **Off-policy**: Learns about the greedy policy while following an exploratory policy (e.g., $\varepsilon$-greedy)
- **Temporal difference**: Updates estimates based on other estimates (bootstrap)
- **Model-free**: Does not require knowledge of transition probabilities $P(s'|s,a)$ or reward function $R(s,a)$

## Algorithm

Here is a full episodic Q-learning algorithm:

```python
def q_learning(env, num_episodes, alpha, gamma, epsilon):
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    for ep in range(num_episodes):
        s = env.reset()
        done = False
        while not done:
            a = epsilon_greedy(Q[s], epsilon)
            s_next, r, done, _ = env.step(a)
            Q[s][a] += alpha * (r + gamma * Q[s_next].max() - Q[s][a])
            s = s_next
    return Q
```

### Hyperparameters

| Parameter | Role |
|-----------|------|
| $\alpha$ | Learning rate; controls update step size |
| $\gamma$ | Discount factor; how much we value future rewards |
| $\varepsilon$ | Exploration rate; probability of random action |

## Variants

### Double Q-Learning

Standard Q-learning overestimates action values due to the max operator. Double Q-learning maintains two Q-functions and uses one to select the action and the other to evaluate it, reducing overestimation bias.

### Expected SARSA

Instead of $\max_{a} Q(S', a)$, uses $\mathbb{E}_{\pi}[Q(S', A')]$, which can reduce variance at the cost of being on-policy if $\pi$ is the behavior policy.

## Practical Tips

<blockquote class="callout-tip">
<p><strong>Tip.</strong> Start with $\varepsilon = 0.1$ and decay it over time. Use a replay buffer (DQN-style) for stability when combining with neural networks.</p>
</blockquote>

- Replay experience to break correlation between consecutive samples
- Use target networks to stabilize training
- Prioritize rare or high-TD-error transitions for faster learning
