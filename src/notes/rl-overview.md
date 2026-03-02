---
topic: reinforcement-learning
title: Reinforcement Learning Overview
summary: "Agents, environments, POMDPs, MDPs, value-based and policy-based RL, exploration, and reward functions."
image: images/rl-overview.svg
---

# Reinforcement Learning Overview

Reinforcement learning (RL) is the study of how to compute **optimal policies** when the environment model is unknown. The agent learns by interacting with the world, receiving rewards, and adapting its behavior to maximize long-term return. This note provides a comprehensive mathematical foundation.

---

## Summary & Agenda

1. **Introduction & Core Concepts** — Agents, policies, episodes, returns, value functions.
2. **Universal Model & POMDPs** — Belief states, observations, world dynamics.
3. **Markov Decision Processes** — Fully observed MDPs, trajectories, goal conditioning.
4. **Contextual MDPs & Bandits** — Hidden context, multi-armed bandits.
5. **Belief State MDPs** — Bayesian updates, exploration-exploitation.
6. **Optimization as Decision Problems** — Best-arm identification, Bayesian optimization.
7. **RL Architectures** — Value-based, policy-based, model-based RL.
8. **Exploration & Rewards** — $\epsilon$-greedy, reward shaping, intrinsic motivation.

---

## 1. Introduction and Core Concepts

### The Agent-Environment Loop

<blockquote class="callout-definition">
<p><strong>Agent.</strong> The learner or decision-maker. At each step, the agent receives an internal state $z_t$ (or observation), selects an action $a_t$, and receives feedback from the environment.</p>
</blockquote>

The action is determined by a **policy** $\pi$:

$$
a_t = \pi(z_t)
$$

For stochastic policies, we write $a_t \sim \pi(\cdot \mid z_t)$.

<blockquote class="callout-definition">
<p><strong>Policy.</strong> A mapping from (belief) states to actions or action distributions. The policy defines the agent's behavior.</p>
</blockquote>

In **Partially Observable Markov Decision Processes (POMDPs)**, the agent does not see the true world state; it receives an **observation** $o_t$ that reveals partial information about the hidden state.

---

### The Maximum Expected Utility Principle

The agent aims to maximize **expected return**:

<blockquote class="callout-definition">
<p><strong>Value function.</strong> The expected return starting from state $s_0$ and following policy $\pi$:</p>
</blockquote>

$$
V_\pi(s_0) = \mathbb{E}_{p(s_0, s_1, a_1, \ldots \mid s_0, \pi)} \left[ \sum_{t=0}^T R(s_t, a_t) \,\Big|\, s_0 \right]
$$

<blockquote class="callout-definition">
<p><strong>Optimal policy.</strong></p>
</blockquote>

$$
\pi^* = \arg\max_\pi \mathbb{E}_{p_0(s_0)}[V_\pi(s_0)]
$$

---

### Episodic vs. Continual Tasks

| Type | Description |
|------|--------------|
| **Episodic** | Tasks that terminate in an absorbing state. A new episode starts from $s_0 \sim p_0$. Horizon $T$ is random but finite. |
| **Continual** | Tasks run indefinitely. Transition dynamics include terms like $\pi(a_1 \mid s_1)\,\rho_{\text{env}}(o_2 \mid a_1, o_1)$ and state updates $\delta(s_2 = U(s_1, a_1, o_2))$. |

In episodic settings, the absorbing state transitions to itself with reward zero.

---

### Return and Discounting

<blockquote class="callout-definition">
<p><strong>Discounted return.</strong> For discount factor $\gamma \in [0, 1]$:</p>
</blockquote>

$$
G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots + \gamma^{T-t-1} r_{T-1}
$$

**Recursive form:**

$$
G_t = r_t + \gamma G_{t+1}
$$

<blockquote class="callout-definition">
<p><strong>State value (discounted).</strong></p>
</blockquote>

$$
V_\pi(s_t) = \mathbb{E}[G_t \mid s_t, \pi]
$$

---

## 2. Universal Model and POMDPs

The **universal model** captures the causal relationships between the environment's hidden world state, the agent's internal belief state, and the resulting observations and actions. The diagram below shows one time step from $t$ to $t+1$.

<blockquote class="callout-tip">
<p><strong>Key idea.</strong> The agent's internal state $z_t$ is separated from the true world state $w_t$. They interact only via actions $a_t$ (agent → environment) and observations $o_{t+1}$ (environment → agent).</p>
</blockquote>

**1. The Agent's Loop:**

<blockquote class="callout-definition">
<p><strong>Policy ($\pi$).</strong> Determines the action from the agent's internal state:</p>
</blockquote>

$$
a_t = \pi(z_t)
$$

<blockquote class="callout-definition">
<p><strong>State update ($P$).</strong> Updates the internal state using the previous state, the action taken, and the new observation:</p>
</blockquote>

$$
z_{t+1} = P(z_t, a_t, o_{t+1})
$$

**2. The Environment (World) Loop:**

<blockquote class="callout-definition">
<p><strong>World model ($M$).</strong> Stochastic transition to the next world state, given current state, action, and noise $\xi^W$:</p>
</blockquote>

$$
w_{t+1} = M(w_t, a_t, \xi_{t+1}^W)
$$

<blockquote class="callout-definition">
<p><strong>Observation emission ($O$).</strong> Generates the observation the agent sees from the new world state and noise $\xi^o$:</p>
</blockquote>

$$
o_{t+1} = O(w_{t+1}, \xi_{t+1}^o)
$$

<blockquote class="callout-definition">
<p><strong>Reward ($R$).</strong></p>
</blockquote>

$$
r_t = R(w_t, a_t)
$$

<p><img src="/images/rl-universal-model.png" alt="Figure 1: Universal Model of Agent-Environment Interaction in a POMDP" style="max-width: 100%; height: auto;" /></p>

**Structural flow (ASCII):**

```
Time t                               Time t+1
======                               ========

(Environment)
  w_t  -----------------M------------>  w_{t+1}
   |                    ^                  |
   |                    |                  |
   v                    |                  v
  r_t = R(w_t, a_t)    a_t         o_{t+1} = O(w_{t+1}, ξ_{t+1}^o)
                        |                  |
(Agent)                 |                  |
                        ^                  v
  z_t  -------P---------|------------>  z_{t+1}
       \                |
        \-------π/------/
```

*Flow:* $a_t$ flows from the agent (policy $\pi$) to the environment (feeds $M$ and $R$); $o_{t+1}$ flows from the environment ($O$) to the agent (feeds $P$).

---

### POMDP Dynamics

In a POMDP, the true world state $w_t$ is **hidden**. The agent receives only the observation $o_t$.

**State transition** (with noise $\epsilon$):

$$
p(w_{t+1} \mid w_t, a_t) = \mathbb{E}_{\epsilon_{t+1}}\left[ \mathbb{I}(w_{t+1} = T(w_t, a_t, \epsilon_{t+1})) \right]
$$

**Observation emission:**

$$
p(o_{t+1} \mid w_{t+1}) = \mathbb{E}_{\epsilon_{t+1}^o}\left[ \mathbb{I}(o_{t+1} = O(w_{t+1}, \epsilon_{t+1}^o)) \right]
$$

**Internal state update** (agent):

$$
z_{t+1} = P(z_t, a_t, o_{t+1})
$$

**Immediate reward:** $r_t = R(w_t, a_t)$ (or $R(s_t, a_t)$ when state is observed).

The joint distribution is $p(w_{t+1}, o_{t+1} \mid w_t, a_t)$. Marginalizing over the hidden world state yields a **non-Markovian** observation distribution:

$$
p(o_{t+1} \mid o_t, \ldots, a_t) = \sum_{w_{t+1}} p(o_{t+1} \mid w_{t+1}) \cdots p(w_t \mid a_t, \ldots)
$$

---

## 3. Markov Decision Processes (MDPs)

An **MDP** is a POMDP where the state is **fully observed** ($o_t = s_t$).

<blockquote class="callout-definition">
<p><strong>MDP.</strong> Tuple $(S, A, p_S, p_R, p_0, \gamma)$. The agent observes $s_t$, takes $a_t$, receives $r_t$, and transitions to $s_{t+1}$.</p>
</blockquote>

**State transition:**

$$
p_S(s_{t+1} \mid s_t, a_t) = \mathbb{E}_{\epsilon_t}\left[ \mathbb{I}(s_{t+1} = f(s_t, a_t, \epsilon_t)) \right]
$$

**Reward:** $r_t \sim p_R(\cdot \mid s_t, a_t, s_{t+1})$. Expected reward for a transition:

$$
R(s_t, a_t, s_{t+1}) = \sum_r r \, p_R(r \mid s_t, a_t, s_{t+1})
$$

**Marginalized reward:**

$$
R(s_t, a_t) = \sum_{s_{t+1}} p_S(s_{t+1} \mid s_t, a_t) \, R(s_t, a_t, s_{t+1})
$$

---

### Sampling a Trajectory

With stochastic policy $\pi(a_t \mid s_t)$:

1. $(s_t, a_t) \sim \pi(\cdot \mid s_t)$
2. $s_{t+1} \sim p_S(\cdot \mid s_t, a_t)$
3. $r_t \sim p_R(\cdot \mid s_t, a_t, s_{t+1})$

A **trajectory** is $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots, s_T)$.

**Trajectory likelihood:**

$$
p(\tau) = p_0(s_0) \prod_{t=0}^{T-1} \pi(a_t \mid s_t) \, p_S(s_{t+1} \mid s_t, a_t) \, p_R(r_t \mid s_t, a_t, s_{t+1})
$$

When $S$ and $A$ are finite, the MDP can be represented as a **Finite State Machine** with transition tables.

---

### Goal-Conditioned MDPs

When the goal is to reach a target state $g$:

**Binary reward:** $R(s, a \mid g) = \mathbb{I}(s = g)$

**Continuous similarity:** $R(s, a \mid g) = \text{sim}(s, g) = \frac{\phi(s)^T \phi(g)}{\|\phi(s)\| \|\phi(g)\|}$ (e.g., cosine similarity)

Goal-conditioned policies $\pi(a \mid s, g)$ are trained to achieve specified goals.

---

## 4. Contextual MDPs and Bandits

### Contextual MDPs

**Contextual MDPs** have dynamics and rewards that depend on a hidden static **context** $c$ (e.g., procedurally generated game content). The agent must generalize across episodes with different contexts.

<blockquote class="callout-tip">
<p><strong>Example.</strong> Video games with procedural generation. Each run has a different $c$; the agent evaluates sequences of related episodes and needs strong generalization.</p>
</blockquote>

This is a special case of an epistemic POMDP with hidden model parameters.

---

### Contextual Bandits

In **Contextual Bandits**, the world state does **not** depend on the agent's actions:

$$
p(w_{t+1} \mid w_t, a_t) = p(w_t)
$$

The agent cannot affect the underlying state, but its action affects the reward: $R(a) = f(w, a)$.

<blockquote class="callout-definition">
<p><strong>Multi-armed bandit.</strong> Finite actions $A = \{a_1, \ldots, a_K\}$. Reward $R(a) = f(w, a)$ depends on context $w$ and action $a$.</p>
</blockquote>

**Applications:** Online advertising, clinical trials, recommendation systems.

---

## 5. Belief State MDPs and Bandits

### Belief State

When the environment is partially observed, the agent maintains a **belief state**—a probability distribution over hidden variables.

<blockquote class="callout-definition">
<p><strong>Belief state.</strong> $b_t = p(w \mid h_t)$ where $h_t = \{o_{1:t}, a_{1:t}, r_{1:t}\}$ is the history of observations, actions, and rewards.</p>
</blockquote>

**Update via Bayes rule:**

$$
b_{t+1} = \text{BayesRule}(b_t, o_{t+1}, a_t, r_{t+1})
$$

**Deterministic transition** (given inputs):

$$
p(b_{t+1} \mid b_t, o_{t+1}, a_t, r_t) = \mathbb{I}(b_{t+1} = B(b_t, o_{t+1}, a_t, r_t))
$$

Solving this continuous-state MDP addresses the **exploration-exploitation** tradeoff.

---

### Bandit Reward Models

| Bandit Type | Reward Model |
|-------------|--------------|
| **Bernoulli** | $p_R(r \mid a) = \text{Ber}(r \mid \mu_a)$, $\mu_a = R(a)$ |
| **Linear regression** | $p_R(r \mid s, a, w) = \mathcal{N}(r \mid \phi(s,a)^T w, \sigma^2)$ |
| **Logistic regression** | $p_R(r \mid s, a, w) = \text{Ber}(r \mid \sigma(\phi(s,a)^T w))$ |
| **Neural** | $p_R(r \mid s, a, w) = \mathcal{N}(r \mid f(s, a; w))$ |

With a well-modeled belief state, optimal policies can be approximated via **UCB** or **Thompson sampling**.

---

## 6. Optimization as Decision Problems

### Best-Arm Identification

**Goal:** Identify the single best arm within a fixed interaction budget $T$.

**Value:** $V_{\pi, T} = \mathbb{E}_{p(a, r \mid \pi)}[R(\hat{a})]$

**Selected arm:** $\hat{a} = \pi_T(a_1, r_1, \ldots, a_{T-1}, r_{T-1})$

---

### Bayesian Optimization

**Goal:** Optimize an expensive black-box function $R(w)$ with few queries.

$$
w^* = \arg\max_w R(w)
$$

**Active learning** queries points $w_a$ strategically to learn $R$.

**Stochastic Gradient Descent (SGD):** The agent queries the gradient $\hat{g}_t = \nabla_w R(w) \mid_{w_t}$ and updates:

$$
w_{t+1} = w_t + \alpha \hat{g}_t
$$

---

### RL Summary

<blockquote class="callout-definition">
<p><strong>Reinforcement learning.</strong> The study of how to compute optimal policies when the environment model is unknown.</p>
</blockquote>

**What the agent learns:** Value function, policy, dynamics model, or a combination.

**Representation:** Tabular or parametric (e.g., neural networks in **Deep RL**).

**Action selection:**
- **On-policy:** Actions from the current learning policy.
- **Off-policy:** Actions from a separate behavior policy; learn a target policy from different data.

---

## 7. High-Level RL Architectures

### Value-Based RL (Approximate Dynamic Programming)

Value-based methods learn the expected return of states or state-action pairs.

<blockquote class="callout-definition">
<p><strong>State-value function.</strong></p>
</blockquote>

$$
V_\pi(s) \triangleq \mathbb{E}_\pi[G_t \mid s_t = s] = \mathbb{E}_\pi\left[ \sum_{k=0}^\infty \gamma^k r_{t+k} \,\Big|\, s_t = s \right]
$$


<blockquote class="callout-definition">
<p><strong>Bellman optimality for $V$.</strong></p>
</blockquote>

$$
V^*(s) = \max_a \left( R(s, a) + \gamma \mathbb{E}_{p_S(s' \mid s, a)}[V^*(s')] \right)
$$


**Temporal Difference (TD) update:**

$$
V(s) \leftarrow V(s) + \eta \left[ r + \gamma V(s') - V(s) \right]
$$

The term $\delta = r + \gamma V(s') - V(s)$ is the **TD error** (temporal difference).

**Pseudocode (tabular TD):**

```python
def TD_update(V, s, r, s_prime, eta=0.1, gamma=0.99):
    delta = r + gamma * V[s_prime] - V[s]
    V[s] = V[s] + eta * delta
```

<blockquote class="callout-definition">
<p><strong>Action-value function (Q-function).</strong></p>
</blockquote>

$$
Q_\pi(s, a) \triangleq \mathbb{E}_\pi[G_t \mid s_t = s, a_t = a] = \mathbb{E}_\pi\left[ \sum_{k=0}^\infty \gamma^k r_{t+k} \,\Big|\, s_t = s, a_t = a \right]
$$


<blockquote class="callout-definition">
<p><strong>Bellman optimality for $Q$.</strong></p>
</blockquote>

$$
Q^*(s, a) = R(s, a) + \gamma \mathbb{E}_{p(s' \mid s, a)}\left[ \max_{a'} Q^*(s', a') \right]
$$


**Q-learning update (off-policy):**

$$
Q(s, a) \leftarrow Q(s, a) + \eta \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]
$$

**Pseudocode (tabular Q-learning):**

```python
def Q_learning_step(Q, s, a, r, s_prime, eta=0.1, gamma=0.99):
    td_target = r + gamma * max(Q[s_prime, a_] for a_ in actions)
    Q[s, a] = Q[s, a] + eta * (td_target - Q[s, a])
```

---

### Policy-Based RL

Policy-based methods optimize the policy parameters $\theta$ directly.

**Objective:**

$$
J(\pi_\theta) = \mathbb{E}_{p(s_0)}[V_{\pi_\theta}(s_0)]
$$

Uses **policy search** and **policy gradient** methods. **Actor-critic** methods combine a value function (critic) with direct policy updates (actor).

---

### Model-Based RL

Model-based RL learns the transition $p_S(s' \mid s, a)$ and reward $R(s, a)$, then plans the policy. Can be highly sample-efficient.

**Partial observability:** When the state is hidden, the agent uses:
- **History** $h_t = (a_1, o_1, \ldots, a_{t-1}, o_t)$
- **Belief state** $b_t = p(w_t \mid h_t)$
- **Flattened observation history** $s_t = h_{t-k:t}$
- **Recurrent policies** (RNNs) to summarize $h_t$

---

## 8. Exploration vs. Exploitation

<blockquote class="callout-tip">
<p><strong>Tradeoff.</strong> Exploit the current best policy or explore to discover higher-reward actions?</p>
</blockquote>

| Strategy | Description |
|----------|-------------|
| **$\epsilon$-greedy** | With probability $\epsilon$, take a random action; otherwise act greedily. |
| **Boltzmann (softmax)** | $\pi(a \mid s_t) = \frac{\exp(\hat{Q}(s_t, a) / \tau)}{\sum_{a'} \exp(\hat{Q}(s_t, a') / \tau)}$, $\tau > 0$ |
| **Intrinsic motivation** | Add exploration bonus $R^i_t(s, a)$ to encourage novel states |

---

## 9. Reward Functions

<blockquote class="callout-definition">
<p><strong>Reward Hypothesis.</strong> Goals can be fully captured by maximizing the expected sum of a scalar reward signal.</p>
</blockquote>

| Concept | Description |
|---------|-------------|
| **Non-Markovian rewards** | Goals may depend on state sequences, not just $(s, a)$. |
| **Reward hacking** | Agent maximizes the specified reward without achieving the intended goal (specification gaming). |
| **Sparse rewards** | $R(s, a) = 0$ almost everywhere; requires deep exploration. |

---

## 10. Reward Shaping

**Basic shaping:** $r' = r + F(s, s')$. Can change the optimal policy.

<blockquote class="callout-definition">
<p><strong>Potential-based reward shaping.</strong> Keeps the optimal policy invariant. For potential $\Phi(s)$:</p>
</blockquote>

$$
F(s, a, s') = \gamma \Phi(s') - \Phi(s)
$$

**Intrinsic rewards** (with action dependency):

$$
F(s, a, s', a') = \gamma \Phi(s', a') - \Phi(s, a), \quad r'(s, a) = r(s, a) + \eta F(s, a, s', a')
$$

---

## Summary Table

| Concept | Formula |
|---------|---------|
| Value function | $V_\pi(s) = \mathbb{E}[G_t \mid s_t = s, \pi]$ |
| Discounted return | $G_t = r_t + \gamma G_{t+1}$ |
| Bellman optimality (V) | $V^*(s) = \max_a \left( R(s,a) + \gamma \mathbb{E}[V^*(s')] \right)$ |
| Bellman optimality (Q) | $Q^*(s,a) = R(s,a) + \gamma \mathbb{E}[\max_{a'} Q^*(s',a')]$ |
| TD update | $V(s) \leftarrow V(s) + \eta[r + \gamma V(s') - V(s)]$ |
| Q-learning | $Q(s,a) \leftarrow Q(s,a) + \eta[r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$ |
| Potential shaping | $F(s,s') = \gamma \Phi(s') - \Phi(s)$ |

---

<blockquote class="callout-recall">
<p><strong>Related notes.</strong> See <a href="/notes/rl-policy-gradient/">Policy Gradient</a> for policy-based methods and <a href="/notes/rl-q-learning/">Q-Learning</a> for value-based methods.</p>
</blockquote>
