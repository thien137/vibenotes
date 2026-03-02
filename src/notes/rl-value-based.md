---
topic: reinforcement-learning
title: Value-Based Reinforcement Learning
summary: "V-functions, Q-functions, Bellman operators, Value/Policy Iteration, TD, SARSA, Q-Learning, VFA, DQN, Double DQN, Dueling DQN."
image: images/rl-q.svg
---

# Value-Based Reinforcement Learning: From Tabular to Deep Approximation

Value-based RL (also called **Approximate Dynamic Programming** or **ADP**) centers on learning the expected return of states or state-action pairs to derive an optimal policy. Rather than directly optimizing the policy parameters, the agent fits models of value functions and acts greedily with respect to those predictions. This note traces the path from tabular planning to deep Q-networks, including the theoretical foundations and practical mitigations for instability.

---

## Motivation: Moving Beyond Policy Gradients

In actor-critic algorithms, we run policy $\pi_\theta(a|s)$, fit a value model $\hat{V}^\pi_\phi(s)$ to estimate return, compute the advantage $\hat{A}^\pi(s_i, a_i) = r(s_i, a_i) + \hat{V}^\pi_\phi(s'_i) - \hat{V}^\pi_\phi(s_i)$, and improve the policy via the policy gradient.

We can **omit the policy gradient entirely**. The advantage $A^\pi(s_t, a_t)$ measures how much better action $a_t$ is than the average under $\pi$. Therefore, $\arg\max_{a} A^\pi(s_t, a)$ is the best action from $s_t$ (assuming we follow $\pi$ thereafter). We can define a strictly better policy by acting greedily with respect to the advantage:

$$
\pi'(a_t | s_t) = \begin{cases} 1 & \text{if } a_t = \arg\max_{a} A^\pi(s_t, a) \\ 0 & \text{otherwise} \end{cases}
$$

This deterministic policy is guaranteed to be at least as good as any action sampled from $\pi(a_t|s_t)$. This shifts the paradigm from **optimizing policies** to **fitting models** of $A^\pi$, $Q^\pi$, or $V^\pi$.

---

## Summary & Agenda

1. **Foundations** — V-function, Q-function, advantage function; Bellman optimality; contraction mapping.
2. **Planning (Known Model)** — Value Iteration, Policy Iteration.
3. **Model-Free Tabular** — Monte Carlo, TD, TD($\lambda$), eligibility traces.
4. **Model-Free Control** — SARSA, Q-Learning; exploration ($\epsilon$-greedy, Boltzmann).
5. **Value Function Approximation** — Linear VFA, Oracle vs. MC/TD targets.
6. **Batch RL** — Least Squares MC, Least Squares TD, LSDQ.
7. **Fitted VI / FQI / Deadly Triad** — FVI (fatal flaw), FQI, online Q-learning, semi-gradient, projection $\Pi$, convergence failure.
8. **DQN** — Experience replay, target networks.
9. **Advanced** — Double DQN, Dueling DQN, PER, continuous actions (NAF, DDPG).

---

## 1. Foundations of Value-Based RL

<blockquote class="callout-definition">
<p><strong>State-Value Function ($V$-function).</strong> The expected return starting from state $s$ and following policy $\pi$.</p>
</blockquote>

$$
V_\pi(s) \triangleq \mathbb{E}_\pi[G_0 \mid s_0 = s] = \mathbb{E}_\pi\left[ \sum_{t=0}^\infty \gamma^t r_t \,\Big|\, s_0 = s \right]
$$

<blockquote class="callout-definition">
<p><strong>Action-Value Function ($Q$-function).</strong> The expected return starting with action $a$ in state $s$, then following policy $\pi$.</p>
</blockquote>

$$
Q_\pi(s, a) \triangleq \mathbb{E}_\pi[G_0 \mid s_0 = s, a_0 = a] = \mathbb{E}_\pi\left[ \sum_{t=0}^\infty \gamma^t r_t \,\Big|\, s_0 = s, a_0 = a \right]
$$

<blockquote class="callout-definition">
<p><strong>Advantage Function.</strong> The relative benefit of taking action $a$ in state $s$ compared to the baseline of strictly following $\pi$.</p>
</blockquote>

$$
A_\pi(s, a) \triangleq \mathit{Adv}_\pi(s, a) = Q_\pi(s, a) - V_\pi(s)
$$

**Equivalence:** $V_\pi(s) = \mathbb{E}_{a \sim \pi(\cdot|s)}[Q_\pi(s, a)]$.

---

### Bellman Optimality and Contraction

A policy is **optimal** if $V_{\pi^*}(s) \ge V_\pi(s)$ for all $s$ and $\pi$. Any finite MDP has at least one deterministic optimal policy. The optimal value functions satisfy the **Bellman Optimality Equations**:

$$
V^*(s) = \max_a \left[ R(s, a) + \gamma \mathbb{E}_{p(s'|s,a)}[V^*(s')] \right]
$$

$$
Q^*(s, a) = R(s, a) + \gamma \mathbb{E}_{p(s'|s,a)}\left[ \max_{a'} Q^*(s', a') \right]
$$

<blockquote class="callout-definition">
<p><strong>Bellman Backup Operator $\mathcal{B}$.</strong> For value function $V$:</p>
</blockquote>

$$
(\mathcal{B}V)(s) = \max_a \mathbb{E}_{T(s'|s,a)}\left[ R(s, a) + \gamma V(s') \right]
$$

In vector form, where $\mathcal{T}_a$ is the transition matrix for action $a$ and $r_a$ the reward vector: $\mathcal{B}V = \max_a (r_a + \gamma \mathcal{T}_a V)$. The optimal value $V^*$ is the unique fixed point: $V^* = \mathcal{B}V^*$.

<blockquote class="callout-lemma">
<p><strong>Contraction Mapping.</strong> $\mathcal{B}$ is a $\gamma$-contraction in the $L_\infty$-norm. For any two value functions $V$ and $\bar{V}$:</p>
</blockquote>

$$
\|\mathcal{B}V - \mathcal{B}\bar{V}\|_\infty \le \gamma \|V - \bar{V}\|_\infty
$$

Tabular Value Iteration converges to $V^*$ because each backup strictly shrinks the maximum distance to the optimum.

**Proof:**

$$
\|\mathcal{B}V - \mathcal{B}U\|_\infty = \max_s \left| \max_a (R(s,a) + \gamma \mathbb{E}[V(s')]) - \max_a (R(s,a) + \gamma \mathbb{E}[U(s')]) \right|
$$

$$
\le \max_{s,a} \left| \gamma \mathbb{E}[V(s')] - \gamma \mathbb{E}[U(s')] \right| \le \gamma \max_{s'} |V(s') - U(s')| = \gamma \|V - U\|_\infty
$$

Since $\gamma < 1$, repeated application of $\mathcal{B}$ converges to a unique fixed point $V^*$. The **Bellman error** is $V^*(s) - V(s)$ or $Q^*(s, a) - Q(s, a)$.

---

## 2. Solving MDPs with Known Models (Planning)

When the transition dynamics $p(s'|s,a)$ and reward function $R(s,a)$ are explicitly known, we face a prediction or control problem that can be solved via Dynamic Programming (DP).

### Value Iteration (VI)

By skipping the explicit policy representation, we compute values directly. Equivalent formulation:

1. Set $Q(s,a) \leftarrow r(s,a) + \gamma \mathbb{E}[V(s')]$
2. Set $V(s) \leftarrow \max_a Q(s,a)$

Or recursively:

$$
V_{k+1}(s) = \max_a \left[ R(s, a) + \gamma \sum_{s'} p(s'|s, a) V_k(s') \right]
$$

By contraction, $\max_s |V_{k+1}(s) - V^*(s)| \le \gamma \max_s |V_k(s) - V^*(s)|$, so iteration converges to $V^*$. **Real-Time Dynamic Programming (RTDP)** focuses computation only on reachable states from a start state rather than the full state space.

**Pseudocode:**

```python
def value_iteration(MDP, theta=1e-6):
    V = {s: 0 for s in MDP.states}
    while True:
        delta = 0
        for s in MDP.states:
            v_old = V[s]
            V[s] = max(sum(p * (r + gamma * V[s_]) for (p, s_, r) in MDP.transitions(s, a))
                       for a in MDP.actions(s))
            delta = max(delta, abs(V[s] - v_old))
        if delta < theta:
            break
    return V
```

---

### Policy Iteration (PI)

**Policy Iteration** alternates between two steps:

1. **Evaluate:** Compute $A^\pi(s,a)$ for the current policy.
2. **Improve:** Set $\pi \leftarrow \pi'$ using the greedy argmax formulation.

Because $A^\pi(s,a) = r(s,a) + \gamma \mathbb{E}[V^\pi(s')] - V^\pi(s)$, evaluation fundamentally requires $V^\pi(s)$.

**1. Policy Evaluation (tabular):** For policy $\pi$, with known dynamics $p(s'|s,a)$:

$$
V^\pi(s) \leftarrow \mathbb{E}_{a \sim \pi(a|s)}\left[ r(s,a) + \gamma \mathbb{E}_{s' \sim p(s'|s,a)}[V^\pi(s')] \right]
$$

For a deterministic policy $\pi(s) = a$, this simplifies to $V^\pi(s) \leftarrow r(s, \pi(s)) + \gamma \mathbb{E}_{s'}[V^\pi(s')]$. In vector form: $\vec{v} = \vec{r} + \gamma T \vec{v}$ implies $\vec{v} = (I - \gamma T)^{-1} \vec{r}$.

**2. Policy Improvement:** Greedy with respect to $V_\pi$:

$$
\pi'(s) = \arg\max_a \left\{ R(s, a) + \gamma \mathbb{E}[V_\pi(s')] \right\}
$$

<blockquote class="callout-lemma">
<p><strong>Policy Improvement Theorem.</strong> The new policy $\pi'$ satisfies $V_{\pi'} \ge V_\pi$ (strictly unless $\pi$ is optimal).</p>
</blockquote>

**Proof:** Because $\pi'$ is greedy, $\vec{r}' + \gamma T' \vec{v}_\pi \ge \vec{r} + \gamma T \vec{v}_\pi = \vec{v}_\pi$. Then:

$$
\vec{v}_\pi \le \vec{r}' + \gamma T' \vec{v}_\pi \le \vec{r}' + \gamma T' (\vec{r}' + \gamma T' \vec{v}_\pi) \le \cdots = (I - \gamma T')^{-1} \vec{r}' = \vec{v}_{\pi'}
$$

---

## 3. Model-Free Tabular Learning

When transition dynamics are unknown, the agent cannot use Dynamic Programming. Instead, it learns from sampled experience — trajectories collected by interacting with the environment.

<blockquote class="callout-definition">
<p><strong>Monte Carlo (MC).</strong> Updates $V(s_t)$ using the full return $G_t$ from rollouts.</p>
</blockquote>

$$
V(s_t) \leftarrow V(s_t) + \eta [G_t - V(s_t)]
$$

*Limitation:* Requires episodic tasks; high variance from long trajectories.

<blockquote class="callout-definition">
<p><strong>Temporal Difference (TD).</strong> Bootstraps off the next-state value.</p>
</blockquote>

$$
V(s_t) \leftarrow V(s_t) + \eta [r_t + \gamma V(s_{t+1}) - V(s_t)]
$$

The term $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ is the **TD error**.

---

### N-Step Returns and TD($\lambda$)

**N-step return:**

$$
G_{t:t+n} = r_t + \gamma r_{t+1} + \cdots + \gamma^{n-1} r_{t+n-1} + \gamma^n V(s_{t+n})
$$

**$\lambda$-return:**

$$
G_t^\lambda \triangleq (1-\lambda) \sum_{n=1}^\infty \lambda^{n-1} G_{t:t+n}
$$

**Eligibility traces** (backward-view): For function approximation with parameters $w$, the trace accumulates gradients of the value function:

$$
z_t = \gamma \lambda z_{t-1} + \nabla_w V_w(s_t), \qquad w_{t+1} = w_t + \eta \delta_t z_t
$$

In the tabular case, $z_t(s)$ accumulates state visit counts: $z_t(s) = \gamma \lambda z_{t-1}(s) + \mathbb{I}(s = s_t)$.

---

## 4. Model-Free Control: SARSA and Q-Learning

To derive a policy without a model, we must learn the action-value function $Q(s,a)$ instead of $V(s)$ — the policy is then $\pi(s) = \arg\max_a Q(s,a)$. Two main approaches differ in which policy's value they estimate:

<blockquote class="callout-definition">
<p><strong>SARSA (On-Policy TD Control).</strong> Learns $Q^\pi$ for the policy $\pi$ the agent is currently following. Uses the actual next action $a'$ in the target.</p>
</blockquote>

$$
Q(s, a) \leftarrow Q(s, a) + \eta [r + \gamma Q(s', a') - Q(s, a)]
$$

To guarantee convergence to $Q^*$, SARSA must use a GLIE (Greedy in the Limit with Infinite Exploration) policy. **SARSA($\lambda$)** extends this with eligibility traces (see Section 3).

**Exploration:** Always selecting $a_t = \arg\max_a Q(s_t, a)$ leads to poor exploration. Common strategies:

| Strategy | Description |
|----------|-------------|
| **$\epsilon$-greedy** | Greedy with probability $1-\epsilon$, random action with probability $\epsilon$ |
| **Boltzmann exploration** | $\pi(a_t \mid s_t) \propto \exp(Q(s_t, a_t) / \tau)$, $\tau > 0$ |

<blockquote class="callout-definition">
<p><strong>Q-Learning (Off-Policy TD Control).</strong> Learns $Q^*$ while following any exploratory behavior policy. The target uses the greedy action $\max_{a'} Q(s', a')$, so it estimates the optimal policy's value.</p>
</blockquote>

$$
Q(s, a) \leftarrow Q(s, a) + \eta [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]
$$

**Pseudocode (tabular Q-Learning):**

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

---

## 5. Value Function Approximation (VFA)

Tabular methods fail when the state space is large or continuous: we cannot store a separate value for each state. **Function approximation** scales RL by generalizing from seen states to unseen ones.

### Motivation

VFA enables scaling beyond trivial tabular settings:

- **No explicit storage:** Avoids learning or storing dynamics, reward models, value functions, or policies for every individual state.
- **Generalization:** By approximating $\hat{v}(s,w) \approx V^\pi(s)$ and $\hat{q}(s,a,w) \approx Q^\pi(s,a)$, the agent generalizes from seen states to unseen states.
- **Continuous updates:** Approximations are updated via Monte Carlo (MC) or Temporal Difference (TD) learning.

**Function approximators in practice:** Linear combinations of features; neural networks (e.g., $\hat{\pi}(a,s,w) \approx \pi(a|s)$); decision trees; nearest neighbors.

---

### Oracle and Linear VFA

If we had an "oracle" providing true values $V^\pi(s)$, we would minimize the expected squared error:

$$
J(w) = \mathbb{E}_\pi\left[ (V^\pi(s) - \hat{v}(s, w))^2 \right]
$$

**Gradient descent** steps in the direction of the negative gradient:

$$
\Delta w = -\frac{1}{2} \alpha \nabla_w J(w), \qquad w_{t+1} = w_t + \Delta w
$$

In **Linear VFA**, the state is a feature vector $x(s) = [x_1(s), \ldots, x_n(s)]^T$ and:

$$
\hat{v}(s, w) = x(s)^T w
$$

The objective becomes $J(w) = \mathbb{E}_\pi [ (V^\pi(s) - x(s)^T w)^2 ]$, and the weight update simplifies to:

$$
\Delta w = \alpha (V^\pi(s) - \hat{v}(s, w)) x(s)
$$

- The update scales the step $\alpha$ by the prediction error and the feature value.
- Because the objective is strictly convex, SGD converges to the global minimum.

**Table lookup equivalence:** Tabular is a special case of Linear VFA with one-hot features:

$$
x^{\text{table}}(s) = (\mathbb{I}(S=s_1), \ldots, \mathbb{I}(S=s_n))^T
$$

Then $\hat{v}(s,w) = x(s)^T w = (\mathbb{I}(S=s_1), \ldots, \mathbb{I}(S=s_n)) (w_1, \ldots, w_n)^T$, yielding $\hat{v}(s_k, w) = w_k$ for state $s_k$.

---

### Model-Free Prediction (Without an Oracle)

In practice we have no oracle. We substitute the oracle value with an empirical target, yielding an incremental update:

$$
\Delta w = \alpha (Target_t - \hat{v}(s_t, w)) \nabla_w \hat{v}(s_t, w)
$$

| Target | Formula | Bias |
|--------|---------|------|
| **MC Target** | $Target_t = G_t$ | Unbiased: $\mathbb{E}[G_t] = V^\pi(S_t)$. Converges in linear and non-linear VFA. |
| **TD Target** | $Target_t = R_{t+1} + \gamma \hat{v}(S_{t+1}, w)$ | Biased: bootstraps off its own approximation. |

**TD update:**

$$
\Delta w = \alpha (R_{t+1} + \gamma \hat{v}(S_{t+1}, w) - \hat{v}(S_t, w)) \nabla_w \hat{v}(S_t, w)
$$

**Pseudocode (TD prediction with linear VFA):**

```python
def td_prediction_linear_vfa(env, policy, x, alpha, gamma, num_episodes):
    w = np.zeros(x(env.reset()).shape[0])
    for ep in range(num_episodes):
        s = env.reset()
        done = False
        while not done:
            a = policy(s)
            s_next, r, done, _ = env.step(a)
            target = r + gamma * x(s_next).dot(w)
            delta = target - x(s).dot(w)
            w += alpha * delta * x(s)
            s = s_next
    return w
```

---

### Model-Free Control (Without an Oracle)

We approximate $\hat{q}(s,a,w) \approx Q^*(s,a)$ for policy evaluation, then use $\epsilon$-greedy for improvement. The same oracle-substitution idea applies:

**With oracle:** $J(w) = \mathbb{E}_\pi [ (Q^\pi(s,a) - \hat{q}(s,a,w))^2 ]$, update $\Delta w = \alpha (Q^\pi(s,a) - \hat{q}(s,a,w)) \nabla_w \hat{q}(s,a,w)$.

**Linear VFA:** $\Delta w = \alpha (Q^\pi(s,a) - \hat{q}(s,a,w)) x(s,a)$.

**Without oracle** — incremental control:

- **SARSA Target:** $\Delta w = \alpha (R_{t+1} + \gamma \hat{q}(S_{t+1}, A_{t+1}, w) - \hat{q}(S_t, A_t, w)) \nabla_w \hat{q}(S_t, A_t, w)$
- **Q-Learning Target:** $\Delta w = \alpha (R_{t+1} + \gamma \max_a \hat{q}(S_{t+1}, a, w) - \hat{q}(S_t, A_t, w)) \nabla_w \hat{q}(S_t, A_t, w)$

**Pseudocode (Q-learning with linear VFA):**

```python
def q_learning_linear_vfa(env, x_sa, alpha, gamma, epsilon, num_episodes):
    w = np.zeros(x_sa(env.reset(), 0).shape[0])
    for ep in range(num_episodes):
        s = env.reset()
        done = False
        while not done:
            a = epsilon_greedy(s, w, x_sa, epsilon)
            s_next, r, done, _ = env.step(a)
            target = r + gamma * max(x_sa(s_next, a_).dot(w) for a_ in range(env.nA))
            delta = target - x_sa(s, a).dot(w)
            w += alpha * delta * x_sa(s, a)
            s = s_next
    return w
```

---

## 6. Batch RL and Least Squares

The incremental updates in Section 5 process one sample at a time, which is sample-inefficient. **Batch RL** instead finds the weights that best fit an entire batch of experience simultaneously, yielding closed-form solutions for linear VFA.

**Objective:** Given $D = \{\langle s_1, v_1^\pi \rangle, \ldots, \langle s_T, v_T^\pi \rangle\}$:

$$
w^* = \arg\min_w \mathbb{E}_D[(v^\pi - \hat{v}(s,w))^2] = \arg\min_w \sum_{t=1}^T (v_t^\pi - \hat{v}(s_t,w))^2
$$

For Linear VFA ($\hat{v}(s,w) = x(s)^T w$), setting the expected gradient to zero ($\mathbb{E}_D[\Delta w] = 0$):

$$
\sum_{t=1}^T x(s_t)(v_t^\pi - x(s_t)^T w) = 0
$$

Solving for $w$ yields the closed-form solution:

$$
w = \left( \sum_{t=1}^T x(s_t) x(s_t)^T \right)^{-1} \sum_{t=1}^T x(s_t) v_t^\pi
$$

*Note:* For $N$ features, the matrix inversion is $O(N^3)$.

---

### Least-Squares Targets

By substituting empirical estimates for the oracle $v^\pi$, we derive:

**1. Least-Squares Monte-Carlo (LSMC):** $v_t^\pi \approx G_t$

$$
w = \left( \sum_{t=1}^T x(s_t) x(s_t)^T \right)^{-1} \sum_{t=1}^T x(s_t) G_t
$$

**2. Least-Squares TD (LSTD):** $v_t^\pi \approx R_{t+1} + \gamma \hat{v}(S_{t+1}, w)$

$$
w = \left( \sum_{t=1}^T x(s_t)(x(s_t) - \gamma x(s_{t+1}))^T \right)^{-1} \sum_{t=1}^T x(s_t) R_{t+1}
$$

**Pseudocode (LSMC / LSTD):**

```python
def ls_mc(D, x):
    """D = [(s_t, G_t)]"""
    X = np.array([x(s) for s, _ in D])
    y = np.array([G for _, G in D])
    return np.linalg.solve(X.T @ X, X.T @ y)

def ls_td(D, x, gamma):
    """D = [(s_t, r_t, s_{t+1})]"""
    X = np.array([x(s) for s, _, _ in D])
    X_next = np.array([x(s_next) for _, _, s_next in D])
    R = np.array([r for _, r, _ in D])
    A = X.T @ (X - gamma * X_next)
    b = X.T @ R
    return np.linalg.solve(A, b)
```

---

### Least Squares Action-Value Function Control (LSDQ)

To perform **control**, we approximate $\hat{q}(s,a,w) = x(s,a)^T w \approx Q^\pi(s,a)$. To efficiently use all experience (including data from behavior policies $\pi_{\text{out}}$), we use **off-policy** learning.

- **Target policy** evaluates the greedy action: $A' = \pi_{\text{new}}(S_{t+1})$.
- **Target:** $R_{t+1} + \gamma \hat{q}(S_{t+1}, A', w)$.
- **LSDQ analytical solution:**

$$
w = \left( \sum_{t=1}^T x(S_t, A_t)(x(S_t, A_t) - \gamma x(S_{t+1}, \pi_{\text{new}}(S_{t+1})))^T \right)^{-1} \sum_{t=1}^T x(S_t, A_t) R_{t+1}
$$

LSDQ repeatedly re-evaluates the offline dataset $D$ with different target policies, effectively performing policy evaluation over $D$ until convergence.

---

## 7. Fitted Value Iteration, Fitted Q-Iteration, and The Deadly Triad

Linear VFA (Section 5–6) works well when we have hand-engineered features. For raw high-dimensional inputs — e.g., images where $|\mathcal{S}|$ is astronomical — we need neural networks. We use $V_\phi: \mathcal{S} \to \mathbb{R}$ or $Q_\phi: \mathcal{S} \times \mathcal{A} \to \mathbb{R}$ parameterized by $\phi$.

### Fitted Value Iteration (FVI)

**Algorithm:**

1. **Compute targets:** $y_i \leftarrow \max_{a_i} (r(s_i, a_i) + \gamma \mathbb{E}[V_\phi(s'_i)])$
2. **Optimize:** $\phi \leftarrow \arg\min_\phi \frac{1}{2} \sum_i \|V_\phi(s_i) - y_i\|^2$

**Fatal flaw:** Step 1 requires the expectation over $s'_i$, i.e., knowledge of $p(s'|s,a)$.

### Fitted Q-Iteration (FQI)

To bypass the need for transition dynamics, we approximate $Q(s,a)$ instead of $V(s)$. Approximate the expected next state value with a point estimate: $\mathbb{E}[V(s'_i)] \approx \max_{a'} Q_\phi(s'_i, a')$.

**Full Fitted Q-Iteration** (batch-mode, off-policy):

1. **Collect dataset:** $\{(s_i, a_i, s'_i, r_i)\}$ using any behavior policy.
2. **Compute targets:** $y_i \leftarrow r(s_i, a_i) + \gamma \max_{a'_i} Q_\phi(s'_i, a'_i)$
3. **Optimize:** $\phi \leftarrow \arg\min_\phi \frac{1}{2} \sum_i \|Q_\phi(s_i, a_i) - y_i\|^2$

The algorithm is **off-policy**: the transition $(s_i, a_i) \to (s'_i, r_i)$ is independent of the policy that collected the data. The target approximates the value of the greedy policy at $s'_i$. If optimization error $\mathcal{E} = 0$, the network satisfies the Bellman Optimality Equation: $Q_\phi(s,a) = r(s,a) + \gamma \max_{a'} Q_\phi(s', a')$.

### Online Q-Learning and the Semi-Gradient

FQI can be adapted to an **online** setting: observe $(s_i, a_i, s'_i, r_i)$, compute $y_i = r(s_i, a_i) + \gamma \max_{a'} Q_\phi(s'_i, a')$, update:

$$
\phi \leftarrow \phi - \alpha \frac{dQ_\phi}{d\phi}(s_i, a_i) (Q_\phi(s_i, a_i) - y_i)
$$

<blockquote class="callout-tip">
<p><strong>Semi-gradient.</strong> This update is <em>not</em> true gradient descent. The target $y_i$ depends on the network parameters $\phi$, so proper gradient descent would require differentiating through $y_i$. Q-learning treats $y_i$ as a fixed constant during the gradient step, forming a "semi-gradient" update.</p>
</blockquote>

---

### The Deadly Triad

While Linear VFA is stable, DNNs remove the feature-design bottleneck but introduce instability. **Divergence** is practically guaranteed when combining:

1. **Function approximation:** Generalizing from a state space much larger than memory/computation.
2. **Bootstrapping:** Updating targets from other estimates (DP, TD) rather than complete returns (MC).
3. **Off-policy training:** Training on transitions from a distribution other than the target policy.

---

### The Convergence Problem with Function Approximation

The update sequence changes from $V \leftarrow \mathcal{B}V$ (tabular) to $V \leftarrow \Pi \mathcal{B}V$ (function approximation).

<blockquote class="callout-definition">
<p><strong>Projection Operator $\Pi$.</strong> Because the hypothesis class $\Omega$ cannot represent every value function, minimizing MSE projects the Bellman backup onto the closest representable function:</p>
</blockquote>

$$
\Pi V = \arg\min_{V' \in \Omega} \frac{1}{2} \sum_s \|V'(s) - V(s)\|^2
$$

**Breakdown of convergence guarantees:**

1. $\mathcal{B}$ is a contraction in the $L_\infty$-norm.
2. $\Pi$ is a contraction in the $L_2$-norm (Euclidean distance).
3. Because the norms are mismatched, **$\Pi \mathcal{B}$ is not a contraction** of any kind.

As a result, Fitted Value Iteration, Fitted Q-Iteration, and Online Q-learning with function approximation **do not generally converge**; they can oscillate or diverge.

<blockquote class="callout-recall">
<p><strong>Corollary.</strong> Fitted bootstrapped policy evaluation in actor-critic ($y_{i,t} \approx r + \gamma \hat{V}^\pi_\phi(s_{i,t+1})$) does not converge either, because it uses the same $\Pi$ projection mechanics.</p>
</blockquote>

---

## 8. Deep Q-Networks (DQN)

Despite the lack of convergence guarantees (Section 7), **Deep Q-Networks (DQN)** achieve strong empirical performance. DQN applies Fitted Q-Iteration with neural networks, but standard online Q-learning suffers from two practical issues: (1) highly correlated sequential samples, and (2) volatile, non-stationary targets. DQN mitigates both:

<blockquote class="callout-definition">
<p><strong>Experience Replay.</strong> Stores raw transitions $(s_t, a_t, r_t, s_{t+1})$ in a replay buffer $D$. By randomly sampling mini-batches $(s, a, r, s') \sim D$, the network breaks temporal correlation before computing targets and running SGD.</p>
</blockquote>

<blockquote class="callout-definition">
<p><strong>Fixed Target Networks.</strong> Target weights $\bar{w}$ (or $w^-$) are fixed for multiple steps to prevent the target from oscillating with learning. Q-learning targets use these frozen parameters. Optionally: update via periodic copy $\bar{w} \leftarrow w$ every $C$ steps, or EMA $\bar{w} \leftarrow \rho \bar{w} + (1-\rho) w$.</p>
</blockquote>

**DQN loss and weight update:**

$$
L(w) = \mathbb{E}_{(s,a,r,s') \sim D}\left[ \left( r + \gamma \max_{a'} \hat{Q}(s', a', \bar{w}) - \hat{Q}(s, a, w) \right)^2 \right]
$$

$$
\Delta w = \alpha (r + \gamma \max_{a'} \hat{Q}(s', a', \bar{w}) - \hat{Q}(s, a, w)) \nabla_w \hat{Q}(s, a, w)
$$

**Pseudocode (DQN):**

```python
def dqn_train_step(replay_buffer, Q, Q_target, optimizer, gamma, batch_size):
    batch = replay_buffer.sample(batch_size)  # (s, a, r, s')
    s, a, r, s_next = batch
    target = r + gamma * Q_target(s_next).max(dim=1)[0]
    q_sa = Q(s).gather(1, a)
    loss = F.mse_loss(q_sa, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

## 9. Overcoming DQN Limitations (Advanced Architectures)

### Double DQN and Maximization Bias

Standard Q-learning natively suffers from **Maximization Bias**. Statistically, $\mathbb{E}[\max_n X_n] \ge \max_n \mathbb{E}[X_n]$. Because the target uses $\max_{a'} Q(s',a')$, positive noise is amplified, leading to heavily inflated Q-values.

**Double DQN** decouples action *selection* (online network $w$) from action *evaluation* (target network $\bar{w}$):

$$
\Delta w = \alpha (r + \gamma \hat{Q}(s', \arg\max_{a'} \hat{Q}(s', a', w), \bar{w}) - \hat{Q}(s, a, w)) \nabla_w \hat{Q}(s, a, w)
$$

Or equivalently: $y(s,a) = r + \gamma \hat{Q}(s', \arg\max_{a'} \hat{Q}(s', a', w), \bar{w})$.

Modern variants like **Clipped Double DQN** and **Randomized Ensemble DQN (REDQ)** extend this by taking the minimum across multiple target networks to strictly penalize overestimation.

### Dueling DQN

Dueling DQN splits the neural network into two separate, parallel processing branches: one estimates the base state-value $V(s)$, and the other estimates the specific action-advantage $A(s, a)$:

$$
Q(s, a) = V(s) + A(s, a)
$$

By decoupling these estimates, the network can learn which states are inherently valuable independent of the precise action taken, improving stability when facing many redundant actions.

### Prioritized Experience Replay (PER)

Rather than uniform sampling, PER actively assigns sampling priority to transitions with large absolute TD errors $|\delta_i|$. This ensures the network trains most frequently on the "surprising" experiences it currently understands the least. The probability of sampling transition $i$ is:

$$
P(i) = \frac{p_i}{\sum_k p_k}
$$

---

## 10. Q-Learning for Continuous Action Spaces

Standard Q-learning relies on computing $\arg\max_a Q(s,a)$, which is trivial for discrete actions but intractable for continuous action spaces. Several architectural solutions resolve this:

1. **Global Optimization (QT-Opt):** Treats the action vector as continuous and uses derivative-free global optimization (e.g., Cross-Entropy Method, CEM) to iteratively sample actions and discover the maximum value on the fly.

2. **Normalized Advantage Functions (NAF):** Mathematically restricts the Q-network to be strictly quadratic in the action:

$$
Q_\phi(s,a) = V_\phi(s) - \frac{1}{2} (a - \mu_\phi(s))^T P_\phi(s) (a - \mu_\phi(s))
$$

Because the advantage is a negative quadratic, the maximum action is trivially found by setting the derivative to zero: $a^* = \mu_\phi(s)$, yielding maximum value exactly $V_\phi(s)$.

3. **Deep Deterministic Policy Gradients (DDPG):** Maintains a separate Actor network $\mu_\theta(s)$ to predict the optimal action $a = \mu_\theta(s)$, entirely avoiding the $\max$ operator. The Actor is trained by passing gradients backwards directly through the continuous Q-network (Critic).

---

## Summary of Key Equations

| Concept | Formula |
|---------|---------|
| V-function | $V_\pi(s) = \mathbb{E}_\pi[G_0 \mid s_0 = s]$ |
| Q-function | $Q_\pi(s,a) = \mathbb{E}_\pi[G_0 \mid s_0 = s, a_0 = a]$ |
| Advantage | $A_\pi(s,a) = Q_\pi(s,a) - V_\pi(s)$ |
| Bellman optimality (V) | $V^*(s) = \max_a [R(s,a) + \gamma \mathbb{E}[V^*(s')]]$ |
| Bellman optimality (Q) | $Q^*(s,a) = R(s,a) + \gamma \mathbb{E}[\max_{a'} Q^*(s',a')]$ |
| TD update | $V(s) \leftarrow V(s) + \eta[r + \gamma V(s') - V(s)]$ |
| Q-Learning | $Q(s,a) \leftarrow Q(s,a) + \eta[r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$ |
| SARSA | $Q(s,a) \leftarrow Q(s,a) + \eta[r + \gamma Q(s',a') - Q(s,a)]$ |
| DQN loss | $L = \mathbb{E}_D[(r + \gamma \max_{a'} Q_{\bar{w}}(s',a') - Q_w(s,a))^2]$ |

---

<blockquote class="callout-recall">
<p><strong>Related notes.</strong> See <a href="/notes/rl-overview/">RL Overview</a> for foundations, <a href="/notes/rl-q-learning/">Q-Learning</a> for a compact treatment, and <a href="/notes/rl-policy-gradient/">Policy Gradient</a> for policy-based methods.</p>
</blockquote>
