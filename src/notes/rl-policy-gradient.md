---
topic: reinforcement-learning
title: Policy Gradient
summary: "Policy parameterization, REINFORCE, variance reduction (causality, baselines, entropy), off-policy gradients, GAE, surrogate advantage, and TRPO-style KL penalty."
image: images/rl-policy.svg
---

# Policy Gradient Methods

## Summary & Agenda

This note develops **policy-based reinforcement learning** from first principles. We directly parameterize and optimize the policy $\pi_\theta(a|s)$ to maximize expected return, deriving algorithms that do **not** require the transition dynamics. Here is what we accomplish:

1. **Parameterizing the Policy** — Map $\pi_\theta(a|s)$ to implementation: discrete actions (logits → log-softmax) and continuous actions (diagonal Gaussian).
2. **The Policy Gradient Objective** — Define $J(\theta)$, use the log-derivative trick to obtain a gradient form that depends only on the policy. This leads to **REINFORCE**.
3. **Reducing Variance** — Via (a) **causality** (reward-to-go), (b) **baselines**, and (c) **entropy regularization**. Derive the optimal baseline.
4. **Off-Policy Gradients & Importance Sampling** — Reuse samples from an older policy; derive the full off-policy gradient with per-decision importance sampling.
5. **The Surrogate Advantage (TRPO Foundations)** — Prove the **Performance Difference Lemma**. Estimate advantages with **GAE**. Bound the state distribution shift. Optimize a surrogate objective.
6. **Dual Gradient Ascent and KL Penalty** — Replace TV with KL via Pinsker's inequality; solve the constrained optimization via a Lagrangian.

**Applications:** Continuous action spaces (robotics, control), sample-efficient off-policy learning, and safe policy updates (TRPO, PPO).

---

## Preliminaries

We consider a finite-horizon setting with horizon $T$. A **trajectory** $\tau$ is a sequence $(s_1, a_1, r_1, s_2, a_2, r_2, \ldots, s_T, a_T, r_T)$.

<blockquote class="callout-definition">
<p><strong>Definition (Trajectory Probability).</strong> The probability of trajectory $\tau$ under policy $\pi_\theta$ is</p>
<p>$$p_\theta(\tau) = p(s_1) \prod_{t=1}^T \pi_\theta(a_t|s_t) \, p(s_{t+1}|s_t, a_t)$$</p>
<p>where $p(s_1)$ is the initial state distribution and $p(s_{t+1}|s_t, a_t)$ is the Markov transition. Only $\pi_\theta(a_t|s_t)$ depends on $\theta$.</p>
</blockquote>

<blockquote class="callout-definition">
<p><strong>Definition (Return).</strong> The return of a trajectory is the sum of rewards:</p>
<p>$$r(\tau) = \sum_{t=1}^T r(s_t, a_t)$$</p>
<p>(We use undiscounted return here; the same derivations extend to discounted return.)</p>
</blockquote>

---

## Parameterizing the Policy

The math treats $\pi_\theta(a|s)$ abstractly. For implementation (especially in robotics and continuous control), we need concrete mappings from network outputs to probabilities.

**Discrete actions.** The network outputs **logits** $\ell \in \mathbb{R}^K$ for $K$ actions. Use **log-softmax** (not softmax followed by log) for numerical stability:

$$
\log \pi_\theta(a|s) = \ell_a - \log\sum_{a'} e^{\ell_{a'}} = \text{log\_softmax}(\ell)_a
$$

*Reason:* Computing $\log(\text{softmax}(\ell)_a)$ can underflow; $\text{log\_softmax}$ stays numerically stable and gives $\nabla_\theta \log \pi_\theta$ directly for the policy gradient.

**Continuous actions.** For control tasks, the network outputs the **mean** $\mu_\theta(s) \in \mathbb{R}^d$ and (often) the **log-std** $\log \sigma_\theta(s)$ of a diagonal Gaussian:

$$
\pi_\theta(a|s) = \mathcal{N}(a; \mu_\theta(s), \sigma_\theta(s)^2 I), \quad \log \pi_\theta(a|s) = -\frac{1}{2}\sum_i \frac{(a_i - \mu_i)^2}{\sigma_i^2} - \log \sigma_i + \text{const}
$$

The gradient $\nabla_\theta \log \pi_\theta(a|s)$ flows through $\mu_\theta(s)$ and $\sigma_\theta(s)$.

---

## 1. The Policy Gradient Objective

The goal is to find optimal parameters $\theta^*$ that maximize the expected return:

$$
\begin{aligned}
J(\theta) &= \mathbb{E}_{\tau \sim p_\theta(\tau)}\left[ \sum_{t=1}^T r(s_t, a_t) \right] \\
&= \mathbb{E}_{\tau \sim p_\theta(\tau)}[r(\tau)] \\
&= \int p_\theta(\tau) \, r(\tau) \, d\tau
\end{aligned}
$$

**Step 1.** The expectation is the integral of $p_\theta(\tau) r(\tau)$ over all trajectories.

**Step 2.** Differentiate with respect to $\theta$ (assuming we can interchange gradient and integral). Since $r(\tau)$ does not depend on $\theta$, it stays inside:

$$
\nabla_\theta J(\theta) = \int \nabla_\theta p_\theta(\tau) \, r(\tau) \, d\tau
$$

**Step 3.** Apply the **log-derivative trick** (REINFORCE trick). For any differentiable density $p_\theta(x)$:

$$
\begin{aligned}
\nabla_\theta p_\theta(x) &= p_\theta(x) \cdot \frac{\nabla_\theta p_\theta(x)}{p_\theta(x)} \\
&= p_\theta(x) \, \nabla_\theta \log p_\theta(x)
\end{aligned}
$$

*Explanation:* We multiply and divide by $p_\theta(x)$; the chain rule gives $\nabla_\theta \log p_\theta = \frac{\nabla_\theta p_\theta}{p_\theta}$.

Substituting into the integral:

$$
\begin{aligned}
\nabla_\theta J(\theta) &= \int p_\theta(\tau) \, \nabla_\theta \log p_\theta(\tau) \, r(\tau) \, d\tau \\
&= \mathbb{E}_{\tau \sim p_\theta(\tau)}\left[ \nabla_\theta \log p_\theta(\tau) \, r(\tau) \right]
\end{aligned}
$$

<blockquote class="callout-lemma">
<p><strong>Lemma (Log-Derivative Trick).</strong> For a differentiable density $p_\theta(x)$: $\nabla_\theta p_\theta(x) = p_\theta(x) \, \nabla_\theta \log p_\theta(x)$.</p>
</blockquote>

**Step 4.** Expand $\nabla_\theta \log p_\theta(\tau)$. Using the trajectory factorization (with index $k$ to avoid confusion when we later take gradients):

$$
p_\theta(\tau) = p(s_1) \prod_{k=1}^T \pi_\theta(a_k|s_k) \, p(s_{k+1}|s_k, a_k)
$$

Taking the log turns the product into a sum:

$$
\begin{aligned}
\log p_\theta(\tau) &= \log p(s_1) + \sum_{t=1}^T \log \pi_\theta(a_t|s_t) + \sum_{t=1}^T \log p(s_{t+1}|s_t, a_t)
\end{aligned}
$$

The terms $p(s_1)$ and $p(s_{t+1}|s_t, a_t)$ do **not** depend on $\theta$—they are environment dynamics—so their gradients vanish. Only the policy terms remain:

$$
\nabla_\theta \log p_\theta(\tau) = \sum_{t=1}^T \nabla_\theta \log \pi_\theta(a_t|s_t)
$$

*Explanation:* We do **not** need to model the environment transition dynamics.

<blockquote class="callout-tip">
<p><strong>Key insight.</strong> The Markov property is not strictly required. Policy gradients work even in non-Markovian environments, as long as we can sample trajectories and compute $\nabla_\theta \log \pi_\theta(a|s)$.</p>
</blockquote>

**Step 5.** Substitute back into the gradient:

$$
\begin{aligned}
\nabla_\theta J(\theta) &= \mathbb{E}_{\tau \sim p_\theta(\tau)}\left[ \nabla_\theta \log p_\theta(\tau) \, r(\tau) \right] \\
&= \mathbb{E}_{\tau \sim p_\theta(\tau)}\left[ \left( \sum_{t=1}^T \nabla_\theta \log \pi_\theta(a_t|s_t) \right) \left( \sum_{t=1}^T r(s_t, a_t) \right) \right]
\end{aligned}
$$

*Explanation:* We now have a form we can estimate by sampling trajectories. No derivatives of the environment appear.

---

## The REINFORCE Algorithm

We estimate the gradient with a Monte Carlo average over $N$ sampled trajectories:

$$
\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^N \left[ \sum_{t=1}^T \nabla_\theta \log \pi_\theta(a_{i,t}|s_{i,t}) \right] \left( \sum_{t=1}^T r(s_{i,t}, a_{i,t}) \right)
$$

**REINFORCE steps:**

1. **Sample** trajectories by rolling out the current policy in the environment (or on a robot).
2. **Compute** the empirical gradient (sum of score functions weighted by returns; good trajectories get upweighted).
3. **Update** parameters: $\theta \leftarrow \theta + \alpha \, \nabla_\theta J(\theta)$ (gradient ascent).

<blockquote class="callout-definition">
<p><strong>Definition (REINFORCE).</strong> A Monte Carlo policy gradient algorithm that samples trajectories, computes $\nabla_\theta \log \pi_\theta(a_t|s_t)$ at each step, weights by total return $r(\tau)$, and performs gradient ascent.</p>
</blockquote>

**Trajectory data structure.** Use a strict, type-safe representation for rollouts. Dataclasses (or namedtuples) keep batches easy to debug and pass into gradient updates:

```python
from dataclasses import dataclass
from typing import List
import numpy as np

@dataclass
class Transition:
    state: np.ndarray
    action: np.ndarray
    reward: float
    log_prob: float  # log π_θ(a|s), stored at sample time

@dataclass
class Trajectory:
    transitions: List[Transition]
    def states(self): return np.stack([t.state for t in self.transitions])
    def actions(self): return np.stack([t.action for t in self.transitions])
    def rewards(self): return np.array([t.reward for t in self.transitions])
    def log_probs(self): return np.array([t.log_prob for t in self.transitions])
```

**Pseudocode:**

```python
def REINFORCE(env, policy, alpha, num_iterations, batch_size, T):
    for _ in range(num_iterations):
        trajectories = [rollout(env, policy, T) for _ in range(batch_size)]
        grad = 0
        for tau in trajectories:
            score = tau.log_probs().sum()  # sum_t ∇_θ log π_θ(a_t|s_t)
            reward = tau.rewards().sum()   # r(τ)
            grad += score * reward
        policy.theta += alpha * (grad / batch_size)
```

---

## 2. Reducing Variance

A major issue with REINFORCE is **high variance** in the gradient estimate. Variance $\text{Var}(X) = \mathbb{E}[X^2] - \mathbb{E}[X]^2$ grows with trajectory length $T$.

---

### Hack 1: Causality (Reward-to-Go)

**Idea:** Future actions cannot influence past rewards. So we can drop past rewards from each term without changing the expectation.

<blockquote class="callout-lemma">
<p><strong>Lemma (Score Function Has Zero Mean).</strong> $\mathbb{E}_{\tau \sim p_\theta(\tau)}[\nabla_\theta \log p_\theta(\tau)] = 0$.</p>
</blockquote>

**Proof:**

$$
\begin{aligned}
\mathbb{E}[\nabla_\theta \log p_\theta(\tau)] &= \int p_\theta(\tau) \, \nabla_\theta \log p_\theta(\tau) \, d\tau \\
&= \int p_\theta(\tau) \frac{\nabla_\theta p_\theta(\tau)}{p_\theta(\tau)} \, d\tau \\
&= \int \nabla_\theta p_\theta(\tau) \, d\tau \\
&= \nabla_\theta \int p_\theta(\tau) \, d\tau \\
&= \nabla_\theta 1 \\
&= 0
\end{aligned}
$$

*Explanation:* The log-derivative trick gives $p_\theta \nabla_\theta \log p_\theta = \nabla_\theta p_\theta$; the integral of a density over the sample space is 1. ∎

Because $\pi_\theta(a_t|s_t)$ is independent of past rewards $r(s_{t'}, a_{t'})$ for $t' < t$ over the trajectory distribution:

$$
\mathbb{E}\left[ \nabla_\theta \log \pi_\theta(a_t|s_t) \, r(s_{t'}, a_{t'}) \right] = 0 \quad \text{for } t' < t
$$

*Explanation:* Given $s_t$, the policy at time $t$ does not affect rewards before $t$; the expectation over the joint distribution makes the cross-term zero.

Therefore we can replace the full return with the **reward-to-go** from time $t$:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim p_\theta(\tau)}\left[ \sum_{t=1}^T \nabla_\theta \log \pi_\theta(a_t|s_t) \left( \sum_{t'=t}^T r(s_{t'}, a_{t'}) \right) \right]
$$

*Explanation:* For each $t$, we only need rewards from $t$ onward. This reduces variance because each term sums fewer random variables.

<blockquote class="callout-recall">
<p><strong>Recall.</strong> We used the same causality idea when expanding $\nabla_\theta \log p_\theta(\tau)$: past transitions do not depend on the current policy parameters. Here we use it again to drop past rewards from each gradient term.</p>
</blockquote>

---

### Hack 2: Baselines

We can subtract a **baseline** $b$ (a scalar or state-dependent function) from the return. The gradient expectation is unchanged.

**Unbiasedness:** We need $\mathbb{E}[\nabla_\theta \log p_\theta(\tau) \cdot b] = 0$. When $b$ is a constant, this holds because $\mathbb{E}[\nabla_\theta \log p_\theta(\tau)] = 0$ (see the lemma above). So

$$
\mathbb{E}\left[ \nabla_\theta \log p_\theta(\tau) \, (r(\tau) - b) \right] = \mathbb{E}\left[ \nabla_\theta \log p_\theta(\tau) \, r(\tau) \right] - b \, \mathbb{E}\left[ \nabla_\theta \log p_\theta(\tau) \right] = \mathbb{E}\left[ \nabla_\theta \log p_\theta(\tau) \, r(\tau) \right]
$$

*Explanation:* The baseline term contributes zero in expectation, so the estimator remains unbiased. Subtracting a baseline can only reduce (or leave unchanged) the variance.

<blockquote class="callout-tip">
<p><strong>Intuition.</strong> If $b$ approximates the average return, then $r(\tau) - b$ measures how much better or worse this trajectory was. We reinforce good trajectories and discourage bad ones, instead of reinforcing everything when $r(\tau) > 0$.</p>
</blockquote>

---

### Optimal Baseline

To minimize variance, we choose $b$ that minimizes $\text{Var}(\nabla_\theta \log p_\theta(\tau)(r(\tau) - b))$.

Let $g(\tau) = \nabla_\theta \log p_\theta(\tau)$. The variance of the baseline-adjusted gradient estimate $\hat{g} = g(\tau)(r(\tau) - b)$ is:

$$
\begin{aligned}
\text{Var}(\hat{g}) &= \mathbb{E}_{\tau \sim p_\theta(\tau)}\left[ (g(\tau)(r(\tau) - b))^2 \right] \\
&\quad - \mathbb{E}_{\tau \sim p_\theta(\tau)}\left[ g(\tau)(r(\tau) - b) \right]^2
\end{aligned}
$$

Because $\mathbb{E}[g(\tau)] = 0$, the second term simplifies to $\mathbb{E}[g(\tau)\, r(\tau)]^2$, which does **not** depend on $b$. So we minimize only the first term: $\mathbb{E}[g(\tau)^2 (r(\tau) - b)^2]$.

**Step 1.** Expand the quadratic and take expectation:

$$
(r(\tau) - b)^2 = r(\tau)^2 - 2b\, r(\tau) + b^2
$$

$$
\begin{aligned}
\mathbb{E}\left[ g(\tau)^2 (r(\tau) - b)^2 \right] &= \mathbb{E}[g(\tau)^2 r(\tau)^2] - 2b \, \mathbb{E}[g(\tau)^2 r(\tau)] + b^2 \, \mathbb{E}[g(\tau)^2]
\end{aligned}
$$

**Step 2.** Take the derivative with respect to $b$:

$$
\begin{aligned}
\frac{d}{db} \mathbb{E}\left[ g(\tau)^2 (r(\tau) - b)^2 \right] &= \frac{d}{db} \left( \mathbb{E}[g(\tau)^2 r(\tau)^2] - 2b \, \mathbb{E}[g(\tau)^2 r(\tau)] + b^2 \, \mathbb{E}[g(\tau)^2] \right) \\
&= -2 \, \mathbb{E}[g(\tau)^2 r(\tau)] + 2b \, \mathbb{E}[g(\tau)^2]
\end{aligned}
$$

Set to zero to find the minimizer:

$$
-2 \, \mathbb{E}[g(\tau)^2 r(\tau)] + 2b \, \mathbb{E}[g(\tau)^2] = 0
$$

**Step 3.** Solve for $b$:

$$
b^* = \frac{\mathbb{E}[g(\tau)^2 \, r(\tau)]}{\mathbb{E}[g(\tau)^2]}
$$

*Explanation:* This is a weighted average of $r(\tau)$, with weights proportional to $g(\tau)^2$ (the squared gradient magnitude). In practice, we often use a learned value function $V(s)$ as a state-dependent baseline.

<blockquote class="callout-recall">
<p><strong>Recall.</strong> The score function $\nabla_\theta \log p_\theta(\tau)$ has zero mean. The baseline exploits that to subtract a term that leaves the expectation unchanged but reduces variance.</p>
</blockquote>

**Algorithm: REINFORCE with baseline**

```python
def REINFORCE_with_baseline(env, policy, value_fn, alpha, alpha_v, batch_size, T, num_iterations):
    for _ in range(num_iterations):
        trajectories = [rollout(env, policy, T) for _ in range(batch_size)]
        grad_pi, grad_v = 0, 0
        for tau in trajectories:
            returns = np.cumsum(tau.rewards()[::-1])[::-1]  # reward-to-go
            b = value_fn(tau.states()).squeeze()  # or running mean of returns
            advantages = returns - b
            grad_pi += (tau.log_probs() * advantages).sum()
            grad_v += ((value_fn(tau.states()).squeeze() - returns) ** 2).sum()
        policy.theta += alpha * grad_pi / batch_size
        value_fn.theta -= alpha_v * grad_v / batch_size  # MSE loss
```

---

### Entropy Regularization

Modern policy gradients almost always add an **entropy bonus** to encourage exploration and prevent premature convergence.

**Modified objective:**

$$
\tilde{J}(\theta) = J(\theta) + \alpha \, \mathbb{E}_{\tau \sim p_\theta}\left[ \sum_{t=1}^T H(\pi_\theta(\cdot|s_t)) \right]
$$

where $H(\pi(\cdot|s)) = -\mathbb{E}_{a \sim \pi}[\log \pi(a|s)]$ is the entropy of the policy at state $s$, and $\alpha > 0$ is a hyperparameter.

**Gradient:** The entropy term is differentiable. For a Gaussian policy, $H = \frac{1}{2}\log(2\pi e \sigma^2)$; for a categorical policy, $H = -\sum_a \pi(a|s) \log \pi(a|s)$. The total gradient becomes:

$$
\nabla_\theta \tilde{J} = \nabla_\theta J + \alpha \sum_t \nabla_\theta H(\pi_\theta(\cdot|s_t))
$$

*Intuition:* Higher entropy means more exploration; the bonus prevents the policy from collapsing to a deterministic strategy too early.

---

## 3. Off-Policy Gradients & Importance Sampling

Standard policy gradients are **on-policy**: we need fresh samples from the current policy. To reuse samples from an older policy $\bar{p}(\tau)$, we use **importance sampling**.

<blockquote class="callout-definition">
<p><strong>Importance Sampling Identity.</strong> For any distributions $p$ and $q$ with $\text{supp}(p) \subseteq \text{supp}(q)$:</p>
<p>$$\mathbb{E}_{x \sim p}[f(x)] = \mathbb{E}_{x \sim q}\left[ \frac{p(x)}{q(x)} f(x) \right]$$</p>
</blockquote>

**Proof:**

$$
\begin{aligned}
\mathbb{E}_{x \sim p}[f(x)] &= \int p(x) \, f(x) \, dx \\
&= \int q(x) \, \frac{p(x)}{q(x)} \, f(x) \, dx \\
&= \mathbb{E}_{x \sim q}\left[ \frac{p(x)}{q(x)} f(x) \right]
\end{aligned}
$$

*Explanation:* We rewrite the integral under $p$ as an integral under $q$, with the importance weight $\frac{p(x)}{q(x)}$ correcting the distribution. ∎

We sample from $\bar{p}(\tau)$ (e.g., an older policy) but want the expectation under $p_\theta(\tau)$:

$$
J(\theta) = \mathbb{E}_{\tau \sim p_\theta(\tau)}[r(\tau)] = \mathbb{E}_{\tau \sim \bar{p}(\tau)}\left[ \frac{p_\theta(\tau)}{\bar{p}(\tau)} r(\tau) \right]
$$

*Explanation:* We use the identity with $p = p_\theta$, $q = \bar{p}$, and $f = r$.

---

### Trajectory Ratio Simplification

When computing $\frac{p_{\theta'}(\tau)}{p_\theta(\tau)}$, expand both numerator and denominator:

$$
\begin{aligned}
\frac{p_{\theta'}(\tau)}{p_\theta(\tau)} &= \frac{p(s_1) \prod_{t=1}^T \pi_{\theta'}(a_t|s_t) \, p(s_{t+1}|s_t, a_t)}{p(s_1) \prod_{t=1}^T \pi_\theta(a_t|s_t) \, p(s_{t+1}|s_t, a_t)} \\
&= \prod_{t=1}^T \frac{\pi_{\theta'}(a_t|s_t)}{\pi_\theta(a_t|s_t)}
\end{aligned}
$$

*Explanation:* The initial state $p(s_1)$ and transition dynamics $p(s_{t+1}|s_t, a_t)$ do **not** depend on $\theta$, so they cancel. Only the ratio of policy probabilities remains.

<blockquote class="callout-recall">
<p><strong>Recall.</strong> In the on-policy gradient, $\nabla_\theta \log p_\theta(\tau)$ also dropped the environment terms. Here we see the same simplification for the importance weight.</p>
</blockquote>

---

### Off-Policy Gradient Estimator

**Step 1.** We want $\nabla_{\theta'} J(\theta')$ but have samples from $p_\theta(\tau)$. Apply importance sampling to the policy gradient formula:

$$
\nabla_{\theta'} J(\theta') = \mathbb{E}_{\tau \sim p_\theta(\tau)}\left[ \frac{p_{\theta'}(\tau)}{p_\theta(\tau)} \, \nabla_{\theta'} \log p_{\theta'}(\tau) \, r(\tau) \right]
$$

*Explanation:* The expectation is over $\tau \sim p_\theta$; the importance weight $\frac{p_{\theta'}(\tau)}{p_\theta(\tau)}$ corrects for sampling from the wrong distribution. The gradient is taken with respect to $\theta'$.

**Step 2.** Expand $\nabla_{\theta'} \log p_{\theta'}(\tau)$ (same as on-policy: environment terms drop):

$$
\nabla_{\theta'} \log p_{\theta'}(\tau) = \sum_{t=1}^T \nabla_{\theta'} \log \pi_{\theta'}(a_t|s_t)
$$

**Step 3.** Before applying causality, the full expansion (with trajectory ratio, score, and return) is:

$$
\begin{aligned}
\nabla_{\theta'} J(\theta') &= \mathbb{E}_{\tau \sim p_\theta(\tau)} \Bigg[ \left( \prod_{t=1}^T \frac{\pi_{\theta'}(a_t|s_t)}{\pi_\theta(a_t|s_t)} \right) \\
&\quad \cdot \left( \sum_{t=1}^T \nabla_{\theta'} \log \pi_{\theta'}(a_t|s_t) \right) \left( \sum_{t=1}^T r(s_t, a_t) \right) \Bigg]
\end{aligned}
$$

**Step 4.** Apply causality (reward-to-go): past rewards do not depend on future actions. The importance weight at time $t$ must correct only for $(s_1, a_1, \ldots, s_t, a_t)$, so we use the per-decision weight $\prod_{t'=1}^t \frac{\pi_{\theta'}}{\pi_\theta}$ and reward-to-go $\sum_{t'=t}^T r(s_{t'}, a_{t'})$:

$$
\begin{aligned}
\nabla_{\theta'} J(\theta') &= \mathbb{E}_{\tau \sim p_\theta(\tau)} \Bigg[ \sum_{t=1}^T \nabla_{\theta'} \log \pi_{\theta'}(a_t|s_t) \\
&\quad \cdot \left( \prod_{t'=1}^t \frac{\pi_{\theta'}(a_{t'}|s_{t'})}{\pi_\theta(a_{t'}|s_{t'})} \right) \left( \sum_{t'=t}^T r(s_{t'}, a_{t'}) \right) \Bigg]
\end{aligned}
$$

*Explanation:* This is the off-policy policy gradient with per-decision importance sampling.

---

## 4. The Surrogate Advantage (TRPO Foundations)

We can view policy gradient as a form of **policy iteration**. The key is the **Performance Difference Lemma**, which we prove below.

<blockquote class="callout-definition">
<p><strong>Performance Difference Lemma.</strong> For two policies $\pi_\theta$ and $\pi_{\theta'}$, with $A^{\pi_\theta}(s, a) = Q^{\pi_\theta}(s, a) - V^{\pi_\theta}(s)$:</p>
<p>$$J(\theta') - J(\theta) = \mathbb{E}_{\tau \sim p_{\theta'}(\tau)}\left[ \sum_{t=0}^{\infty} \gamma^t A^{\pi_\theta}(s_t, a_t) \right]$$</p>
</blockquote>

**Full telescoping proof.** We work in the discounted infinite-horizon setting. Write $J(\theta) = \mathbb{E}_{s_0 \sim \rho}[V^{\pi_\theta}(s_0)]$ and $J(\theta') = \mathbb{E}_{\tau \sim p_{\theta'}(\tau)}[\sum_{t=0}^\infty \gamma^t r(s_t, a_t)]$.

**Step 1.** Express the difference in terms of the old policy's value at the start:

$$
J(\theta') - J(\theta) = J(\theta') - \mathbb{E}_{s_0 \sim \rho}[V^{\pi_\theta}(s_0)]
$$

**Step 2.** The initial state $s_0$ has the same distribution regardless of policy. Evaluate the second term over trajectories from $\pi_{\theta'}$:

$$
= J(\theta') - \mathbb{E}_{\tau \sim p_{\theta'}(\tau)}[V^{\pi_\theta}(s_0)]
$$

**Step 3.** Telescoping identity: $V^{\pi_\theta}(s_0) = \sum_{t=0}^\infty \gamma^t V^{\pi_\theta}(s_t) - \sum_{t=1}^\infty \gamma^t V^{\pi_\theta}(s_t)$. Shift the second sum by $t \mapsto t-1$:

$$
\begin{aligned}
V^{\pi_\theta}(s_0) &= \sum_{t=0}^\infty \gamma^t V^{\pi_\theta}(s_t) - \gamma \sum_{t=0}^\infty \gamma^t V^{\pi_\theta}(s_{t+1}) \\
&= \sum_{t=0}^\infty \gamma^t \left( V^{\pi_\theta}(s_t) - \gamma V^{\pi_\theta}(s_{t+1}) \right)
\end{aligned}
$$

**Step 4.** The Bellman equation gives $V^{\pi_\theta}(s_t) = \mathbb{E}[r(s_t, a_t) + \gamma V^{\pi_\theta}(s_{t+1}) \mid s_t]$. Hence $V^{\pi_\theta}(s_t) - \gamma V^{\pi_\theta}(s_{t+1}) = r(s_t, a_t) - A^{\pi_\theta}(s_t, a_t)$, where $A^{\pi_\theta}(s_t, a_t) = r(s_t, a_t) + \gamma V^{\pi_\theta}(s_{t+1}) - V^{\pi_\theta}(s_t)$ (the TD error / advantage).

**Step 5.** Substitute into the objective. We have $J(\theta') = \mathbb{E}_{\tau \sim p_{\theta'}}[\sum_{t=0}^\infty \gamma^t r(s_t, a_t)]$:

$$
\begin{aligned}
J(\theta') - J(\theta) &= \mathbb{E}_{\tau \sim p_{\theta'}(\tau)} \left[ \sum_{t=0}^\infty \gamma^t r(s_t, a_t) - \sum_{t=0}^\infty \gamma^t \left( V^{\pi_\theta}(s_t) - \gamma V^{\pi_\theta}(s_{t+1}) \right) \right] \\
&= \mathbb{E}_{\tau \sim p_{\theta'}(\tau)} \left[ \sum_{t=0}^\infty \gamma^t \left( r(s_t, a_t) + \gamma V^{\pi_\theta}(s_{t+1}) - V^{\pi_\theta}(s_t) \right) \right]
\end{aligned}
$$

**Step 6.** The quantity in parentheses is exactly $A^{\pi_\theta}(s_t, a_t)$:

$$
J(\theta') - J(\theta) = \mathbb{E}_{\tau \sim p_{\theta'}(\tau)} \left[ \sum_{t=0}^\infty \gamma^t A^{\pi_\theta}(s_t, a_t) \right]
$$

∎

*Explanation:* The improvement equals the expected sum of advantages under the *new* policy. Actions are taken by $\pi_{\theta'}$, but advantages are evaluated under $\pi_\theta$.

**The challenge:** The expectation is over $p_{\theta'}(\tau)$, the state distribution under the *new* policy. We do not have samples from $\theta'$ yet.

**Approximation:** If $\pi_\theta$ and $\pi_{\theta'}$ are close, then $p_\theta(s_t) \approx p_{\theta'}(s_t)$. We can optimize using the *old* state distribution. To justify this, we must bound how much the state distribution shifts.

---

### Bounding the State Distribution Shift

We need to justify using the *old* state distribution $p_\theta(s_t)$ when optimizing the surrogate. If the policy changes too much, the state distribution under the new policy $\pi_{\theta'}$ could drift far from $p_\theta$, and our surrogate would be a poor approximation.

<blockquote class="callout-tip">
<p><strong>Intuition.</strong> If $\pi_{\theta'}$ is close to $\pi_\theta$ at every state, then the probability of following the same trajectory as the old policy is high. With high probability we stay in states the old policy would have visited, so $p_{\theta'}(s_t) \approx p_\theta(s_t)$. Bounding the policy difference (via Total Variation) lets us bound how much the state distribution can shift.</p>
</blockquote>

---

**Formal setup.** Suppose $D_{\text{TV}}(\pi_{\theta'}(\cdot|s), \pi_\theta(\cdot|s)) \le \epsilon$ for all $s$. The probability of taking a *different* action than $\pi_\theta$ at any step is at most $\epsilon$; the probability of making *no* deviations up to time $t$ is at least $(1-\epsilon)^t$.

**Formal proof.** The new state distribution can be written as a mixture:

$$
p_{\theta'}(s_t) = (1-\epsilon)^t \, p_\theta(s_t) + \left(1 - (1-\epsilon)^t\right) \, p_{\text{mistake}}(s_t)
$$

where $p_{\text{mistake}}$ is the state distribution conditioned on at least one deviation.

The Total Variation between $p_{\theta'}(s_t)$ and $p_\theta(s_t)$ is:

$$
\begin{aligned}
D_{\text{TV}}(p_{\theta'}(s_t), p_\theta(s_t)) &= \frac{1}{2} \sum_{s_t} \left| p_{\theta'}(s_t) - p_\theta(s_t) \right| \\
&= \left(1 - (1-\epsilon)^t\right) \cdot \frac{1}{2} \sum_{s_t} \left| p_{\text{mistake}}(s_t) - p_\theta(s_t) \right| \\
&\le 1 - (1-\epsilon)^t \le \epsilon t
\end{aligned}
$$

(using $(1-\epsilon)^t \ge 1 - \epsilon t$ for $\epsilon \in [0,1]$).

**Conclusion:** $D_{\text{TV}}(p_{\theta'}(s_t), p_\theta(s_t)) \le \epsilon t$. For small $\epsilon$, the state distribution does not drift far; using $p_\theta$ in the surrogate is justified.

---

### Surrogate Objective

Optimize the following using the *old* state distribution:

$$
\theta' \leftarrow \arg\max_{\theta'} \sum_t \mathbb{E}_{s_t \sim p_\theta}\left[ \mathbb{E}_{a_t \sim \pi_\theta}\left[ \frac{\pi_{\theta'}(a_t|s_t)}{\pi_\theta(a_t|s_t)} A^{\pi_\theta}(s_t, a_t) \right] \right]
$$

*Explanation:* We sample $s_t$ from $p_\theta$ and $a_t$ from $\pi_\theta$, then weight by the importance ratio $\frac{\pi_{\theta'}}{\pi_\theta}$ and the advantage. For small enough policy changes, optimizing this surrogate improves $J(\theta') - J(\theta)$.

---

### Generalized Advantage Estimation (GAE)

The surrogate objective requires estimates of $A^{\pi_\theta}(s_t, a_t)$. The definition $A(s,a) = Q(s,a) - V(s)$ is clear theoretically, but in practice we estimate it using **Generalized Advantage Estimation (GAE)**.

**TD residual.** The one-step TD error $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ is an unbiased estimate of $A(s_t, a_t)$ (given a value function $V$).

**GAE.** We form a $\lambda$-weighted average of $k$-step TD residuals to trade off bias and variance:

$$
\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}
$$

**Recursive form (used in implementation):**

$$
\hat{A}_t = \delta_t + \gamma \lambda \hat{A}_{t+1}
$$

with $\hat{A}_T = 0$ at the end of the trajectory.

**Interpretation:**
- $\lambda = 0$: $\hat{A}_t = \delta_t$ (low variance, high bias)
- $\lambda = 1$: $\hat{A}_t = \sum_{l \ge 0} \gamma^l \delta_{t+l} = \text{reward-to-go} - V(s_t)$ (Monte Carlo; higher variance, lower bias)

Typical choice: $\lambda \in [0.95, 0.99]$. The value function $V$ is learned alongside the policy (e.g., with TD or MSE loss on returns).

---

## 5. Dual Gradient Ascent and KL Penalty

Total Variation is hard to optimize. We use **KL divergence** instead, via **Pinsker's inequality**:

$$
\frac{1}{2} \sum_a \left| \pi_{\theta'}(a|s) - \pi_\theta(a|s) \right| \le \sqrt{\frac{1}{2} D_{\text{KL}}(\pi_\theta(\cdot|s) \Vert \pi_{\theta'}(\cdot|s))}
$$

*Explanation:* Bounding KL divergence bounds TV divergence. Minimizing KL with respect to $\theta'$ keeps the new policy close to the old one.

**KL from samples:** KL divergence can be estimated using sampled trajectories:

$$
D_{\text{KL}}(p_1 \Vert p_2) = \mathbb{E}_{x \sim p_1}\left[ \log \frac{p_1(x)}{p_2(x)} \right]
$$

For policies: $D_{\text{KL}}(\pi_\theta \Vert \pi_{\theta'}) = \mathbb{E}_{s,a \sim \pi_\theta}[\log \pi_\theta(a|s) - \log \pi_{\theta'}(a|s)]$.

**Constrained optimization:**

$$
\begin{aligned}
&\max_{\theta'} \sum_{t} \mathbb{E}_{s_t \sim p_\theta(s_t)}\left[ \mathbb{E}_{a_t \sim \pi_\theta(a_t|s_t)}\left[ \frac{\pi_{\theta'}(a_t|s_t)}{\pi_\theta(a_t|s_t)} A^{\pi_\theta}(s_t, a_t) \right] \right] \\
&\text{subject to } \quad D_{\text{KL}}(\pi_\theta(a_t|s_t) \Vert \pi_{\theta'}(a_t|s_t)) \le \epsilon
\end{aligned}
$$

*Explanation:* Maximize the surrogate advantage while keeping the policy within an $\epsilon$-KL ball of the old policy.

---

### Lagrangian and Dual Gradient Ascent

Introduce the Lagrangian with multiplier $\beta \ge 0$:

$$
\begin{aligned}
\mathcal{L}(\theta', \beta) &= \sum_{t} \mathbb{E}_{s_t \sim p_\theta}\left[ \mathbb{E}_{a_t \sim \pi_\theta}\left[ \frac{\pi_{\theta'}(a_t|s_t)}{\pi_\theta(a_t|s_t)} A^{\pi_\theta}(s_t, a_t) \right] \right] \\
&\quad - \beta \left( D_{\text{KL}}(\pi_\theta \Vert \pi_{\theta'}) - \epsilon \right)
\end{aligned}
$$

*Explanation:* The constraint $D_{\text{KL}} \le \epsilon$ is penalized; $\beta$ controls the trade-off.

**Algorithm:**

1. **Maximize** $\mathcal{L}(\theta', \beta)$ with respect to $\theta'$ using gradient ascent on sampled trajectories.
2. **Update** $\beta$: $\beta \leftarrow \beta + \alpha_\beta (D_{\text{KL}} - \epsilon)$. If $D_{\text{KL}} > \epsilon$, increase $\beta$ to penalize large policy changes.

*Explanation:* Dual gradient ascent alternates between improving the surrogate (step 1) and tightening the constraint (step 2). This is the foundation of Trust Region Policy Optimization (TRPO) and related algorithms like PPO.

**Algorithm: TRPO-style dual gradient ascent**

```python
def TRPO_dual_ascent(env, policy, value_fn, epsilon, alpha_pi, alpha_beta, batch_size, T, num_iterations):
    beta = 0.1
    for _ in range(num_iterations):
        trajectories = [rollout(env, policy, T) for _ in range(batch_size)]
        # 1. Compute GAE advantages for each trajectory
        for tau in trajectories:
            tau.advantages = compute_GAE(tau, value_fn, gamma=0.99, lam=0.95)
        # 2. Surrogate objective: E[(π_θ'/π_θ) A]
        #    Store old log_probs at sample time; new log_probs = policy.log_prob(s, a)
        ratios = exp(new_log_probs - old_log_probs)
        surrogate = (ratios * advantages).mean()
        # 3. KL penalty (estimate from samples)
        kl = (old_log_probs - new_log_probs).mean()
        loss = -surrogate + beta * max(0, kl - epsilon)
        # 4. Gradient ascent on policy
        policy.theta += alpha_pi * grad(loss, policy.theta)
        # 5. Dual update: increase β if constraint violated
        beta += alpha_beta * max(0, kl - epsilon)
        # 6. Update value function (e.g., MSE on returns)
        update_value_fn(value_fn, trajectories)
```

<blockquote class="callout-recall">
<p><strong>Recall.</strong> The surrogate objective $\frac{\pi_{\theta'}}{\pi_\theta} A^{\pi_\theta}$ appeared in Section 4. Here we add the KL constraint and solve it via the Lagrangian.</p>
</blockquote>

---

## Summary of Key Equations

| Concept | Formula |
|---------|---------|
| Objective | $J(\theta) = \mathbb{E}_{\tau \sim p_\theta(\tau)}[r(\tau)]$ |
| With entropy bonus | $\tilde{J} = J + \alpha \sum_t H(\pi_\theta(\cdot \mid s_t))$ |
| Log-derivative trick | $\nabla_\theta p_\theta = p_\theta \, \nabla_\theta \log p_\theta$ |
| Trajectory gradient | $\nabla_\theta \log p_\theta(\tau) = \sum_{t=1}^T \nabla_\theta \log \pi_\theta(a_t \mid s_t)$ |
| Policy gradient | $\nabla_\theta J = \mathbb{E}\left[ \left( \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \right) r(\tau) \right]$ |
| With causality | $\nabla_\theta J = \mathbb{E}\left[ \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \left( \sum_{t'=t}^T r_{t'} \right) \right]$ |
| Optimal baseline | $b^* = \frac{\mathbb{E}[g^2 r]}{\mathbb{E}[g^2]}$ |
| GAE | $\hat{A}_t = \sum_{l \ge 0} (\gamma \lambda)^l \delta_{t+l}$, $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ |
| Importance weight | $\frac{p_{\theta'}(\tau)}{p_\theta(\tau)} = \prod_t \frac{\pi_{\theta'}(a_t \mid s_t)}{\pi_\theta(a_t \mid s_t)}$ |
| Surrogate objective | $\max_{\theta'} \mathbb{E}\left[ \frac{\pi_{\theta'}}{\pi_\theta} A^{\pi_\theta} \right]$ s.t. $D_{\text{KL}}(\pi_\theta \Vert \pi_{\theta'}) \le \epsilon$ |

---

## Practical Tips

<blockquote class="callout-tip">
<p><strong>Tip.</strong> Use reward-to-go, baselines (e.g., a learned $V(s)$), and entropy regularization to reduce variance and encourage exploration. Estimate advantages with GAE ($\lambda \approx 0.95$--$0.99$). For stable updates, constrain policy change via KL (TRPO) or clip the importance ratio (PPO). Store rollouts in typed dataclasses for clean batching and debugging.</p>
</blockquote>
