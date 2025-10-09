## Neural Thompson Sampling  
**Reference:**  
Weitong Zhang, Dongruo Zhou, Lihong Li, and Quanquan Gu. (2020). Neural Thompson Sampling. *arXiv preprint arXiv:2010.00827.*

---

### Goal and Motivation  
This paper extends Thompson Sampling (TS) to neural network models for contextual bandits. In traditional TS, the agent samples parameters from a posterior distribution and selects the action that maximizes the expected reward under those sampled parameters. Neural Thompson Sampling (NeuralTS) generalizes this idea by constructing a posterior over functions represented by a neural network, making it suitable for non-linear reward structures.

While NeuralUCB (Zhou et al., 2020) addresses exploration through deterministic confidence bounds, NeuralTS introduces stochastic exploration through posterior sampling. Both methods rely on the neural tangent kernel (NTK) to approximate the neural network’s local behavior, but NeuralTS interprets the resulting covariance as a distribution over possible functions rather than an uncertainty radius.

---

### Core Idea  
NeuralTS maintains a posterior over the neural network’s predicted reward function using the NTK features. The mean of this posterior is the current network prediction \( f(x; \theta_t) \), and its covariance is derived from gradients with respect to network parameters:
\[
g(x; \theta_t) = \nabla_\theta f(x; \theta_t),
\]
and the covariance matrix:
\[
\Sigma_t = \lambda I + \sum_{s=1}^{t-1} g(x_s; \theta_s) g(x_s; \theta_s)^\top.
\]
At each time step, the algorithm samples a function from this posterior by drawing a parameter vector perturbation consistent with the NTK covariance. The agent then chooses the arm with the highest predicted reward from that sampled function and updates both the posterior and network parameters based on the observed reward.

This framework leads to a regret bound of order \( \tilde{O}(\sqrt{T}) \), comparable to NeuralUCB, though obtained via Bayesian analysis rather than deterministic concentration inequalities.

---

### Comparison to NeuralUCB  
The primary difference between NeuralTS and NeuralUCB lies in how they handle exploration:

- NeuralUCB defines an exploration bonus using an upper confidence bound:
  \[
  U_t(x) = f(x; \theta_t) + \alpha \sqrt{g(x; \theta_t)^\top \Sigma_t^{-1} g(x; \theta_t)}.
  \]
  This encourages the agent to pick arms with high predicted reward or high uncertainty.
  
- NeuralTS, instead of adding a fixed exploration bonus, samples a function \( \tilde{f}_t \) from the posterior:
  \[
  \tilde{f}_t(x) = f(x; \theta_t) + \epsilon_t^\top g(x; \theta_t),
  \]
  where \( \epsilon_t \sim \mathcal{N}(0, \Sigma_t^{-1}) \).
  Exploration thus emerges naturally from randomization rather than explicit optimism.

This means NeuralUCB explores deterministically by optimism in the face of uncertainty, while NeuralTS explores stochastically by Bayesian sampling. In practice, NeuralTS can yield more diverse exploration across similar contexts, whereas NeuralUCB can be more conservative but stable.

---

### Relevance to ViT-UCB  
NeuralTS provides a non-UCB based approach to tackling the contextual bandits problem. For a ViT-based contextual bandit, this approach could yield a ViT-TS variant that uses transformer gradients to generate posterior samples. However, this project would pursue the ViT-UCB approach instead for the following reasons
* Existing literature on using UCB to tackle image contexts (CNN-UCB), allowing for comparisons
* UCB's deterministic nature allows for more reproducible and interpretable results 

---
