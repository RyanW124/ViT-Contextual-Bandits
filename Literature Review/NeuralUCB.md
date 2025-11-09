## NeuralUCB: Neural Contextual Bandits with UCB Exploration  
**Reference:**  
Zhou, D., Li, L., Li, J., Tang, L., & Gu, Q. (2020). *Neural Contextual Bandits with UCB-based Exploration.* Proceedings of the 37th International Conference on Machine Learning (ICML), PMLR 119:11492–11503.

---

### Goal and Motivation
NeuralUCB extends the LinUCB algorithm to settings where the expected reward cannot be modeled as a linear function of the context. In many real-world applications, such as image-based recommendation or personalized feedback, the reward depends on complex, nonlinear relationships that linear models cannot capture. NeuralUCB replaces the linear assumption with a neural network function approximator, while retaining LinUCB’s principled upper confidence bound (UCB) exploration strategy.

The main challenge is to design an exploration bonus that remains theoretically grounded when the model is nonlinear. NeuralUCB addresses this using a connection between neural networks and kernel methods via the Neural Tangent Kernel (NTK) theory.

---

### Core Idea
Let $ f(x; \theta) $ denote a neural network parameterized by weights $ \theta $. The goal is to learn a mapping from context $ x_{t,a} $ to expected reward $ \mathbb{E}[r_{t,a}] \approx f(x_{t,a}; \theta^*) $, where $ \theta^* $ represents the true (unknown) parameters.

At each round $ t $, NeuralUCB maintains a parameter estimate $ \hat{\theta}_t $ obtained by minimizing the squared loss on past observations. For each candidate arm $ a $, the algorithm computes an upper confidence bound:

$$
U_{t,a} = f(x_{t,a}; \hat{\theta}_t) + \alpha \sqrt{g_t(x_{t,a})^\top Z_t^{-1} g_t(x_{t,a})}
$$

where $ Z_t $ represents a regularized approximation of the local curvature of the network’s output with respect to its parameters. This term captures how uncertain the model is about the output for the given context.

---

### Why the Exploration Term Works
In linear models, uncertainty is directly tied to the covariance matrix of the feature vectors. In NeuralUCB, the feature space is implicitly defined by the gradients of the network output with respect to its parameters. Specifically, the gradient

$$
g_t(x) = \nabla_\theta f(x; \hat{\theta}_t)
$$

serves as a nonlinear feature representation of the context. The uncertainty in the prediction for arm $ a $ can then be expressed as

$$
\sqrt{g_t(x_{t,a})^\top Z_t^{-1} g_t(x_{t,a})}
$$

This measures how sensitive the model’s prediction is to small parameter changes. If the gradient has high magnitude in directions where the model is poorly constrained, it implies high uncertainty. The exploration bonus therefore encourages sampling arms with high model uncertainty, ensuring that the algorithm explores novel or underrepresented contexts.

Intuitively, NeuralUCB treats the neural network as a locally linear function around its current parameters. The uncertainty term estimates how far the current function might deviate from the true reward in that local neighborhood.

---

### The Neural Tangent Kernel (NTK) Approximation
The NTK provides a way to relate a neural network’s predictions to an equivalent kernel regression model. Under certain assumptions (wide networks, small learning rates, and smooth activations), the evolution of the neural network during training can be approximated by a kernel method with kernel

$$
K(x, x') = \nabla_\theta f(x; \theta_0)^\top \nabla_\theta f(x'; \theta_0)
$$

where $ \theta_0 $ is the network’s initialization.

NeuralUCB leverages this idea by treating the gradient vectors $ g_t(x) $ as implicit features governed by the NTK. The confidence term based on $ Z_t^{-1} $ therefore acts as an estimate of the uncertainty in the kernel regression approximation. This connection allows the use of UCB principles in nonlinear neural models while retaining theoretical regret guarantees.

---

### Algorithm Summary
1. Initialize network parameters $ \theta_0 $ and regularization constant $ \lambda > 0 $.  
2. Initialize matrix $ Z_0 = \lambda I $.  
3. For each round $ t = 1, 2, \ldots, T $:  
   - Compute the current gradient features $ g_t(x_{t,a}) = \nabla_\theta f(x_{t,a}; \hat{\theta}_t) $ for all arms $ a $.  
   - For each arm, calculate  
     $$
     U_{t,a} = f(x_{t,a}; \hat{\theta}_t) + \alpha \sqrt{g_t(x_{t,a})^\top Z_t^{-1} g_t(x_{t,a})}
     $$
   - Select $ a_t = \arg\max_a U_{t,a} $.  
   - Observe reward $ r_t $.  
   - Update the matrix and network parameters:  
     $$
     Z_{t+1} = Z_t + g_t(x_{t,a_t}) g_t(x_{t,a_t})^\top
     $$
     $$
     \hat{\theta}_{t+1} = \text{TrainNN}(\hat{\theta}_t, \{x_{s,a_s}, r_s\}_{s=1}^t)
     $$
     
---

### Key Contributions
- Extends LinUCB to nonlinear models by integrating deep neural networks with UCB-based exploration.  
- Introduces a theoretically justified confidence term derived from NTK analysis.  
- Provides regret bounds of order $ \tilde{O}(d \sqrt{T}) $ under suitable regularity conditions.  

---

### Strengths
- Captures nonlinear reward structures beyond linear models.  
- Maintains a clear exploration-exploitation balance through UCB principles.  
- Retains theoretical regret guarantees under mild smoothness assumptions.  

---

### Limitations
- Requires gradient computation for each arm at every step, increasing computational cost.  
- Theoretical assumptions rely on wide-network approximations, which may not hold in practice.  

---

### Relevance to ViT-UCB
NeuralUCB provides the conceptual bridge between classical linear contextual bandits and non-linear contextual bandits. It shows that even when there is no clear relation between the context and reward, agent is still able to utilize the context effectively
