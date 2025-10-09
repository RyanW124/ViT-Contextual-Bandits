## 📖 CNN-UCB: Convolutional Neural Bandit for Visual-Aware Recommendation  
**Reference:**  
Yikun Ban, Jingrui He. (2021). *Convolutional Neural Bandit for Visual-Aware Recommendation*. arXiv preprint arXiv:2107.07438.

---

### Goal and Motivation
Classical contextual bandit algorithms, such as LinUCB, assume a linear relationship between context features and expected rewards. While these algorithms provide theoretical guarantees and computational efficiency, they fail to capture the complex, high-dimensional structure of visual contexts.  

CNN-UCB addresses this limitation by modeling the expected reward as a **non-linear function of image features** extracted by a Convolutional Neural Network (CNN). This enables the agent to make more informed decisions in scenarios like visual recommendation systems, online advertising, or movie thumbnail selection, where the context for each arm is an image.  

---

### Core Idea
CNN-UCB combines a CNN-based feature extractor with an Upper Confidence Bound (UCB) exploration mechanism:

1. **CNN for Reward Estimation:**  
   Each arm’s context \( x_{t,a} \) is an image processed through a CNN \( f(x_{t,a}; \theta_t) \). The network outputs a predicted reward for the arm:

   \[
   \hat{r}_{t,a} = f(x_{t,a}; \theta_t)
   \]

   Here, \( \theta_t \) are the CNN parameters updated online.

2. **Upper Confidence Bound (UCB) Exploration:**  
   The exploration bonus is derived from the neural tangent kernel (NTK) approximation, similar to NeuralUCB. The UCB for arm \( a \) at time \( t \) is:

   \[
   U_{t,a} = f(x_{t,a}; \theta_t) + \alpha \sqrt{g(x_{t,a};\theta_t)^\top G_t^{-1} g(x_{t,a};\theta_t)}
   \]

   where \( g(x_{t,a};\theta_t) = \nabla_\theta f(x_{t,a};\theta_t) \) is the gradient of the network output w.r.t parameters, \( G_t = \sum_{s=1}^{t-1} g(x_{s,a_s};\theta_s) g(x_{s,a_s};\theta_s)^\top + \lambda I \) accumulates past gradients, and \( \alpha \) controls the exploration-exploitation trade-off.

3. **Arm Selection:**  
   At each round, the algorithm selects the arm with the highest UCB:

   \[
   a_t = \arg\max_a U_{t,a}
   \]

4. **Parameter Update:**  
   After observing the reward \( r_t \), the CNN parameters are updated using stochastic gradient descent to minimize the squared error between predicted and observed reward.

---

### Why This Matters
CNN-UCB bridges **linear contextual bandits (LinUCB)** and **non-linear neural bandits (NeuralUCB)** for **visual contexts**:

- Linear methods fail on high-dimensional image inputs.
- NeuralUCB provides a principled exploration term via NTK but was not specifically applied to convolutional architectures.
- CNN-UCB leverages the **structured feature extraction of CNNs** while preserving NTK-style UCB exploration, making it a practical and theoretically grounded method for image-based bandits.

By explicitly modeling uncertainty via the gradient covariance \( G_t \), CNN-UCB can efficiently explore untried arms while exploiting promising visual contexts.

---

### Algorithmic Contributions
1. **CNN-Based Reward Function:** Captures complex, non-linear relationships between images and rewards.  
2. **Gradient-Based UCB Exploration:** Uses the NTK-inspired term to balance exploration and exploitation, extending NeuralUCB to convolutional architectures.  
3. **Regret Analysis:** Provides a near-optimal regret bound \( \tilde{O}(\sqrt{T}) \) under over-parameterized CNN assumptions, offering theoretical guarantees for non-linear visual bandits.  
4. **Connection to CNTK:** The authors show that the dynamic convolutional NTK behaves similarly to the initialization kernel during training, validating the exploration bonus.

---

### Strengths
- **Empirical Performance:** Outperforms traditional linear bandits and heuristic exploration strategies on real-world visual datasets.  
- **Theoretical Guarantees:** Provides provable regret bounds, extending NeuralUCB theory to CNNs.  
- **Adaptability:** Can handle high-dimensional visual contexts and non-linear reward functions.  

---

### Limitations
- **Over-Parameterization Assumption:** The theoretical guarantees assume CNNs are sufficiently wide, which may not hold for smaller networks.  
- **Computational Complexity:** Maintaining \( G_t \) and computing gradient-based UCBs can be memory- and compute-intensive.  
- **Scalability:** Online updates for very deep CNNs or large datasets may require approximations or batching strategies.  

---

### Relevance to ViT-UCB
CNN-UCB provides a blueprint for extending **NeuralUCB to high-dimensional, non-linear contexts**. For ViT-UCB:  

- Replace CNN with a **Vision Transformer (ViT)** to capture global image features via self-attention.  
- Use gradient-based NTK terms for the UCB exploration bonus, similar to NeuralUCB/CNN-UCB.  
- Pretrained ViTs can be frozen or adapted with LoRA to improve sample efficiency while maintaining exploration guarantees.  
- The theoretical connection to NTK justifies using gradients of the ViT embeddings in computing exploration bonuses.

---

This version now ties the CNN-UCB method explicitly to LinUCB, NeuralUCB, NTK theory, and your eventual ViT-UCB project, maintaining the style of your previous literature reviews.
