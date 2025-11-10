## 📖 CNN-UCB: Convolutional Neural Bandit for Visual-Aware Recommendation  
**Reference:**  
Yikun Ban, Jingrui He. (2021). *Convolutional Neural Bandit for Visual-Aware Recommendation*. arXiv preprint arXiv:2107.07438.

---

### Goal and Motivation
Classical contextual bandits like LinUCB assume a linear relationship between context features and rewards. While linear models are computationally efficient and theoretically well-understood, they fail to capture the complex, high-dimensional structure of visual contexts such as images.  

NeuralUCB extended linear bandits by using fully-connected neural networks to model non-linear reward functions and by deriving a gradient-based UCB exploration term via the Neural Tangent Kernel (NTK). CNN-UCB builds on this idea for image-based contexts leveraging Convolutional Neural Networks (CNNs) to model rewards in visual domains. This is particularly relevant for tasks such as movie thumbnail recommendation, online advertising, or any application where each arm is represented by an image.

---

### Core Idea
CNN-UCB can be understood as NeuralUCB specialized for images. It replaces the fully-connected network with a CNN to extract spatially-aware features from images while retaining the NTK-inspired exploration mechanism.

1. CNN Reward Estimation:  
   Each arm’s image context $`x_{t,a}`$ is processed through a CNN:

```math
\hat{r}_{t,a} = f(x_{t,a}; \theta_t) = \text{CNN}(x_{t,a}; \theta_t)
```

   where $`\theta_t`$ are the network parameters updated online.

2. Gradient-Based UCB Exploration:  
   Similar to NeuralUCB, the UCB for arm $`a`$ at time $`t`$ is:

```math
U_{t,a} = f(x_{t,a}; \theta_t) + \alpha \sqrt{g(x_{t,a};\theta_t)^\top G_t^{-1} g(x_{t,a};\theta_t)}
```

   with $`g(x_{t,a};\theta_t) = \nabla_\theta f(x_{t,a}; \theta_t)`$ and

```math
G_{t+1} = G_t + \sum_{s=1}^{t-1} g(x_{s,a_s}; \theta_s) g(x_{s,a_s}; \theta_s)^\top
```

   The exploration term captures the uncertainty of the model in unvisited regions of the context space, analogous to NeuralUCB but now applied to high-dimensional visual features.

3. Arm Selection:  
   At each round, select the arm with the highest UCB:

```math
a_t = \arg\max_a U_{t,a}
```

4. Parameter Update:  
   After observing reward $`r_t`$, update CNN parameters $`\theta_t`$ using gradient descent to minimize the squared error between predicted and observed rewards.

---

### CNN-UCB as the Image Version of NeuralUCB
CNN-UCB extends NeuralUCB to visual contexts:
- Fully-connected networks in NeuralUCB are replaced by CNNs, which are more effective for images because they exploit spatial structure and local correlations.  
- The gradient-based NTK term for exploration remains conceptually the same, allowing principled exploration in high-dimensional context spaces.  

Thus, CNN-UCB is NeuralUCB adapted for images: the architecture changes (CNN) to suit the domain, while the NTK-style UCB exploration mechanism remains intact. This also motivates ViT-UCB, where a Vision Transformer replaces the CNN for global feature extraction.

---

### Strengths
- Outperforms linear bandits and NeuralUCB on real-world image datasets.  
- Provides theoretical guarantees via NTK-based UCB.  
- Efficiently leverages spatial structure in images.

---

### Limitations
- Gradient-based UCB computation can be memory and compute intensive.  
- May require careful hyperparameter tuning for large-scale datasets.

---

### Relevance to ViT-UCB
- CNN-UCB demonstrates how NeuralUCB principles can be applied to high-dimensional image contexts.  
- For ViT-UCB, a pretrained Vision Transformer replaces the CNN to capture global dependencies, while a gradient-based UCB term maintains principled exploration.  
