##  Neural Tangent Kernel: Convergence and Generalization in Neural Networks  
**Reference:**  
Jacot, A., Gabriel, F., & Hongler, C. (2018). *Neural Tangent Kernel: Convergence and Generalization in Neural Networks.* Advances in Neural Information Processing Systems, 31.

---

### Goal and Motivation
The Neural Tangent Kernel (NTK) paper addresses the fundamental question: why do neural networks trained with gradient descent generalize well despite their non-convex and highly over-parameterized nature? The authors provide a rigorous framework to analyze the training dynamics of deep neural networks in the infinite-width limit. This framework allows networks to be studied as kernel machines, connecting neural network training to classical function-space analysis and opening the door for provable convergence guarantees.

---

### Core Idea
Consider a neural network $`f(x; \theta)`$ with parameters $`\theta`$. During gradient descent, the function update can be expressed as:

```math
\frac{df(x)}{dt} = -\eta \nabla_\theta f(x) \cdot \nabla_\theta \mathcal{L}(\theta)
```

where $`\eta`$ is the learning rate and $`\mathcal{L}`$ is the loss function. By linearizing the network around its initialization $`\theta_0`$ and taking the infinite-width limit, the dynamics reduce to a linear system governed by the Neural Tangent Kernel:

```math
\Theta(x, x') = \nabla_\theta f(x; \theta_0)^\top \nabla_\theta f(x'; \theta_0)
```

Here, $`\Theta`$ is the NTK, and it remains approximately constant during training. The network’s function evolution then becomes a kernel regression problem in function space.


### Relevance to ViT-UCB
NTK theory provides the mathematical foundation for treating networks as a locally linear function. This allows the UCB-style exploration bonus to operate in a theoretically grounded function space, motivating approaches like NeuralUCB or CNN-UCB.

