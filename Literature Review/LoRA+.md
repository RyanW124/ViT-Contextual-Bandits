## LoRA+: Efficient Low-Rank Adaptation of Large Models  
**Reference:**  
Hayou, S., Ghosh, N., & Yu, B. (2024). *LoRA+: Efficient Low-Rank Adaptation of Large Models*. arXiv preprint arXiv:2402.12354.

---

### Goal and Motivation
Low-Rank Adaptation (LoRA) is a method for fine-tuning large pre-trained models by introducing low-rank matrices into each layer:

```math
W = W_0 + \Delta W, \quad \Delta W = A B
```

where $`W_0`$ is the frozen pre-trained weight, and $`A \in \mathbb{R}^{d \times r}`$, $`B \in \mathbb{R}^{r \times k}`$ are trainable low-rank matrices with rank $`r \ll \min(d,k)`$.  

The original LoRA uses the same learning rate for $`A`$ and $`B`$, which may not be optimal for very large models. LoRA+ addresses this by using separate learning rates:

```math
\eta_A = \eta, \quad \eta_B = \rho \cdot \eta
```

where $`\rho`$ is a fixed ratio chosen to improve convergence and feature adaptation.

---

### Core Idea
LoRA+ modifies the gradient update for the low-rank adapters:

```math
A \leftarrow A - \eta_A \nabla_A \mathcal{L}, \quad B \leftarrow B - \eta_B \nabla_B \mathcal{L}
```

This allows the model to adapt the low-rank subspace more effectively while keeping the original model weights $`W_0`$ frozen.

---

### Why This Matters
LoRA+ improves fine-tuning for large models by:
- Increasing adaptation efficiency in large-width layers.
- Reducing fine-tuning time.
- Maintaining or improving performance without increasing computational cost.

---

### Algorithmic Contributions
1. **Learning Rate Ratio:** Introduces a fixed ratio $`\rho`$ between the learning rates of $`A`$ and $`B`$, allowing asymmetric updates:

```math
\Delta W = A B, \quad \eta_B = \rho \eta_A
```

2. **Parameter Efficiency:** Only $`O(r(d+k))`$ parameters are trained instead of $`O(dk)`$, making the method memory and computation-efficient.

3. **Empirical Results:** Demonstrates 1–2% performance improvement and up to 2× faster fine-tuning compared to standard LoRA on multiple benchmarks.

---

### Strengths
- Efficient parameter adaptation for very large models.
- Improved convergence and performance compared to standard LoRA.
- No increase in computational overhead.

---

### Limitations
- Effectiveness depends on the hyperparameter $`\rho`$.
- Performance gains may vary across tasks or architectures.

---

### Relevance to ViT-UCB
LoRA+ can be applied to pre-trained Vision Transformers (ViTs) in contextual bandit settings. After each observation of reward, we can use LoRA+ to update the ViT
