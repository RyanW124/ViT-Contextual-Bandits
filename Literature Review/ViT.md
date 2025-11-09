##  An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale  
**Reference:**  
Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby. (2020). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*. arXiv:2010.11929.

---

### Goal and Motivation
While CNNs have historically dominated computer vision, they are limited by local receptive fields and fixed inductive biases, which can hinder global reasoning about images. Vision Transformers (ViTs) overcome these limitations by processing images as sequences of patches, allowing self-attention mechanisms to model long-range dependencies across the entire image.  

For contextual bandits where each arm has a high-dimensional image context, global understanding of the image is crucial. For example, in a movie recommendation task, ViTs can capture the overall layout, color composition, and object interactions in a thumbnail, rather than just local textures that CNNs typically emphasize.

---

### Core Idea
ViT represents an image as a sequence of non-overlapping patches:

$$
x \in \mathbb{R}^{H \times W \times C} \rightarrow \{x_p^1, x_p^2, \dots, x_p^N\}, \quad x_p^i \in \mathbb{R}^{P^2 \cdot C}
$$

Each patch is flattened and linearly projected into an embedding. A standard Transformer encoder is applied to the sequence:

$$
z_0 = [x_{class}; x_p^1 E; x_p^2 E; \dots; x_p^N E] + E_{pos}
$$

$$
z_\ell' = \text{MSA}(\text{LN}(z_{\ell-1})) + z_{\ell-1}, \quad
z_\ell = \text{MLP}(\text{LN}(z_\ell')) + z_\ell'
$$

where $MSA$ is multi-head self-attention, $E$ is the linear projection, and $E_{pos}$ adds positional embeddings.

This structure allows global attention across all patches, capturing dependencies that CNNs may miss due to local convolutions.

---

### CNN vs. ViT Advantages for ViT-UCB
1. **Global Context:** Self-attention considers all patches simultaneously, which is crucial for high-level visual features in contextual bandits.  
2. **Adaptive Feature Weighting:** Attention dynamically assigns importance to patches relevant for predicting rewards.  
3. **Flexibility:** ViTs can scale to higher resolutions without the exponential growth in parameters required for deep CNNs.  

In contrast, CNNs rely heavily on local kernels and pooling, which may ignore subtle but globally relevant patterns in image contexts.

---

### Relevance to ViT-UCB
- **Representation Learning:** The ViT provides a powerful estimate of the reward, providing the exploitation term
- **LoRA / Parameter-Efficient Fine-Tuning:** ViTs can be adapted efficiently with LoRA to the bandit task, without losing pretrained knowledge.  

---

### Strengths
- Global feature extraction through self-attention.  
- Superior transfer learning capabilities from large pretrained datasets.  
- Efficient handling of high-resolution images compared to CNNs.  

---

### Limitations
- Requires pretraining on large datasets for optimal performance.  
- Training Transformers from scratch is computationally expensive.  
