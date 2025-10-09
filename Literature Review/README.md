# Literature Review

Includes detailed literature review of papers highly related to this project. It mainly consists of existing contextual bandits algorithms and ViT related methods/theories. See below for a short overview of each paper.

## Linear UCB

The Linear Upper Confidence Bound (LinUCB) algorithm models the reward as a linear function of context features:
\[
r_t = \theta^\top x_t + \epsilon_t
\]
and chooses actions according to an optimistic estimate:
\[
a_t = \arg\max_a \hat{\theta}_t^\top x_a + \alpha \sqrt{x_a^\top A_t^{-1} x_a}
\]
where \( A_t = \lambda I + \sum_{s<t} x_s x_s^\top \). The second term represents an uncertainty bonus derived from the inverse covariance matrix, balancing exploration and exploitation. 
LinUCB provides tight regret bounds of order \( O(\sqrt{Td\ln^3(KT\ln(T)/\delta)}) \) and serves as a foundation for later nonlinear extensions.

## Neural UCB

NeuralUCB extends LinUCB to nonlinear function approximation using neural networks. Instead of assuming a linear reward model, it defines:
\[
r_t = f(x_t; \theta^*) + \epsilon_t
\]
and maintains a confidence bound based on the neural tangent kernel (NTK) feature representation:
\[
a_t = \arg\max_a f(x_a; \theta_t) + \alpha \sqrt{g(x_a; \theta_t)^\top Z_t^{-1} g(x_a; \theta_t)}
\]
where \( g(x_a; \theta_t) = \nabla_\theta f(x_a; \theta_t) \) and \( Z_t \) is the cumulative outer product of gradients. The exploration term uses NTK geometry to measure how informative each sample is in parameter space. NeuralUCB generalizes the linear UCB confidence to neural networks, offering sublinear regret under mild smoothness assumptions. 

## CNN-UCB

CNN-UCB adapts NeuralUCB to image-based contextual bandits by using a convolutional neural network as the feature extractor. The architecture consists of two convolutional layers with 32 and 64 channels followed by two fully connected layers, outputting a scalar reward prediction. The exploration term follows the NeuralUCB formulation. Compared to NeuralUCB, CNN-UCB explicitly integrates spatial structure, leading to faster convergence on vision tasks. Conceptually, CNN-UCB can be viewed as an image-specific realization of NeuralUCB. However, CNNs remain limited in modeling long-range dependencies, motivating the shift to transformer-based architectures.

## Neural Thompson Sampling

Neural Thompson Sampling replaces UCB's deterministic confidence bonus with Bayesian posterior sampling. Instead of an explicit uncertainty term, it samples network parameters from an approximate posterior:
\[
\tilde{\theta}_t \sim \mathcal{N}(\hat{\theta}_t, \nu^2 Z_t^{-1})
\]
and selects:
\[
a_t = \arg\max_a f(x_a; \tilde{\theta}_t)
\]
This introduces stochastic exploration, theoretically equivalent to Thompson Sampling. NeuralTS faces practical issues in high-dimensional or non-convex neural settings, as posterior sampling becomes computationally expensive and unstable. Empirical comparisons show that NeuralTS variants often perform worse than deterministic UCB counterparts due to uncalibrated uncertainty estimates, reinforcing the motivation to prefer UCB-based exploration for architectures like ViT.

## LoRA and LoRA+

LoRA introduces a low-rank decomposition of weight updates during fine-tuning:
\[
W' = W + BA
\]
where \( A \in \mathbb{R}^{r \times k} \), \( B \in \mathbb{R}^{k \times r} \), and \( r \ll k \). LoRA+ improves the original formulation by learning adaptive scaling factors and integrating rank selection based on layer sensitivity. For ViT-UCB, LoRA+ provides an efficient way to adapt large pretrained ViTs online without retraining the full model, allowing the bandit algorithm to fine-tune exploration behavior in real-time while keeping the parameter count tractable.

## Neural Tangent Kernel

The Neural Tangent Kernel describes how the output of a neural network changes linearly with small perturbations in parameters around initialization:
\[
f(x; \theta) \approx f(x; \theta_0) + \nabla_\theta f(x; \theta_0)^\top (\theta - \theta_0)
\]
and defines the kernel:
\[
K(x, x') = \nabla_\theta f(x; \theta_0)^\top \nabla_\theta f(x'; \theta_0)
\]
In NeuralUCB and CNN-UCB, this kernel structure provides a way to quantify uncertainty geometrically, informing how to scale or approximate the UCB bonus for transformer. This method should also be transferable to ViTs.

## Vision Transformer and ViT-UCB

The Vision Transformer (ViT) applies the transformer architecture to images by splitting them into patches, embedding them, and processing them via self-attention layers. ViTs capture global contextual relationships rather than local features, outperforming CNNs on many vision benchmarks. Integrating ViTs into UCB-based bandits potentially provides richer, globally coherent reward estimates.