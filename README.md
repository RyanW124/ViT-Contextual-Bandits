# ViT Contextual Bandits

## 📋 Overview

A research project exploring the integration of Vision Transformers (ViT) with Upper Confidence Bound (UCB) algorithms for contextual image-based bandit learning. The project finetunes a ViT using LoRA as rewards estimate and leverages neural tangent kernels as exploration bonus. In a nutshell, the UCB score is calculated as 
```math
UCB_{t, a}=f_{\text{ViT}}({x}_{t, a};{\theta}_{\text{LoRA}}^t)+\alpha\sqrt{{g}_{\text{ViT}}({x}_{t,a};{\theta}_{\text{LoRA}}^t)^\top A_t^{-1} {g}_{\text{ViT}}({x}_{t,a};{\theta}_{\text{LoRA}}^t)}
```
where $`{A}_t = \lambda {I} + \sum_{i=1}^t \frac{{g}_{\text{ViT}}({x}_i; {\theta}_{\text{LoRA}}^i) {g}_{\text{ViT}}({x}_i; {\theta}_{\text{LoRA}}^i)^\top}{d_{\text{LoRA}}}`$ and $`{g}_{\text{ViT}}({x}; {\theta}_{\text{LoRA}}) = \nabla_{{\theta}_{\text{LoRA}}} f_{\text{ViT}}({x}; {\theta}_{\text{ViT}}, {\theta}_{\text{LoRA}})`$.

## 🚀 Project Timeline

| Date | Milestone | Description |
|------|------------|--------------|
| 2025-09-25 | **Project Proposal** | Defined research scope and completed preliminary literature review on contextual bandits and ViTs. Found [here](others/Project%20Proposal.pdf)|
| 2025-09-30 | **Proposal Presentation** | Presented project idea to class. Slides found [here](others/Proposal%20Presentation.pdf) |
| 2025-10-09 | **Literature Review** | Reviewed and annotated, in detail, literature that pertains to the project. Found in [Literature Review](Literature%20Review/)  |
| 2025-10-13 | **Environmemt Setup** | Code for fetching data and bandit environment |
| 2025-10-21 | **Algorithm Implementations** | Preliminary implementations of baselines (LinUCB, CNNUCB) and ViTUCB  |
| 2025-10-23 | **Proof of Concept** | Experiment on handwritten digit dataset, demonstrating that ViTUCB is viable. Results found in [prelim results](prelim%20results/) |
| 2025-10-28 | **Rough Proof of UCB Bound** | Came up with rough proof that the difference between true reward and reward estimate is bounded by the exploration bonus: $`\left\lvert f^*({x}) - f_{\mathrm{ViT}}({x}; {\theta}_{\mathrm{ViT}}, {\theta}_{\mathrm{LoRA}}) \right\rvert \le \alpha \,\left\lVert \frac{{g}_{\mathrm{ViT}}({x}; {\theta}_{\mathrm{LoRA}})}{\sqrt{d_{\mathrm{LoRA}}}} \right\rVert_{{A}_t^{-1}} + \beta`$ |
| 2025-11-02 | **Hyperparameter Tuning** | Tune hyperparameters of algorithms, using [hyperparameter tune.ipynb](hyperparameter%20tune.ipynb) |
| 2025-11-05 | **Experiment 1** | Experiment on handwritten digit dataset, with tuned hyperparameters. Results found in [mnist results](mnist%20results/) |
| 2025-11-08 | **Experiment 2** | Experiment on anime dataset, with tuned hyperparameters. Results found in [anime results](anime%20results/) |
| 2025-11-11 | **Midterm Presentation** | Presented project idea to class. Slides found [here](others/Midterm%20Presentation.pdf) |
| In Progress | **Paper First Draft** | Write paper detailing results and findings |
