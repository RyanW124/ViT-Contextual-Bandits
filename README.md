# ViT Contextual Bandits

A research project exploring the integration of Vision Transformers (ViT) with Upper Confidence Bound (UCB) algorithms for contextual image-based bandit learning. The project finetunes a ViT using LoRA as rewards estimate and leverages neural tangent kernels as exploration bonus. In a nutshell, the UCB score is calculated as $$UCB_{t, a}=ViT(x_{t, a};\theta_t)+\alpha\sqrt{g(x_{t,a};\theta_t)^\top G_t^{-1} g(x_{t,a};\theta_t)}$$

## 📋 Overview

## 🚀 Project Timeline

| Date | Milestone | Description |
|------|------------|--------------|
| 2025-09-25 | **Project Proposal** | Defined research scope and completed preliminary literature review on contextual bandits and ViTs. Found [here](others/Project%20Proposal.pdf)|
| 2025-09-30 | **Proposal Presentation** | Presented project idea to class. Found [here](others/Proposal%20Presentation.pdf) |
| 2025-10-09 | **Literature Review** | Reviewed and annotated, in detail, literature that pertains to the project. Found in [Literature Review](Literature%20Review/)  |
| 2025-10-13 | **Environmemt Setup** | Code for fetching data and bandit environment |
| 2025-10-21 | **Algorithm Implementations** | Preliminary implementations of baselines (LinUCB, CNNUCB) and ViTUCB  |
| 2025-10-23 | **Proof of Concept** | Experiment on handwritten digit dataset, demonstrating that ViTUCB is viable. Found in [prelim results](prelim%20results/) |
| In Progress | **Actual Experiment** | Conduct experiment on TV show thumbails and ratings |