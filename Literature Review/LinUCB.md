## LinUCB: Contextual Bandits with Linear Payoff Functions  
**Reference:**  
Chu, W., Li, L., Reyzin, L., & Schapire, R. (2011). *Contextual Bandits with Linear Payoff Functions.* Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics (AISTATS), PMLR, pp. 208–214.  

---

### **Goal and Motivation**  
LinUCB was developed to extend the classical multi-armed bandit (MAB) problem to scenarios where context is available at each decision step. Traditional MAB algorithms assume a stationary reward distribution for each arm, but in many real-world problems—like personalized news recommendation—the reward depends on both the user’s context and the chosen action. The LinUCB algorithm formalizes this setting by assuming that the expected reward is a linear function of the context features.  

---

### **Core Idea**  
At each round $ t $, for each arm $ a $, LinUCB assumes:  

$$
\mathbb{E}[r_{t,a}] = \theta^{*\top} x_{t,a}
$$

where $ x_{t,a} \in \mathbb{R}^d $ is the context vector and $ \theta^* $ is an unknown parameter vector.  
The algorithm maintains an estimate $ \hat{\theta}_t $ of $ \theta^* $ using ridge regression and computes an Upper Confidence Bound (UCB) for each arm:

$$
U_{t,a} = \hat{\theta}_t^\top x_{t,a} + \alpha \sqrt{x_{t,a}^\top A_t^{-1} x_{t,a}}
$$

where  

$$
A_t = \lambda I + \sum_{s=1}^{t-1} x_{s,a_s} x_{s,a_s}^\top
$$

is the regularized covariance matrix.  

The first term ($ \hat{\theta}_t^\top x_{t,a} $) represents exploitation (the estimated mean reward) while the second term ($ \alpha \sqrt{x_{t,a}^\top A_t^{-1} x_{t,a}} $) represents exploration, encouraging the algorithm to try arms with higher uncertainty.

---

### **Why the Exploration Term Works**  
The term $ \sqrt{x_{t,a}^\top A_t^{-1} x_{t,a}} $ measures the uncertainty in the model’s prediction for context $ x_{t,a} $.  
Intuitively:
- The matrix $ A_t $ encodes how much data the algorithm has seen for each direction in feature space.  
- Its inverse, $ A_t^{-1} $, represents uncertainty or **variance in parameter estimates: large values in a direction mean that region of feature space is underexplored.  
- Multiplying $ A_t^{-1} $ by $ x_{t,a} $ propagates this uncertainty to the predicted reward of the current arm.

Thus, arms whose features lie in poorly explored regions will receive a higher exploration bonus, prompting the algorithm to try them more often.  
Over time, as more data is gathered, $ A_t $ grows, $ A_t^{-1} $ shrinks, and the confidence intervals tighten—naturally decaying exploration as knowledge improves.

This probabilistic justification can be viewed as deriving from a confidence ellipsoid around $ \hat{\theta}_t $:
$$
P(|x_{t,a}^\top (\hat{\theta}_t - \theta^*)| \le \alpha \sqrt{x_{t,a}^\top A_t^{-1} x_{t,a}}) \ge 1 - \delta
$$
which guarantees that the true expected reward lies within the UCB with high probability $ (1-\delta) $.

---

### **Algorithm Summary**
1. Initialize $ A_0 = \lambda I $, $ b_0 = 0 $.  
2. For each round $ t $:  
   - For each arm $ a $, compute $ \hat{\theta}_t = A_t^{-1} b_t $.  
   - Compute $ U_{t,a} $ as above.  
   - Select the arm $ a_t = \arg\max_a U_{t,a} $.  
   - Observe reward $ r_t $ and update:  
     $$
     A_{t+1} = A_t + x_{t,a_t} x_{t,a_t}^\top, \quad b_{t+1} = b_t + r_t x_{t,a_t}
     $$

---

### **Key Contributions**
- **Context awareness:** Introduces a principled way to incorporate contextual information into bandit decision-making.  
- **Theoretical guarantees:** Achieves a regret bound of $ O(\sqrt{Td\ln^3(KT\ln(T)/\delta)}) $, which scales logarithmically with time and linearly with feature dimension.

---

### **Strengths**
- Simple, interpretable, and computationally efficient.  
- Strong theoretical foundation for exploration-exploitation trade-off.  
- Serves as the standard benchmark for contextual bandit research.  

---

### **Limitations**
- **Linearity assumption:** The model assumes a linear relationship between context and expected reward, which often fails in real-world applications or complex domains (e.g., images or text).   
- **Not expressive enough for non-linear structures** like those captured by neural networks or transformers.  

---

### **Relevance to ViT-UCB**
LinUCB provides the theoretical and algorithmic backbone for modern contextual bandit algorithms. In ViT-UCB, the same UCB principle is retained, but the context vector $ x_{t,a} $ is replaced with an image and instead of maintaining a $\hat \theta $ as estimate, it uses a ViT.