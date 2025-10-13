import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from transformers import ViTModel
import random
from environment import ImageBanditEnv


# -----------------
# Base Algorithm
# -----------------
class BanditAlgo:
    def __init__(self, env: ImageBanditEnv):
        self.name=""
        self.env = env
        self.n_arms = env.n
        self.counts = np.zeros(self.n_arms)
        self.rewards = np.zeros(self.n_arms)
        self.history = []
        self.regret = 0
        self.t = 0
        self.cum_reward = 0
        self.regret_graph = [0]
        self.cum_reward_graph = [0]

    def select_arm(self):
        raise NotImplementedError

    def update(self, arm):
        reward = self.env.step(arm)

        self.history.append((arm, reward))
        self.counts[arm] += 1
        self.t += 1

        self.rewards[arm] += reward
        self.cum_reward += reward
        self.cum_reward_graph.append(self.cum_reward)
        self.regret += self.env.best_mu - self.env.arms[arm].mu
        self.regret_graph.append(self.regret)

    def run(self, n_steps=1):
        for _ in range(n_steps):
            self.select_arm()

    


# -----------------
# UCB1 (non-contextual)
# -----------------
class UCB1(BanditAlgo):
    def __init__(self, env):
        super().__init__(env)
        self.name="UCB1"
    def select_arm(self):
        total_counts = np.sum(self.counts)
        if 0 in self.counts:
            arm = np.argmin(self.counts)
        else:
            ucb_vals = self.rewards / self.counts + np.sqrt(2 * np.log(total_counts) / self.counts)
            arm = np.argmax(ucb_vals)
        self.update(arm)

