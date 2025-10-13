import numpy as np
import json
from PIL import Image
import requests, random, os
from pathlib import Path
from matplotlib import pyplot as plt

class Arm:
    def __init__(self, mu, sigma, img_path):
        self.mu = mu
        self.sigma = sigma
        self.img_path = img_path
        self.context = Image.open(img_path)
        self.count = 0
    def pull(self):
        self.count += 1
        return np.random.normal(self.mu, self.sigma)
class ImageBanditEnv:
    def __init__(self, data_file, n, sigma, temp="temp/"):
        with open(data_file, 'r') as f:
            data = json.load(f)
        self.n = n
        self.temp = Path(temp)
        self.temp.mkdir(exist_ok=True)
        arms_raw = random.sample(data, n)

        self.arms: list[Arm] = []
        self.best_mu = 0
        
        for i, arm in enumerate(arms_raw):
            mu = arm['score']
            self.best_mu = max(mu, self.best_mu)
            response = requests.get(arm['images']['jpg']['image_url'], stream=True)
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
            img_path = self.temp.joinpath(f'arm_{i}.jpg')
            with open(img_path, 'wb') as f:
                f.write(response.content)
                # for chunk in response.iter_content(chunk_size=8192):
                #     f.write(chunk)
            self.arms.append(Arm(mu, sigma, img_path))
    def step(self, i):
        reward = self.arms[i].pull()
        return reward

    def get_contexts(self):
        return [img for img, _ in self.arms]
    
class Experiment:
    def __init__(self, env, agents, results="results/"):
        self.results = Path(results)
        self.results.mkdir(exist_ok=True)
        self.agents = agents
        self.env = env

    def generate_results(self):
        for agent in self.agents:
            plt.plot(agent.cum_reward_graph, label=agent.name)
        plt.legend()
        plt.savefig(self.results.joinpath("Cum_Reward.png"))
        plt.close()
        for agent in self.agents:
            plt.plot(agent.regret_graph, label=agent.name)
        plt.legend()
        plt.savefig(self.results.joinpath("Regret.png"))
        plt.close()

    def run(self, n_steps):
        for i in self.agents:
            i.run(n_steps)