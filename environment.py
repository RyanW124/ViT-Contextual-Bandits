"""Contains environment for image contextual bandit problems and experiment class for running experiments."""

import numpy as np
import json
from scipy.ndimage import gaussian_filter1d
from PIL import Image, ImageOps
import requests, random, os
from pathlib import Path
from matplotlib import pyplot as plt
from torchvision import transforms
import pickle
from io import BytesIO
import requests, torch
from tqdm.auto import tqdm

class Arm:
    """Arm for image contextual bandit problem used in ImageBanditEnv. (experiments 1 and 2 in paper)"""
    def __init__(self, mu, sigma, images):
        
        """
        Initializes an Arm object.

        Parameters
        ----------
        mu : float
            The mean of the normal distribution.
        sigma : float
            The standard deviation of the normal distribution.
        images : numpy.ndarray
            The images associated with this arm.

        Returns
        -------
        None
        """

        self.mu = mu
        self.sigma = sigma
        self.images = images
        self.count = 0
    def context(self):
        """Samples image"""
        return random.choice(self.images)
    def pull(self):
        """Samples reward"""
        self.count += 1
        return np.random.normal(self.mu, self.sigma)
def rgba_to_rgb(image):
    """Converts an RGBA image to RGB."""
    if image.mode == "RGBA":
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))  # white
        image = Image.alpha_composite(background, image)
        image = image.convert("RGB")
    else:
        image = image.convert("RGB")
    return image
class ImageBanditEnv:
    """Environment for image contextual bandit problems used in experiments 1 and 2 in the paper. (Each arm has a fixed reward distribution)"""
    @classmethod
    def from_mnist(cls, path, sigma):
        """
        Creates an ImageBanditEnv object for experiment 1
        
        Parameters
        ----------
        path : str
            The file containing the images.
        sigma : float
            The standard deviation of the normal distribution.
        """
        ret = cls(None, 10, sigma)
        ret.arms = []
        bar = tqdm(total=6000, desc="Loading images")
        for i in range(10):
            ipath = path.joinpath(str(i), str(i))
            images = []
            for image in os.listdir(ipath)[:600]:
                image = Image.open(ipath.joinpath(image))
                images.append(rgba_to_rgb(image))
                # image.close()
                bar.update(1)
            ret.arms.append(Arm(i, sigma, images))
        ret.best_mu = 9
        ret.n = 10
        return ret
            
            
    def __init__(self, data_file, n, sigma, temp='temp'):
        
        """
        Initializes an ImageBanditEnv object for experiment 2.

        Parameters
        ----------
        data_file : str
            The file containing the JSON data.
        n : int
            The number of arms.
        sigma : float
            The standard deviation of the normal distribution.
        temp : str, optional
            The temporary directory to store the images. Defaults to 'temp'.
        
        Returns
        -------
        None
        """
        if data_file is None:
            return
        data = []
        with open(data_file, 'r') as f:
            for line in f:
                line = json.loads(line)
                if line['score']:
                    data.append(line)
        self.n = n
        self.image_path = Path(temp)
        self.image_path.mkdir(exist_ok=True)
        data.sort(key=lambda x: x['score'])
        size=len(data)//n
        chunks = []
        mus = []
        bar = tqdm(total=size*n, desc="Loading images")
        for i in range(n):
            chunk = data[i*size:(i+1)*size]
            ratings = []
            images = []
            for j in chunk:
                try: response = requests.get(j['images']['jpg']['small_image_url'])
                except: continue
                if response.status_code == 200:
                    im_path = self.image_path.joinpath(f"{j['mal_id']}.jpg")
                    ratings.append(j['score'])
                    if im_path.is_file():
                        image = Image.open(im_path)
                    else:
                        image = Image.open(BytesIO(response.content))
                        image.save(im_path)
                    images.append(image)
                bar.update(1)
            # chunks.append(torch.stack(images))
            chunks.append(images)
            mus.append(sum(ratings)/len(ratings))
        # self.temp = Path(temp)
        # self.temp.mkdir(exist_ok=True)
        # arms_raw = random.sample(data, n)

        self.arms: list[Arm] = []
        self.best_mu = 0
        for i, (images, mu) in enumerate(zip(chunks, mus)):
            self.best_mu = max(mu, self.best_mu)
            self.arms.append(Arm(mu, sigma, images))
    def step(self, i):
        reward = self.arms[i].pull()
        return reward

    def get_contexts(self):
        return [arm.context() for arm in self.arms]
    def save(self, file='env.pkl'):
        with open(file, 'wb') as f:
            pickle.dump(self, f)
    @classmethod
    def load(self, file='env.pkl'):
        with open(file, 'rb') as f:
            return pickle.load(f)
        
class ImageBanditEnv2(ImageBanditEnv):
    """Environment for esperiment 3 in paper"""
    def __init__(self, data_file, n, sigma, temp='temp'):
        """
        Initializes an ImageBanditEnv2 object.

        Parameters
        ----------
        data_file : str
            The file containing the JSON data.
        n : int
            The number of arms.
        sigma : float
            The standard deviation of the normal distribution.
        temp : str, optional
            The temporary directory to store the images. Defaults to 'temp'.

        Returns
        -------
        None
        """
        if data_file is None:
            return
        data = []
        with open(data_file, 'r') as f:
            for line in f:
                line = json.loads(line)
                if line['score']:
                    data.append(line)
        self.n = n
        self.image_path = Path(temp)
        self.image_path.mkdir(exist_ok=True)
        self.data = []
        self.arms = [Arm(0, sigma, []) for _ in range(self.n)]
        for i in tqdm(data, desc="Loading images"):
            try: response = requests.get(i['images']['jpg']['small_image_url'])
            except: continue
            im_path = self.image_path.joinpath(f"{i['mal_id']}.jpg")
            if im_path.is_file():
                image = Image.open(im_path)
            else:
                if response.status_code == 200:
                    image = Image.open(BytesIO(response.content))
                    image.save(im_path)
                else: continue
            self.data.append((image, i['score']))
    def step(self, i):
        return self.arms[i].pull()

    def get_contexts(self):
        indices = random.sample(range(len(self.data)), self.n)
        self.best_mu = 0
        for i, j in enumerate(indices):
            self.arms[i].mu=  self.data[j][1]
            self.best_mu = max(self.best_mu, self.data[j][1])
        return [self.data[i][0] for i in indices]
    
class Experiment:
    """Class for running experiments"""
    def __init__(self, env, agents, results="results/"):
        """
        Initializes an Experiment object.

        Parameters
        ----------
        env : ImageBanditEnv or ImageBanditEnv2
            The environment to run the experiment on.
        agents : list
            A list of agents to run the experiment with.
        results : str, optional
            The directory to store the results. Defaults to 'results/'.
        """

        self.results = Path(results)
        self.results.mkdir(exist_ok=True)
        self.agents = agents
        self.env = env

    def generate_results(self, smooth=0):
        """
        Generates plots for the experiment results.

        Parameters
        ----------
        smooth : int, optional
            The sigma value for the Gaussian filter to smooth the plots. Defaults to 0.

        Generates the following plots:

        - Cumulative RewardOver Time
        - Cumulative RegretOver Time
        - Loss Over Time
        - Detailed plot of the agent's history, including the # of times chosen, UCB bonus, and UCB bonus 0-centered.

        Saves all plots to the directory specified in the Experiment initialization.

        Returns
        -------
        None
        """
        self.results.mkdir(exist_ok=True)
        for agent in self.agents:
            plt.plot(agent.cum_reward_graph, label=agent.name)
        plt.legend()
        plt.title("Cumulative Reward Over Time")
        plt.ylabel("Cumulative Reward")
        plt.xlabel("Time Step")
        plt.savefig(self.results.joinpath("Cumulative_Reward.png"))
        plt.close()
        for agent in self.agents:
            plt.plot(agent.regret_graph, label=agent.name)
        plt.legend()
        plt.title("Cumulative Regret Over Time")
        plt.ylabel("Cumulative Regret")
        plt.xlabel("Time Step")
        plt.savefig(self.results.joinpath("Cumulative_Regret.png"))
        plt.close()
        for agent in self.agents:
            plt.plot(agent.loss_graph, label=agent.name)
        plt.legend()
        plt.title("Loss Over Time")
        plt.ylabel("MSE")
        plt.xlabel("Time Step")
        plt.savefig(self.results.joinpath("Loss.png"))
        plt.close()

        
        for agent in self.agents:
            if not agent.detailed:
                continue
            fig, axes = plt.subplots(1, 3, figsize=(10, 4))
            counts = np.array(agent.count_graph).T
            for i, arr in enumerate(counts):
                axes[0].plot(arr, label=f'Arm {i}')
            axes[0].set_title('# of Times Chosen')
            axes[0].legend()

            # Right plot
            bonus = np.array(agent.bonus_graph)
            n = self.env.n
            normed = (bonus-np.repeat((np.sum(bonus, axis=1)/n)[:, None], n, axis=1)).T
            
            bonus = bonus.T
            for i, arr in enumerate(bonus):
                p = gaussian_filter1d(arr, sigma=smooth) if smooth else arr
                axes[1].plot(p, label=f'Arm {i}')
                arr2 = normed[i]
                p =gaussian_filter1d(arr2, sigma=smooth) if smooth else arr
                axes[2].plot(p, label=f'Arm {i}')
                
            axes[1].set_title('UCB Bonus')
            axes[1].legend()
            axes[2].set_title('UCB Bonus 0-Centered')
            axes[2].legend()
            fig.suptitle(f"{agent.name} Details")
            plt.tight_layout()
            plt.savefig(self.results.joinpath(f"{agent.name}_detail.png"))
            plt.close()


    def run(self, n_steps):
        """
        Runs the experiment for n_steps for each agent.

        Parameters
        ----------
        n_steps : int
            The number of time steps to run the experiment for.

        Returns
        -------
        None
        """
        for i in self.agents:
            i.run(n_steps)