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
    def __init__(self, mu, sigma, images):
        self.mu = mu
        self.sigma = sigma
        self.images = images
        self.count = 0
    def context(self):
        return random.choice(self.images)
        # idx = random.randint(0, self.images.shape[0]-1)
        # return self.images[idx]
    def pull(self):
        self.count += 1
        return np.random.normal(self.mu, self.sigma)
def rgba_to_rgb(image):
    if image.mode == "RGBA":
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))  # white
        image = Image.alpha_composite(background, image)
        image = image.convert("RGB")
    else:
        image = image.convert("RGB")
    return image
class ImageBanditEnv2:
    def __init__(self, data_file, n, sigma, temp='temp'):
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
        for i in tqdm(data, desc="Loading images"):
            try: response = requests.get(j['images']['jpg']['small_image_url'])
            except: continue
            if response.status_code == 200:
                im_path = self.image_path.joinpath(f"{i['mal_id']}.jpg")
                if im_path.is_file():
                    image = Image.open(im_path)
                else:
                    image = Image.open(BytesIO(response.content))
                    image.save(im_path)
                self.data.append((image, i['score']))
class ImageBanditEnv:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),   # resize to model input
        transforms.ToTensor()
    ])
    @classmethod
    def from_mnist(cls, path, sigma):
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
    def __init__(self, data_file, n, sigma, temp='temp'):
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
        for i in tqdm(data, desc="Loading images"):
            try: response = requests.get(j['images']['jpg']['small_image_url'])
            except: continue
            if response.status_code == 200:
                im_path = self.image_path.joinpath(f"{i['mal_id']}.jpg")
                if im_path.is_file():
                    image = Image.open(im_path)
                else:
                    image = Image.open(BytesIO(response.content))
                    image.save(im_path)
                self.data.append((image, i['score']))
    def step(self, i):
        return self.arms[i]

    def get_contexts(self):
        indices = random.sample(range(len(self.data)), self.n)
        self.arms = [self.data[i][1] for i in indices]
        return [self.data[i][0] for i in indices]
    
class Experiment:
    def __init__(self, env, agents, results="results/"):
        self.results = Path(results)
        self.results.mkdir(exist_ok=True)
        self.agents = agents
        self.env = env

    def generate_results(self, smooth=0):
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
        for i in self.agents:
            i.run(n_steps)