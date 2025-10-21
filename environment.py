import numpy as np
import json
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
        for agent in self.agents:
            plt.plot(agent.loss_graph, label=agent.name)
        plt.legend()
        plt.savefig(self.results.joinpath("Loss.png"))
        plt.close()

        
        for agent in self.agents:
            if not agent.detailed:
                continue
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            counts = np.array(agent.count_graph).T
            for i, arr in enumerate(counts):
                axes[0].plot(arr, label=f'Arm {i}')
            axes[0].set_title('# of Times Chosen')
            axes[0].legend()

            # Right plot
            bonus = np.array(agent.bonus_graph).T
            for i, arr in enumerate(bonus):
                axes[1].plot(arr, label=f'Arm {i}')
            axes[1].set_title('UCB Bonus')
            axes[1].legend()
            plt.tight_layout()
            plt.savefig(self.results.joinpath(f"{agent.name}_detail.png"))
            plt.close()


    def run(self, n_steps):
        for i in self.agents:
            i.run(n_steps)