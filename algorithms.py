import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from transformers import ViTModel
import random
from environment import ImageBanditEnv
from tqdm.auto import tqdm
from peft import LoraConfig, get_peft_model, TaskType
from transformers import ViTForImageClassification, AutoConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
# -----------------
# Base Algorithm
# -----------------
class BanditAlgo:
    def __init__(self, env: ImageBanditEnv, detailed=True):
        self.name=""
        self.env = env
        self.n_arms = env.n
        self.detailed = detailed
        self.counts = np.zeros(self.n_arms)
        self.rewards = np.zeros(self.n_arms)
        self.bonus_graph = []
        self.loss_graph = []
        self.history = []
        self.regret = 0
        self.t = 0
        self.count_graph = []
        self.cum_reward = 0
        self.regret_graph = [0]
        self.cum_reward_graph = [0]

    def select_arm(self):
        raise NotImplementedError

    def update(self, arm, bonus, preds):
        reward = self.env.step(arm)
        self.history.append((arm, reward))
        self.counts[arm] += 1
        self.t += 1
        if self.detailed:
            if len(bonus.shape)>1:
                bonus = bonus.flatten()
            self.bonus_graph.append(bonus)
            self.count_graph.append(self.counts.copy())

        self.rewards[arm] += reward
        self.cum_reward += reward
        self.cum_reward_graph.append(self.cum_reward)
        self.regret += self.env.best_mu - self.env.arms[arm].mu
        self.regret_graph.append(self.regret)
        self.loss_graph.append(F.mse_loss(torch.from_numpy(preds).view(-1), torch.tensor([arm.mu for arm in self.env.arms])).view(-1))
        return reward

    def run(self, n_steps=1):
        for _ in tqdm(range(n_steps), desc=f"Running {self.name}"):
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
            bonus = np.sqrt(2 * np.log(total_counts) / (self.counts+0.01))
        else:
            bonus = np.sqrt(2 * np.log(total_counts) / self.counts)
            ucb_vals = self.rewards / self.counts + bonus
            arm = np.argmax(ucb_vals)
        self.update(arm, bonus)


# -----------------
# Epsilon-Greedy
# -----------------
class EGreedy(BanditAlgo):
    def __init__(self, n_arms, epsilon=0.1):
        super().__init__(n_arms)
        self.epsilon = epsilon

    def select_arm(self, contexts=None):
        if random.random() < self.epsilon:
            return np.random.randint(self.n_arms)
        avg_rewards = self.rewards / np.maximum(self.counts, 1)
        return np.argmax(avg_rewards)


# -----------------
# LinUCB (on ViT embeddings)
# -----------------
class LinUCB(BanditAlgo):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    def __init__(self, env, alpha=1.0, model_name="WinKawaks/vit-tiny-patch16-224"):
        super().__init__(env)
        self.env = env
        self.alpha = alpha
        self.vit = ViTModel.from_pretrained(model_name)
        self.embed_dim = self.vit.config.hidden_size
        self.name = 'linucb'

        # Single universal A and b across all arms
        self.A = np.identity(self.embed_dim)
        self.b = np.zeros((self.embed_dim, 1))

    def get_embedding(self, img):
        with torch.no_grad():
            if len(img.shape) == 3:
                img = img.unsqueeze(0)
            outputs = self.vit(pixel_values=img)
            emb = outputs.last_hidden_state[:, 0, :]  # CLS token
        return emb.squeeze(0).numpy()

    def predict(self, contexts):
        contexts = [self.transform(i) for i in self.env.get_contexts()]
        A_inv = np.linalg.inv(self.A)
        theta = A_inv @ self.b

        preds = np.zeros(self.env.n)
        for a, img in enumerate(contexts):
            x = self.get_embedding(img).reshape(-1, 1)
            pred = float(theta.T @ x)
            preds[a] = pred
        return preds

    def select_arm(self):
        contexts = [self.transform(i) for i in self.env.get_contexts()]
        A_inv = np.linalg.inv(self.A)
        theta = A_inv @ self.b

        vals = np.zeros(self.env.n)
        preds = np.zeros(self.env.n)
        bonuses = np.zeros(self.env.n)

        for a, img in enumerate(contexts):
            x = self.get_embedding(img).reshape(-1, 1)
            pred = float(theta.T @ x)
            bonus = self.alpha * np.sqrt(x.T @ A_inv @ x)
            vals[a] = pred + bonus
            preds[a] = pred
            bonuses[a] = bonus

        arm = int(np.argmax(vals))
        self.update(arm, bonuses, preds, contexts[arm])
        return arm

    def update(self, arm, bonuses, preds, context):
        reward = super().update(arm, bonuses, preds)
        x = self.get_embedding(context).reshape(-1, 1)
        self.A += x @ x.T
        self.b += reward * x


# -----------------
# CNN-UCB
# -----------------




class CNNRewardNet(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(9216, 1)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class ReplayBuffer:

    def __init__(self, d, capacity):
        self.buffer = {'context':np.zeros((capacity, *d)), 'reward': np.zeros((capacity,1))}
        # print(self.buffer['context'].shape)
        self.capacity = capacity
        self.size = 0
        self.pointer = 0


    def add(self, context, reward):
        self.buffer['context'][self.pointer] = context
        self.buffer['reward'][self.pointer] = reward
        self.size = min(self.size+1, self.capacity)
        self.pointer = (self.pointer+1)%self.capacity

    def sample(self, n):
        idx = np.random.randint(0,self.size,size=min(self.size, n))
        return self.buffer['context'][idx], self.buffer['reward'][idx].flatten()

class CNN_UCB(BanditAlgo):
    transform = transforms.Compose([
        transforms.Resize((50, 50)),   # resize to model input
        transforms.ToTensor()
    ])
    def __init__(self, env, alpha=1.0, lambda_reg=1.0):
        super().__init__(env)
        self.name = "cnnucb"
        self.device = "cuda"
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        self.env = env
        self.replay = ReplayBufferFIFO((3, 50, 50), 10)
        model = CNNRewardNet()
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        # Store gradient vectors for low-rank covariance approximation
        self.grad_list = []
        self.m=10
        self.numel = sum(w.numel() for w in self.model.parameters() if w.requires_grad)
        self.sigma_inv = lambda_reg * np.eye(self.numel, dtype=np.float32)
    def _preprocess_pil_list(self, contexts):
        tensors = []
        for ctx in contexts:
            x = self.transform(ctx)  # apply resize, ToTensor, normalize
            tensors.append(x)

        batch = torch.stack(tensors).to(self.device)
        return batch
    def compute_grad_vector(self, x):
        """Compute flattened gradient vector w.r.t. all parameters"""
        out = self.model(x)
        self.optimizer.zero_grad()
        out.require_grad
        out.backward()
        grad_list = []
        for param in self.model.parameters():
            grad_list.append(param.grad.view(-1))
        grad_vector = torch.cat(grad_list).detach()  # flattened vector
        return grad_vector.cpu().numpy()
    
    def select_arm(self):
        ucb_scores = []
        g = np.zeros((self.env.n, self.numel), dtype=np.float32)
        contexts = self.env.get_contexts()
        for i, img in enumerate(contexts):
            g[i] = self.grad(self.transform(img).cuda()).cpu().numpy()
        with torch.no_grad():
            bonus =  self.alpha * np.sqrt(np.matmul(np.matmul(g[:, None, :], self.sigma_inv), g[:, :, None])[:, 0, :])
            preds = self.model(self._preprocess_pil_list(contexts)).cpu().numpy()
            ucb_scores =  preds +bonus
                # Compute exploration bonus using stored gradients
                # if len(self.grad_list) == 0:
                #     bonus = 0
                # else:
                #     G = np.stack(self.grad_list, axis=1)  # shape: param_dim x t
                #     print(G.shape[0])
                #     A = self.lambda_reg * np.eye(G.shape[0]) + G @ G.T
                #     grad_vec = self.compute_grad_vector(x)
                #     bonus = self.alpha * np.sqrt(grad_vec.T @ np.linalg.inv(A) @ grad_vec)
                
                # ucb_scores.append(pred + bonus)
        arm = int(np.argmax(ucb_scores))
        self.update(arm, bonus, contexts[arm], preds)
    def grad(self, x):
        if len(x.shape)<4:
            x = x.unsqueeze(0)

        y = self.model(x)
        self.optimizer.zero_grad()
        y.backward()
        return torch.cat(
                [w.grad.detach().flatten() / np.sqrt(self.m) for w in self.model.parameters() if w.requires_grad]
            ).to(self.device)
    def sherman_morrison_update(self, v):
        self.sigma_inv -= (self.sigma_inv @ v @ v.T @ self.sigma_inv) / (1+v.T @ self.sigma_inv @ v)
    def update(self, arm, bonus, image, preds):
        reward = super().update(arm, bonus, preds)
        image = self.transform(image).to(self.device)
        # pred = self.model(self.image_tensors[arm])
        # loss = F.mse_loss(pred, r)
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()
        
        # store gradient for UCB (faithful to paper)
        self.sherman_morrison_update(self.grad(image).cpu().numpy()[:, None])
        self.replay.add(image.cpu().numpy(), reward)
        self.train()
    def train(self):
        x, y = self.replay.sample(64)
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        y = torch.tensor(y, dtype=torch.float32).to(self.device)
        y_hat = self.model(x)
        loss = F.mse_loss(y_hat.view(-1), y.view(-1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
# -----------------
# ViT-UCB
# -----------------


# ViT reward network wrapper with LoRA
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoConfig, ViTForImageClassification
from peft import get_peft_model, LoraConfig, TaskType
from torchvision import transforms
from PIL import Image


class ViTRewardModel(nn.Module):
    def __init__(self, model_name="WinKawaks/vit-tiny-patch16-224", device="cuda",
                 lora_r=8, lora_alpha=16, lora_dropout=0.0):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Base config
        config = AutoConfig.from_pretrained(model_name)
        config.num_labels = 1000  # keep backbone output dim
        config.problem_type = "regression"

        # Load pretrained model
        self.vit = ViTForImageClassification.from_pretrained(
            model_name, config=config
        )

        # Inject LoRA
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q", "v", "k", "o", "query", "key", "value", "proj", "v_proj"],
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION
        )
        self.vit = get_peft_model(self.vit, lora_config)

        # Replace classifier with MLP head: 1000 -> 256 -> 1
        self.mlp_head = nn.Sequential(
            nn.Linear(1000, 1)
        )

        self.to(self.device)

    def predict(self, tensor_batch):
        """Forward in eval mode, return predictions (no grad)."""
        self.eval()
        with torch.no_grad():
            feats = self.vit(pixel_values=tensor_batch).logits  # (B, 1000)
            out = self.mlp_head(feats)  # (B, 1)

            return out.view(-1).cpu()
    def forward(self, tensor_batch):
        """Forward in eval mode, return predictions (no grad)."""
        feats = self.vit(pixel_values=tensor_batch).logits  # (B, 1000)
        out = self.mlp_head(feats)  # (B, 1)
        return out.view(-1)

    def forward_scalar(self, x_tensor):
        """Forward pass returning scalar tensor for gradient computation."""
        self.train()
        feats = self.vit(pixel_values=x_tensor).logits  # (1, 1000)

        out = self.mlp_head(feats)  # (1, 1)
        return out.view(-1)

    def get_trainable_params(self):
        """Return parameters for UCB tracking (last block + MLP head)."""
        params = []

        # Last transformer block if available
        # last_block = list(self.vit.vit.encoder.layer)[-1] if hasattr(self.vit, "vit") else None
        # if last_block is not None:
        #     params += [p for p in last_block.parameters() if p.requires_grad]

        # LoRA adapters from last block
        for name, p in self.vit.named_parameters():
            if "lora_" in name and "encoder.layer.11" in name and p.requires_grad:
                params.append(p)
        # for i in params:
        #     print(i.shape)
        # print()
        # Add MLP head params
        params += [p for p in self.mlp_head.parameters() if p.requires_grad]
        # print( [p for p in self.mlp_head.parameters() if p.requires_grad])
        return params


# ================== in ViT_UCB ==================
# replace this line in __init__:



# replace training section:
from collections import deque

class ReplayBufferFIFO:

    def __init__(self, d, capacity):
        self.buffer = {'context':deque(maxlen=capacity), 'reward': deque(maxlen=capacity),}
        # print(self.buffer['context'].shape)
        self.capacity = capacity


    def add(self, context, reward):
        self.buffer['context'].append(context)
        self.buffer['reward'].append(reward)

    def sample(self, n):
        return np.array(self.buffer['context']), np.array(self.buffer['reward']).flatten()

# ViT-UCB algorithm using LoRA fine-tuning
class ViT_UCB(BanditAlgo):
    def __init__(self, env, model_name="WinKawaks/vit-tiny-patch16-224", alpha=1.0, lambda_reg=1.0,
                 lora_r=8, lora_alpha=16, lora_dropout=0.0, device="cuda"):
        super().__init__(env)
        self.name = "vit_ucb"
        self.device = device if torch.cuda.is_available() else "cpu"
        self.alpha = float(alpha)
        self.lambda_reg = float(lambda_reg)
        self.env = env

        # preprocessing transform (ViT-Base expects 224x224)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
        ])

        # model with LoRA
        self.vit = ViTRewardModel(model_name=model_name, device=self.device,
                          lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        # replay buffer shape matches (C,H,W)
        self.replay = ReplayBufferFIFO((3, 224, 224), capacity=50)
        # self.replay = ReplayBuffer((3, 224, 224), capacity=1000)
        # optimizer only for trainable params (LoRA + classifier head)
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.vit.parameters()), lr=1e-4)
        # training hyperparams
        self.train_batch = 32
        self.train_every = 1  # train every 10 bandit steps
        self.update_steps_per_train = 1

        # get number of trainable params and initialize sigma_inv
        self.trainable_params = self.vit.get_trainable_params()
        self.numel = sum(p.numel() for p in self.trainable_params)
        # initialize inverse covariance as lambda * I (numpy)
        self.sigma_inv = (self.lambda_reg * np.eye(self.numel, dtype=np.float64))

        # number used for gradient normalization (NTK scaling)
        self.m = max(1, int(self.numel))  # fallback
        # cache device tensor for ones used in backward
        self._ones_cache = None
    
    def _grad_vector_for_tensor(self, x_tensor):
        """
        Compute flattened gradient vector (numpy) of the model output w.r.t. trainable params for a single input.
        x_tensor: torch.Tensor shape (1,C,H,W) on device
        returns: 1D numpy vector shape (numel,)
        """
        # zero grads
        self.optimizer.zero_grad()
        # forward
        out = self.vit.forward_scalar(x_tensor)  # returns 1-d tensor
        # ensure ones for backward shape
        if self._ones_cache is None or self._ones_cache.shape != out.shape:
            self._ones_cache = torch.ones_like(out).to(out.device)
        out.backward(self._ones_cache)  # accumulate grads into trainable params

        # collect grads
        grads = []
        for p in self.trainable_params:
            g = p.grad
            if g is None:
                grads.append(torch.zeros(p.numel(), device=self.device))
            else:
                grads.append(g.detach().view(-1))
        if len(grads) == 0:
            return np.zeros((self.numel,), dtype=np.float32)
        grad_vector = torch.cat(grads)
        # optional NTK scaling (paper uses 1/sqrt(m))
        grad_vector = grad_vector / np.sqrt(self.m)
        return grad_vector.cpu().numpy().astype(np.float64)
    def _preprocess_pil_list(self, contexts):
        tensors = []
        for ctx in contexts:
            x = self.transform(ctx)  # apply resize, ToTensor, normalize
            tensors.append(x)

        batch = torch.stack(tensors).to(self.device)
        return batch
    def _batch_preds(self, contexts):
        # contexts: list of PIL images or tensors
        batch = self._preprocess_pil_list(contexts)
        preds = self.vit.predict(batch).view(-1)  # returns torch tensor (B,)
        return preds.numpy().astype(np.float64)

    def sherman_morrison_update(self, v):
        # v: column vector shape (numel, 1) (numpy)
        v = v.reshape(-1, 1)
        s = float((v.T @ self.sigma_inv @ v).item())
        denom = 1.0 + s
        if denom <= 1e-12:
            denom = 1e-12
        self.sigma_inv = self.sigma_inv - (self.sigma_inv @ v @ v.T @ self.sigma_inv) / denom
        # keep symmetric
        self.sigma_inv = 0.5 * (self.sigma_inv + self.sigma_inv.T)

    def select_arm(self):
        # get contexts (PIL images) from env
        contexts = self.env.get_contexts()  # should return list length n_arms
        # compute grad vectors for each arm (could be expensive; we do one backward per arm)
        grads = np.zeros((self.n_arms, self.numel), dtype=np.float64)
        for i, img in enumerate(contexts):
            # # preprocess single
            if isinstance(img, Image.Image):
                x = self.transform(img).unsqueeze(0).to(self.device)
            else:
                # assume tensor (C,H,W)
                x = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(self.device)
            grads[i] = self._grad_vector_for_tensor(x)

        # compute bonuses: alpha * sqrt(g^T Sigma_inv g)
        bonuses = np.zeros((self.n_arms,), dtype=np.float64)
        for i in range(self.n_arms):
            gi = grads[i].reshape(-1, 1)
            val = float((gi.T @ self.sigma_inv @ gi).item())
            bonuses[i] = self.alpha * np.sqrt(max(val, 0.0))

        # compute predicted rewards in batch (no grad)
        preds = self._batch_preds(contexts)  # shape (n_arms,)
        ucb_scores = preds + bonuses
        # print(preds)
        # print(bonuses)
        # print(ucb_scores)
        # print()
        chosen = int(np.argmax(ucb_scores))
        # call update with chosen arm and the bonus array (for logging)
        self.update(chosen, bonuses, contexts[chosen], preds)

    def update(self, arm, bonus, context_image, preds):
        reward = super().update(arm, bonus, preds)
        # compute gradient vector for this chosen context and update sigma_inv
        x=self.transform(context_image).unsqueeze(0).to(self.device)  # context_image

        grad_vec = self._grad_vector_for_tensor(x)  # shape (numel,)
        self.sherman_morrison_update(grad_vec[:, None])
        # add context + reward to replay
        # store as numpy array (C,H,W)
        img_np = x.squeeze(0).cpu().numpy()
        self.replay.add(img_np, reward)
        # training schedule
        if self.t % self.train_every == 0:
            self.train()

    # def train(self, steps=1):
    #     # standard supervised regression on replay buffer to fit reward -> model(pixel_values)
    #     for _ in range(steps):
    #         x_batch, y_batch = self.replay.sample(self.train_batch)
    #         if x_batch is None:
    #             return
    #         x = torch.tensor(x_batch, dtype=torch.float32).to(self.device)
    #         y = torch.tensor(y_batch, dtype=torch.float32).to(self.device).view(-1, 1)
    #         self.vit.model.train()
    #         preds = self.vit.model(pixel_values=x).logits  # (B,1)
    #         loss = F.mse_loss(preds, y)
    #         self.optimizer.zero_grad()
    #         loss.backward()
    #         self.optimizer.step()
    def train(self, steps=1):
        """Train regression head on replay buffer (reward prediction)."""
        for _ in range(steps):
            x_batch, y_batch = self.replay.sample(self.train_batch)
            if x_batch is None:
                return
            x = torch.tensor(x_batch, dtype=torch.float32).to(self.device)
            y = torch.tensor(y_batch, dtype=torch.float32).to(self.device)

            self.vit.train()
            preds = self.vit(x)  # goes through vit + mlp_head
            # print(preds.shape, y.shape)
            loss = F.mse_loss(preds, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
