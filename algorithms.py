"""Contains all the contextual bandit algorithms."""

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
from torchvision import transforms
from PIL import Image
from collections import deque

# -----------------
# Base Algorithm
# -----------------
class BanditAlgo:
    """
    Base class for all contextual bandit algorithms."""
    def __init__(self, env: ImageBanditEnv, detailed=True):
        """
        Initializes the BanditAlgo object.

        Parameters
        ----------
        env : ImageBanditEnv
            The environment object.
        detailed : bool, optional
            Whether to keep detailed records of the algorithm's progress. Defaults to True.
        """
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
        """
        Selects an arm according to the algorithm's policy.

        Returns
        -------
        int
            The index of the selected arm.
        """
        raise NotImplementedError

    def update(self, arm, bonus, preds):
        """
        Updates the algorithm's internal state after an arm is selected.

        Parameters
        ----------
        arm : int
            The index of the selected arm.
        bonus : float or numpy array
            The exploration bonus for the selected arm.
        preds : numpy array
            The predicted rewards for each arm.

        Returns
        -------
        float
            The reward obtained from the selected arm.
        """
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
# LinUCB (on ViT embeddings)
# -----------------
class LinUCB(BanditAlgo):
    """
    LinUCB algorithm for contextual bandits. Utilizes context vectors are passed through a ViT model.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    def __init__(self, env, alpha=1.0, model_name="WinKawaks/vit-tiny-patch16-224", detailed=True):
        
        """
        Initializes the LinUCB algorithm.

        Parameters
        ----------
        env : BanditEnvironment
            The environment the algorithm will interact with.
        alpha : float, optional
            The exploration-exploitation trade-off parameter. Defaults to 1.0.
        model_name : str, optional
            The name of the pre-trained ViT model to use. Defaults to "WinKawaks/vit-tiny-patch16-224".
        detailed : bool, optional
            Whether to keep detailed records of the algorithm's history. Defaults to True.

        Returns
        -------
        None
        """
        super().__init__(env, detailed)
        self.env = env
        self.alpha = alpha
        self.vit = ViTModel.from_pretrained(model_name)
        self.embed_dim = self.vit.config.hidden_size
        self.name = 'linucb'

        self.A = np.identity(self.embed_dim)
        self.b = np.zeros((self.embed_dim, 1))

    def get_embedding(self, img):
        """
        Compute the embedding of an image using the pre-trained ViT model.

        Parameters
        ----------
        img : numpy.ndarray
            The image to compute the embedding of.

        Returns
        -------
        emb : numpy.ndarray
            The embedding of the image.
        """
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
    """
    CNNUCB's CNN for reward prediction.
    """
    def __init__(self, input_channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(9216, 100)
        self.fc2 = nn.Linear(100, 1)
    def get_trainable_params(self):
        """Return parameters that are tracked in matrix A (every param except fc1)."""
        params = []
        for name, param in self.named_parameters():
            if "fc1" not in name and param.requires_grad:
                params.append(param)
        return params
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CNN_UCB(BanditAlgo):
    """
    CNN-UCB algorithm for contextual bandits.
    """
    transform = transforms.Compose([
        transforms.Resize((50, 50)),   # resize to model input
        transforms.ToTensor()
    ])
    def __init__(self, env, alpha=1.0, lambda_reg=1.0, model=CNNRewardNet, name="cnnucb", detailed=True):
        
        """
        Initialize the CNN-UCB algorithm.

        Parameters:
        - env: a ContextualBanditEnv object
        - alpha: the exploration hyperparameter
        - lambda_reg: the regularization hyperparameter
        - model: the reward model (default: CNNRewardNet)
        - name: the name of the algorithm (default: "cnnucb")
        - detailed: whether to log detailed information (default: True)

        Note that the reward model is moved to the CUDA device if available.
        """
        super().__init__(env, detailed)
        self.name = name
        self.device = torch.device("cuda")
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        self.env = env
        self.replay = ReplayBufferFIFO((3, 50, 50), 50)
        model = model()
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        # Store gradient vectors for low-rank covariance approximation
        self.grad_list = []
        self.m=10
        self.numel = sum(w.numel() for w in self.model.get_trainable_params() if w.requires_grad)
        # self.sigma_inv = lambda_reg * np.eye(self.numel, dtype=np.float32)
        self.sigma_inv = (self.lambda_reg * torch.eye(self.numel, dtype=torch.float32, device=self.device))
    
    def _preprocess_pil_list(self, contexts):
        """
        Preprocess a list of PIL images by applying resizing, ToTensor, and normalization.
        
        Parameters:
        - contexts: a list of PIL images to preprocess
        Returns:
        - batch: a torch tensor of shape (B, 3, H, W) containing the preprocessed images
        """
        tensors = []
        for ctx in contexts:
            x = self.transform(ctx)  # apply resize, ToTensor, normalize
            tensors.append(x)

        batch = torch.stack(tensors).to(self.device)
        return batch
    
    def select_arm(self):
        """Selects arm based on CNNUCB algorithm."""
        ucb_scores = []
        g = torch.zeros((self.env.n, self.numel), dtype=torch.float32, device=self.device)
        contexts = self.env.get_contexts()
        for i, img in enumerate(contexts):
            g[i] = self.grad(self.transform(img).cuda())
        with torch.no_grad():
            vals = torch.einsum('bi,ij,bj->b', g, self.sigma_inv, g)  # shape: (n_arms,)
            
            # compute bonuses
            bonus = self.alpha * torch.sqrt(torch.clamp(vals, min=0.0)).cpu().numpy().flatten()
            preds = self.model(self._preprocess_pil_list(contexts)).cpu().numpy().flatten()
            ucb_scores =  preds +bonus
        arm = int(np.argmax(ucb_scores))
        self.update(arm, bonus, contexts[arm], preds)
    def grad(self, x):
        """
        Compute the gradient of the model output with respect to the model parameters (only for models tracked in matrix A)

        Parameters:
        - x: a tensor of shape (B, 3, H, W) containing the input images

        Returns:
        - grad: a tensor of shape (self.numel,) containing the gradient of the model output with respect to the model parameters, divided by the square root of m.
        """
        if len(x.shape)<4:
            x = x.unsqueeze(0)

        y = self.model(x)
        self.optimizer.zero_grad()
        y.backward()
        return torch.cat(
                [w.grad.detach().flatten() / np.sqrt(self.m) for w in self.model.get_trainable_params() if w.requires_grad]
            ).to(self.device)
    def sherman_morrison_update(self, v):
        """Performs Sherman-Morrison update to update inverse"""
        self.sigma_inv -= (self.sigma_inv @ v @ v.T @ self.sigma_inv) / (1+v.T @ self.sigma_inv @ v)
    def update(self, arm, bonus, image, preds):
        reward = super().update(arm, bonus, preds)
        image = self.transform(image).to(self.device)
        # store gradient for UCB (faithful to paper)
        self.sherman_morrison_update(self.grad(image)[:, None])
        self.replay.add(image.cpu().numpy(), reward)
        self.train()
    def train(self):
        """Train model on past 50 rewards"""
        x, y = self.replay.sample(50)
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

class ViTRewardModel(nn.Module):
    """
    Reward model for ViT-UCB algorithm."""
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
        self.eval()
        with torch.no_grad():
            feats = self.vit(pixel_values=tensor_batch).logits  # (B, 1000)
            out = self.mlp_head(feats)  # (B, 1)

            return out.view(-1).cpu()
    def forward(self, tensor_batch):
        feats = self.vit(pixel_values=tensor_batch).logits  # (B, 1000)
        out = self.mlp_head(feats)  # (B, 1)
        return out.view(-1)

    def forward_scalar(self, x_tensor):
        self.train()
        feats = self.vit(pixel_values=x_tensor).logits  # (1, 1000)

        out = self.mlp_head(feats)  # (1, 1)
        return out.view(-1)

    def get_trainable_params(self, every=False):
        """Return parameters for UCB tracking (last block + MLP head)."""
        params = []
        # LoRA adapters from last block
        for name, p in self.vit.named_parameters():
            if "lora_" in name and ("encoder.layer.11" in name or every) and p.requires_grad:
                params.append(p)
        # Add MLP head params
        params += [p for p in self.mlp_head.parameters() if p.requires_grad]
        return params

class ReplayBufferFIFO:
    """FIFO replay buffer."""
    def __init__(self, d, capacity):
        self.buffer = {'context':deque(maxlen=capacity), 'reward': deque(maxlen=capacity),}
        # print(self.buffer['context'].shape)
        self.capacity = capacity


    def add(self, context, reward):
        self.buffer['context'].append(context)
        self.buffer['reward'].append(reward)

    def sample(self, n):
        return np.array(self.buffer['context']), np.array(self.buffer['reward']).flatten()


class ViT_UCB(BanditAlgo):
    """
    ViT-UCB algorithm.
    """
    def __init__(self, env, model_name="WinKawaks/vit-tiny-patch16-224", alpha=1.0, lambda_reg=1.0,
                 lora_r=8, lora_alpha=16, lora_dropout=0.0, device="cuda", detailed=True):
        """
        Initializes the ViT-UCB algorithm.

        Parameters
        ----------
        env : BanditEnvironment
            The environment the algorithm will interact with.
        model_name : str, optional
            The name of the pre-trained ViT model to use. Defaults to "WinKawaks/vit-tiny-patch16-224".
        alpha : float, optional
            The exploration-exploitation trade-off parameter. Defaults to 1.0.
        lambda_reg : float, optional
            The regularization strength for LoRA. Defaults to 1.0.
        lora_r : int, optional
            The rank of the LoRA approximation. Defaults to 8.
        lora_alpha : int, optional
            The alpha hyperparameter for LoRA. Defaults to 16.
        lora_dropout : float, optional
            The dropout rate used in LoRA. Defaults to 0.0.
        device : str, optional
            The device to use for computations. Defaults to "cuda" if available, otherwise "cpu".
        detailed : bool, optional
            Whether to keep detailed records of the algorithm's history. Defaults to True.

        Returns
        -------
        None
        """
        super().__init__(env, detailed)
        self.name = "vit_ucb"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        self.init_A()
        # number used for gradient normalization (NTK scaling)
        self.m = max(1, int(self.numel))  # fallback
        # cache device tensor for ones used in backward
        self._ones_cache = None
    def offload_A(self):
        self.sigma_inv.detach().cpu()
        del self.sigma_inv
        self.sigma_inv = None
    def init_A(self):
        self.sigma_inv = (self.lambda_reg * torch.eye(self.numel, dtype=torch.float32, device=self.device))
    def _grad_vector_for_tensor(self, x_tensor):
        """
        Compute flattened gradient vector (only for params tracked in matrix A)
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
        return grad_vector
        # return grad_vector.cpu().numpy().astype(np.float64)
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
        # return preds
        return preds.numpy().astype(np.float64)

    @torch.no_grad()
    def sherman_morrison_update(self, v: torch.Tensor):
        """
        Sherman–Morrison rank-1 inverse update.
        Args:
            v: column vector of shape (numel, 1), on same device as self.sigma_inv
        """
        if v.dim() == 1:
            v = v.view(-1, 1)
    
        # Compute scalar s = vᵀ Σ⁻¹ v
        s = (v.T @ self.sigma_inv @ v).item()
        denom = 1.0 + s
        if denom <= 1e-12:
            denom = 1e-12
    
        # Update inverse covariance
        self.sigma_inv -= (self.sigma_inv @ v @ v.T @ self.sigma_inv) / denom
    
        # Keep symmetric for numerical stability
        self.sigma_inv = 0.5 * (self.sigma_inv + self.sigma_inv.T)


    def select_arm(self):
        # get contexts (PIL images) from env
        """
        Select the arm to pull based on the ViTUCB algorithm.
        """

        contexts = self.env.get_contexts()  # should return list length n_arms
        # compute grad vectors for each arm (could be expensive; we do one backward per arm)
        grads = torch.zeros((self.n_arms, self.numel), dtype=torch.float32, device=self.device)
        for i, img in enumerate(contexts):
            # # preprocess single
            if isinstance(img, Image.Image):
                x = self.transform(img).unsqueeze(0).to(self.device)
            else:
                # assume tensor (C,H,W)
                x = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(self.device)
            grads[i] = self._grad_vector_for_tensor(x)

        bonuses = np.zeros((self.n_arms,), dtype=np.float64)
        vals = torch.einsum('bi,ij,bj->b', grads, self.sigma_inv, grads)  # shape: (n_arms,)
        
        # Ensure non-negative and compute bonuses
        bonuses = self.alpha * torch.sqrt(torch.clamp(vals, min=0.0))
        bonuses = bonuses.cpu().numpy()
        # compute predicted rewards in batch (no grad)
        preds = self._batch_preds(contexts)  # shape (n_arms,)
        ucb_scores = preds + bonuses
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

    def train(self, steps=1):
        """Train on replay buffer (reward prediction)."""
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



class ViT_UCB_Diag(BanditAlgo):
    """ViT UCB but only stores diagonal entries of A"""
    def __init__(self, env, model_name="google/vit-base-patch16-224", alpha=1.0, lambda_reg=1.0,
                 lora_r=8, lora_alpha=16, lora_dropout=0.0, device="cuda", detailed=True):
        super().__init__(env, detailed)
        self.name = "Diag"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        # optimizer only for trainable params (LoRA + classifier head)
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.vit.parameters()), lr=1e-4)
        # training hyperparams
        self.train_batch = 32
        self.train_every = 1  # train every 10 bandit steps
        self.update_steps_per_train = 1

        # get number of trainable params and initialize sigma_inv
        self.trainable_params = self.vit.get_trainable_params(True)
        self.numel = sum(p.numel() for p in self.trainable_params)

        # store ONLY diagonal of Σ^{-1} = λI
        self.sigma_inv_diag = (self.lambda_reg * torch.ones(self.numel, 
                                                            dtype=torch.float32,
                                                            device=self.device))

        self.m = max(1, int(self.numel))
        self._ones_cache = None
    def offload_A(self):
        self.sigma_inv.detach().cpu()
        del self.sigma_inv
        self.sigma_inv = None
    def init_A(self):
        self.sigma_inv_diag = (self.lambda_reg * torch.ones(self.numel, 
                                                                    dtype=torch.float32,
                                                                    device=self.device))
    def _grad_vector_for_tensor(self, x_tensor):
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
        return grad_vector
        # return grad_vector.cpu().numpy().astype(np.float64)
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
        # return preds
        return preds.numpy().astype(np.float64)

    @torch.no_grad()
    def sherman_morrison_update(self, v):
        """
        v: shape (numel, 1)
        Σ^{-1} is stored as a diagonal vector self.sigma_inv_diag.
        Update: d_i -= (d_i^2 * v_i^2) / (1 + sum(d_i * v_i^2))
        """
        if v.dim() == 2:
            v = v.view(-1)
    
        d = self.sigma_inv_diag          # (numel,)
        v2 = v * v                        # v_i^2
    
        # scalar s = v^T Σ^{-1} v
        s = torch.sum(d * v2).item()
        denom = 1.0 + max(s, 1e-12)
    
        # diag update
        self.sigma_inv_diag = d - (d * d * v2) / denom


    def select_arm(self):
        # get contexts (PIL images) from env
        contexts = self.env.get_contexts()  # should return list length n_arms
        # compute grad vectors for each arm (could be expensive; we do one backward per arm)
        grads = torch.zeros((self.n_arms, self.numel), dtype=torch.float32, device=self.device)
        for i, img in enumerate(contexts):
            # # preprocess single
            if isinstance(img, Image.Image):
                x = self.transform(img).unsqueeze(0).to(self.device)
            else:
                # assume tensor (C,H,W)
                x = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(self.device)
            grads[i] = self._grad_vector_for_tensor(x)

        bonuses = np.zeros((self.n_arms,), dtype=np.float64)
        vals = torch.sum((grads * grads) * self.sigma_inv_diag, dim=1) # shape: (n_arms,)
        
        # Ensure non-negative and compute bonuses
        bonuses = self.alpha * torch.sqrt(torch.clamp(vals, min=0.0))
        bonuses = bonuses.cpu().numpy()
        # compute predicted rewards in batch (no grad)
        preds = self._batch_preds(contexts)  # shape (n_arms,)
        ucb_scores = preds + bonuses
        chosen = int(np.argmax(ucb_scores))
        # call update with chosen arm and the bonus array (for logging)
        self.update(chosen, bonuses, contexts[chosen], preds)

    def update(self, arm, bonus, context_image, preds):
        reward = super().update(arm, bonus, preds)
        # compute gradient vector for this chosen context and update sigma_inv
        x=self.transform(context_image).unsqueeze(0).to(self.device)  # context_image

        grad_vec = self._grad_vector_for_tensor(x).view(-1)
        self.sherman_morrison_update(grad_vec)
        # add context + reward to replay
        # store as numpy array (C,H,W)
        img_np = x.squeeze(0).cpu().numpy()
        self.replay.add(img_np, reward)
        # training schedule
        if self.t % self.train_every == 0:
            self.train()

    def train(self, steps=1):
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
