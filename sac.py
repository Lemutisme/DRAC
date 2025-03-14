from utils import evaluate_policy_SAC as evaluate_policy
from utils import Action_adapter_symm as Action_adapter
from utils import Action_adapter_symm_reverse as Action_adapter_reverse
from utils import build_net, Reward_adapter
from environment_modifiers import register

import copy
import math
import hydra
import logging
import platform

import random
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from tqdm import tqdm
from pathlib import Path
from scipy.special import logsumexp
from scipy.optimize import minimize_scalar
from omegaconf import DictConfig, OmegaConf
from abc import ABC, abstractmethod

######################################################
# NOTE: 
# Actor, Double_Q_Critic, V_Critic: MLP
# Transition learner: VAE
# Three robust dual optimization options:
# 1. beta, minimize loss.mean()
# 2. functional g, minimize scalar loss, equivalent to replace beta with g(s,a)
# 3. independent optimize, too slow
######################################################

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hid_shape, hid_layers, hidden_activation=nn.ReLU, output_activation=nn.ReLU):
        super(Actor, self).__init__()
        layers = [state_dim] + list(hid_shape) * hid_layers

        self.a_net = build_net(layers, hidden_activation, output_activation)
        self.mu_layer = nn.Linear(layers[-1], action_dim)
        self.log_std_layer = nn.Linear(layers[-1], action_dim)

        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20

    def forward(self, state, deterministic, with_logprob):
        '''Network with Enforcing Action Bounds'''
        net_out = self.a_net(state)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, min=self.LOG_STD_MIN, max=self.LOG_STD_MAX)  
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        u = mu if deterministic else dist.rsample()

        '''↓↓↓ Enforcing Action Bounds, see Page 16 of https://arxiv.org/pdf/1812.05905.pdf ↓↓↓'''
        a = torch.tanh(u)
        if with_logprob:
            # Get probability density of logp_pi_a from probability density of u:
            # logp_pi_a = (dist.log_prob(u) - torch.log(1 - a.pow(2) + 1e-6)).sum(dim=1, keepdim=True)
            # Derive from the above equation. No a, thus no tanh(h), thus less gradient vanish and more stable.
            logp_pi_a = dist.log_prob(u).sum(axis=1, keepdim=True) - (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(axis=1, keepdim=True)
        else:
            logp_pi_a = None

        return a, logp_pi_a

class V_Critic(nn.Module):
    def __init__(self, state_dim, hid_shape, hid_layers):
        super(V_Critic, self).__init__()
        self.state_dim = state_dim

        layers = [state_dim] + list(hid_shape) * hid_layers + [1]
        self.V = build_net(layers, nn.ReLU, nn.Identity)

    def forward(self, state):
        output = self.V(state)
        return output

class Double_Q_Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hid_shape, hid_layers):
        super(Double_Q_Critic, self).__init__()
        layers = [state_dim + action_dim] + list(hid_shape) * hid_layers + [1]

        self.Q_1 = build_net(layers, nn.ReLU, nn.Identity)
        self.Q_2 = build_net(layers, nn.ReLU, nn.Identity)   

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        q1 = self.Q_1(sa)
        q2 = self.Q_2(sa)            
        return q1, q2

class TransitionVAE(nn.Module):
    def __init__(self, state_dim, action_dim, out_dim, hidden_dim=64, hidden_layers=1, latent_dim=5):
        super(TransitionVAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder layers
        e_layers = [state_dim * 2 + action_dim] + list(hidden_dim) * hidden_layers
        self.encoder = build_net(e_layers, nn.ReLU, nn.Identity)
        self.e_mu = nn.Linear(e_layers[-1], latent_dim)
        self.e_logvar = nn.Linear(e_layers[-1], latent_dim)

        # Decoder layers
        d_layers = [state_dim + action_dim + latent_dim] + list(hidden_dim) * hidden_layers + [out_dim]
        self.decoder = build_net(d_layers, nn.ReLU, nn.Identity)

    def encode(self, s, a, s_next):
        x = torch.cat([s, a, s_next], dim=1)
        h = self.encoder(x)
        mu = self.e_mu(h)
        logvar = self.e_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, s, a, z):
        x = torch.cat([s, a, z], dim=-1)
        s_next_recon = self.decoder(x)
        return s_next_recon

    def forward(self, s, a, s_next):
        mu, logvar = self.encode(s, a, s_next)
        z = self.reparameterize(mu, logvar)
        s_next_recon = self.decode(s, a, z)
        return s_next_recon, mu, logvar  

    def sample(self, s, a, num_samples):
        batch_size = s.size(0)
        # Sample latent vectors from the prior with shape (batch, num_samples, latent_dim)
        z = torch.randn(batch_size, num_samples, self.latent_dim, device=s.device)
        # Expand s and a along a new sample dimension so that their shapes become (batch, num_samples, feature_dim)
        s_expanded = s.unsqueeze(1).expand(-1, num_samples, -1)
        a_expanded = a.unsqueeze(1).expand(-1, num_samples, -1)
        s_next_samples = self.decode(s_expanded, a_expanded, z)
        return s_next_samples   

class ExpActivation(nn.Module):
    def forward(self, x):
        return torch.exp(x)   

class dual(nn.Module):
    def __init__(self, state_dim, action_dim, hid_shape, hid_layers):
        super(dual, self).__init__()  
        layers = [state_dim + action_dim] + list(hid_shape) * hid_layers + [1]

        self.G = build_net(layers, nn.ReLU, ExpActivation)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)          
        return self.G(sa)

class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size, device):
        self.max_size = max_size
        self.device = device
        self.ptr = 0
        self.size = 0

        self.s = torch.zeros((max_size, state_dim), dtype=torch.float, device=self.device)
        self.a = torch.zeros((max_size, action_dim), dtype=torch.float, device=self.device)
        self.r = torch.zeros((max_size, 1), dtype=torch.float, device=self.device)
        self.s_next = torch.zeros((max_size, state_dim), dtype=torch.float, device=self.device)
        self.dw = torch.zeros((max_size, 1), dtype=torch.bool, device=self.device)

    def add(self, s, a, r, s_next, dw):
        self.s[self.ptr] = torch.from_numpy(s).to(self.device)
        self.a[self.ptr] = torch.from_numpy(a).to(self.device) # Note that a is numpy.array
        self.r[self.ptr] = r
        self.s_next[self.ptr] = torch.from_numpy(s_next).to(self.device)
        self.dw[self.ptr] = dw

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = torch.randint(0, self.size, device=self.device, size=(batch_size,))
        return self.s[ind], self.a[ind], self.r[ind], self.s_next[ind], self.dw[ind]


class SAC_continuous():
    def __init__(self, **kwargs):
        # Init hyperparameters for agent, just like "self.gamma = opt.gamma, self.lambd = opt.lambd, ..."
        self.__dict__.update(kwargs)
        self.tau = 0.005

        self.actor = Actor(self.state_dim, self.action_dim, hid_shape=(self.net_width, self.net_width), hid_layers=self.net_layer).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.a_lr)

        self.v_critic = V_Critic(self.state_dim, hid_shape=(self.net_width, self.net_width), hid_layers=self.net_layer).to(self.device)
        self.v_critic_optimizer = torch.optim.Adam(self.v_critic.parameters(), lr=self.c_lr)
        self.v_critic_target = copy.deepcopy(self.v_critic)
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.v_critic_target.parameters():
            p.requires_grad = False

        self.q_critic = Double_Q_Critic(self.state_dim, self.action_dim, 
                                        hid_shape=(self.net_width, self.net_width), 
                                        hid_layers=self.net_layer).to(self.device)
        self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=self.c_lr)

        if self.robust:    
            print('This is a robust policy.')
            self.transition = TransitionVAE(self.state_dim, self.action_dim, self.state_dim, 
                                            hidden_dim=(self.net_width, self.net_width), 
                                            hidden_layers=self.net_layer, 
                                            latent_dim=5).to(self.device)
            self.trans_optimizer = torch.optim.Adam(self.transition.parameters(), lr=self.r_lr)

            self.log_beta = nn.Parameter(torch.ones((self.batch_size,1), requires_grad=True, device=self.device) * 1.0)
            self.beta_optimizer = torch.optim.Adam([self.log_beta], lr=self.b_lr)

            self.g = dual(self.state_dim, self.action_dim, (self.net_width, self.net_width), self.net_layer).to(self.device)
            self.g_optimizer = torch.optim.Adam(self.g.parameters(), lr=self.g_lr)

        self.replay_buffer = ReplayBuffer(self.state_dim, self.action_dim, max_size=int(1e6), device=self.device)

        if self.adaptive_alpha:
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            self.target_entropy = torch.tensor(-self.action_dim, dtype=float, requires_grad=True, device=self.device)
            # We learn log_alpha instead of alpha to ensure alpha>0
            self.log_alpha = torch.tensor(np.log(self.alpha), dtype=float, requires_grad=True, device=self.device)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.c_lr)

    def select_action(self, state, deterministic):
        # only used when interact with the env
        with torch.no_grad():
            state = torch.FloatTensor(state[np.newaxis,:]).to(self.device)
            a, _ = self.actor(state, deterministic, with_logprob=False)
        return a.cpu().numpy()[0]

    def vae_loss(self, s_next, s_next_recon, mu, logvar):
        recon_loss = F.mse_loss(s_next_recon, s_next, reduction='sum')
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_div

    def dual_func_g(self, s, a, s_next):
        size = s_next.shape[1]
        dual_sa = self.g(s,a)
        return - dual_sa * (torch.logsumexp(-self.v_critic_target(s_next).squeeze(-1)/dual_sa, dim=1, keepdim=True) - math.log(size)) - dual_sa * self.delta  

    def dual_func_beta(self, s_next, beta):
        # Jointly optimize, in tensor
        size = s_next.shape[1]
        return - beta * (torch.logsumexp(-self.v_critic_target(s_next).squeeze(-1)/beta, dim=1, keepdim=True) - math.log(size)) - beta * self.delta     

    def dual_func_ind(self, s_next, beta):
        # Independently optimize, in np.array
        size = s_next.shape[-1]
        v_next = self.v_critic_target(s_next)
        v_next = v_next.cpu().numpy()
        return - beta * (logsumexp(-v_next/beta) - math.log(size)) - beta * self.delta           

    def train(self, writer, step):
        s, a, r, s_next, dw = self.replay_buffer.sample(self.batch_size)

        #----------------------------- ↓↓↓↓↓ Update R Net ↓↓↓↓↓ ------------------------------#
        if self.robust:
            s_next_recon, mu, logvar = self.transition(s, a, s_next)
            tr_loss = self.vae_loss(s_next, s_next_recon, mu, logvar)
            self.trans_optimizer.zero_grad()
            tr_loss.backward()
            self.trans_optimizer.step()
            if self.debug_print:
                print(f"tr_loss: {tr_loss.item()}")
            if writer:
                writer.add_scalar('tr_loss', tr_loss, global_step=step)

            with torch.no_grad():
                s_next_sample = self.transition.sample(s, a, 200)

            #############################################################		
            ### option1: optimize w.r.t beta ###
            # self.log_beta = nn.Parameter(torch.ones_like(self.log_beta, requires_grad=True, device=self.device) * 0.1)
            # self.beta_optimizer = torch.optim.Adam([self.log_beta], lr=self.b_lr)
            if self.robust_optimizer == 'beta':
                for _ in range(5):
                    self.beta = torch.exp(self.log_beta)
                    opt_loss = -self.dual_func_beta(s_next_sample, self.beta)
                    self.beta_optimizer.zero_grad()
                    opt_loss.mean().backward()
                    if self.debug_print:
                        print(opt_loss.sum().item())
                    self.beta_optimizer.step() 

                V_next_opt = self.dual_func_beta(s_next_sample, torch.exp(self.log_beta)) 
            #############################################################		

            #############################################################		
            ### option2: optimize w.r.t functional g ###
            elif self.robust_optimizer == 'functional':
                for _ in range(5):
                    opt_loss = -self.dual_func_g(s, a, s_next_sample)
                    self.g_optimizer.zero_grad()
                    opt_loss.mean().backward()
                    self.g_optimizer.step() 
                    if self.debug_print:
                        print(opt_loss.mean().item())

                V_next_opt = self.dual_func_g(s, a, s_next_sample) 
            #############################################################		

            #############################################################		
            # option3: Use scipy.optimize to separately optimize
            elif self.robust_optimizer == 'separate':
                s_next_sample = s_next_sample.cpu().numpy()
                V_next_opt_acc = np.zeros((self.batch_size, 1))
                for i in range(s_next_sample.shape[0]):
                    opt = minimize_scalar(fun=lambda beta:-self.dual_func_ind(s_next_sample[i], beta), method='Bounded', bounds=(1e-4, 1.0))
                    V_next_opt_acc[i] = -opt.fun
                V_next_opt_acc = torch.from_numpy(V_next_opt_acc).float()
                V_next_opt_acc = V_next_opt_acc.to('cuda' if torch.cuda.is_available() else 'cpu')
            ############################################################		

            else:
                raise NotImplementedError

        #----------------------------- ↓↓↓↓↓ Update Q Net ↓↓↓↓↓ ------------------------------#
        for params in self.q_critic.parameters():
            params.requires_grad = True

        with torch.no_grad():
            V_next = self.v_critic_target(s_next)
            #############################################################		
            ### Q(s, a) = r + γ * (1 - done) * V(s') ###
            if self.robust:
                target_Q = r + (~dw) * self.gamma * V_next_opt
                if self.debug_print:
                    print(((V_next_opt - V_next) / V_next).norm().item()) # difference of robust update
                    # print(((V_next_opt - V_next_opt_acc) / V_next_opt).norm().item()) # difference of reparate and joint optimize
            else:
                target_Q = r + (~dw) * self.gamma * V_next
            #############################################################

        # Get current Q estimates
        current_Q1, current_Q2 = self.q_critic(s, a)

        # JQ(θ)
        q_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q) 

        for name,param in self.q_critic.named_parameters():
            if 'weight' in name:
                q_loss += param.pow(2).sum() * self.l2_reg

        self.q_critic_optimizer.zero_grad()
        q_loss.backward()
        self.q_critic_optimizer.step()
        if self.debug_print:
            print(f"q_loss: {q_loss.item()}")
        if writer:
            writer.add_scalar('q_loss', q_loss, global_step=step)

        #----------------------------- ↓↓↓↓↓ Update V Net ↓↓↓↓↓ ------------------------------#
        for params in self.q_critic.parameters():
            params.requires_grad = False
        for params in self.v_critic.parameters():
            params.requires_grad = True

        a, log_pi_a = self.actor(s, deterministic=False, with_logprob=True)
        current_Q1, current_Q2 = self.q_critic(s, a)
        Q = torch.min(current_Q1, current_Q2)
        ### V(s) = E_pi(Q(s,a) - α * logπ(a|s)) ###
        target_V = (Q - self.alpha * log_pi_a).detach()

        current_V = self.v_critic(s)
        v_loss = F.mse_loss(current_V, target_V)

        for name,param in self.v_critic.named_parameters():
            if 'weight' in name:
                v_loss += param.pow(2).sum() * self.l2_reg

        self.v_critic_optimizer.zero_grad()
        v_loss.backward()
        self.v_critic_optimizer.step()
        if self.debug_print:
            print(f"v_loss: {v_loss.item()}")
        if writer:
            writer.add_scalar('v_loss', v_loss, global_step=step)

        #----------------------------- ↓↓↓↓↓ Update Actor Net ↓↓↓↓↓ ------------------------------#
        # Freeze critic so you don't waste computational effort computing gradients for them when update actor
        for params in self.v_critic.parameters():
            params.requires_grad = False

        # a, log_pi_a = self.actor(s, deterministic=False, with_logprob=True)
        # current_Q1, current_Q2 = self.q_critic(s, a)
        # Q = torch.min(current_Q1, current_Q2)

        # Entropy Regularization
        # Note that the entropy term is not included in the loss function
        #########################################
        ### Jπ(θ) = E[α * logπ(a|s) - Q(s,a)] ###
        a_loss = (self.alpha * log_pi_a - Q).mean()
        #########################################
        self.actor_optimizer.zero_grad()
        a_loss.backward()
        self.actor_optimizer.step()
        if self.debug_print:
            print(f"a_loss: {a_loss.item()}\n")
        if writer:
            writer.add_scalar('a_loss', a_loss, global_step=step)

        #----------------------------- ↓↓↓↓↓ Update alpha ↓↓↓↓↓ ------------------------------#
        if self.adaptive_alpha: # Adaptive alpha SAC
            # We learn log_alpha instead of alpha to ensure alpha>0
            alpha_loss = -(self.log_alpha * (log_pi_a + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp() 

        #----------------------------- ↓↓↓↓↓ Update Target Net ↓↓↓↓↓ ------------------------------#
        for param, target_param in zip(self.v_critic.parameters(), self.v_critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    def save(self, EnvName):
        params = f"{self.std}_{self.robust}"
        model_dir = Path(f"./models/SAC_model/{EnvName}")
        model_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self.actor.state_dict(), model_dir / f"actor_{params}.pth")
        torch.save(self.q_critic.state_dict(), model_dir / f"q_{params}.pth")
        torch.save(self.v_critic.state_dict(), model_dir / f"v_{params}.pth")

    def load(self, EnvName, params):
        model_dir = Path(f"./models/SAC_model/{EnvName}")

        self.actor.load_state_dict(torch.load(model_dir / f"actor_{params}.pth", map_location=self.device, weights_only=True))
        self.q_critic.load_state_dict(torch.load(model_dir / f"q_{params}.pth", map_location=self.device, weights_only=True))
        self.v_critic.load_state_dict(torch.load(model_dir / f"v_{params}.pth", map_location=self.device, weights_only=True))

class Abstract_AC(ABC):
    def __init__(self, cfg: DictConfig):
        """Initialize the SAC trainer with configuration"""
        # Store config
        self.cfg = cfg

        # Set up main logger
        self.log = logging.getLogger(__name__)
        self.setup_logging()

        # Initialize environment and agent
        self.setup_environment()
        self.setup_agent()

        # Set up TensorBoard if needed
        self.setup_tensorboard()

    def setup_logging(self):
        """Set up logging facilities"""
        self.log.info(f"Configuration:\n{OmegaConf.to_yaml(self.cfg)}")

        # Create output directory
        self.output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Configure file logging
        file_handler = logging.FileHandler(self.output_dir / "train.log")
        file_handler.setFormatter(logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] - %(message)s'))
        self.log.addHandler(file_handler)

        # Set up summary logger
        self.summary_path = self.output_dir / "summary.log"
        summary_handler = logging.FileHandler(self.summary_path)
        summary_handler.setLevel(logging.INFO)
        summary_formatter = logging.Formatter('[%(asctime)s] %(message)s')
        summary_handler.setFormatter(summary_formatter)

        self.summary_logger = logging.getLogger("summary")
        self.summary_logger.setLevel(logging.INFO)
        self.summary_logger.addHandler(summary_handler)
        self.summary_logger.propagate = False
        self.summary_logger.info(f"Starting SAC training with configuration: {self.cfg.env_name}")

        # Log system information
        system_info = {
            "Platform": platform.platform(),
            "Python": platform.python_version(),
            "PyTorch": torch.__version__,
            "CUDA Available": torch.cuda.is_available(),
            "CUDA Version": torch.version.cuda if torch.cuda.is_available() else "N/A",
            "GPU": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
        }

        self.log.info(f"System information:")
        for key, value in system_info.items():
            self.log.info(f"  {key}: {value}")
        self.summary_logger.info(f"System: {system_info['Platform']}, PyTorch: {system_info['PyTorch']}, GPU: {system_info['GPU']}")

    def setup_environment(self):
        """Set up training and evaluation environments"""
        # Define environment names and abbreviations
        self.env_names = [
            'Pendulum-v1',
            "ContinuousCartPole",
            'LunarLanderContinuous-v3',
            'Humanoid-v5',
            'HalfCheetah-v4',
            'BipedalWalker-v3',
            'BipedalWalkerHardcore-v3',
            'FrozenLake-v1'
        ]
        self.brief_env_names = [
            'PV1',
            "CPV0",
            'LLdV2',
            'Humanv5',
            'HCv4',
            'BWv3',
            'BWHv3',
            'CRv3'
        ]

        # Create opt object from cfg for compatibility with existing code
        self.opt = DictConfig({})
        for key, value in self.cfg.items():
            if key not in ['hydra']:  # Skip hydra config
                setattr(self.opt, key, value)

        # Create environments based on config
        if hasattr(self.cfg, 'env_mods') and self.cfg.env_mods.use_mods:
            # Import the environment_modifiers module
            from environment_modifiers import create_env_with_mods
            self.log.info("Using environment modifications from config")
            self.env, self.eval_env = create_env_with_mods(
                self.env_names[self.opt.env_index], 
                self.cfg.env_mods
            )
            self.summary_logger.info(f"Environment modifications enabled: {self.cfg.env_mods.use_mods}")
        else:
            # Use legacy noise settings if env_mods is not used
            if not self.opt.noise:
                self.env = gym.make(self.env_names[self.opt.env_index])
                self.eval_env = gym.make(self.env_names[self.opt.env_index])
            else:
                if self.opt.env_index == 0:
                    self.env = gym.make("CustomPendulum-v1", std=self.opt.std)
                    self.eval_env = gym.make("CustomPendulum-v1", std=2*self.opt.std)

        # Extract environment properties
        self.opt.state_dim = self.env.observation_space.shape[0]
        self.opt.action_dim = self.env.action_space.shape[0]
        self.opt.max_action = float(self.env.action_space.high[0])
        self.opt.max_e_steps = self.env._max_episode_steps

        # Log environment info
        self.log.info(
            f"Env: {self.env_names[self.opt.env_index]}  "
            f"state_dim: {self.opt.state_dim}  "
            f"action_dim: {self.opt.action_dim}  "
            f"max_a: {self.opt.max_action}  "
            f"min_a: {self.env.action_space.low[0]}  "
            f"max_e_steps: {self.opt.max_e_steps}"
        )

        # Seed everything for reproducibility
        self.env_seed = self.opt.seed
        random.seed(self.opt.seed)
        np.random.seed(self.opt.seed)
        torch.manual_seed(self.opt.seed)
        torch.cuda.manual_seed(self.opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.log.info(f"Random Seed: {self.opt.seed}")

    @abstractmethod
    def setup_agent(self):
        """Initialize the agent"""
        pass

    def setup_tensorboard(self):
        """Set up TensorBoard logging if enabled"""
        self.writer = None
        if self.opt.write:
            from torch.utils.tensorboard import SummaryWriter
            writepath = self.output_dir / "tensorboard"
            writepath.mkdir(exist_ok=True)
            self.writer = SummaryWriter(log_dir=writepath)
            self.log.info(f"TensorBoard logs will be saved to {writepath}")

    @abstractmethod
    def render(self):
        """Render the agent in the environment"""
        pass

    @abstractmethod
    def evaluate(self):
        """Evaluate the agent's performance"""
        pass

    @abstractmethod
    def train(self):
        """Train the agent"""
        pass

    def run(self):
        """Main method to run the appropriate action based on config"""
        if self.opt.render:
            self.render()
        elif self.opt.eval_model:
            self.evaluate()
        else:
            self.train()

        # Clean up
        self.env.close()
        self.eval_env.close()
        
        if self.writer is not None:
            self.writer.close()

        return self.agent

class DR_SAC(Abstract_AC):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    def setup_agent(self):
        """Initialize the SAC agent"""
        # Create directory for saving models
        self.model_dir = Path(f'models/SAC_model/{self.brief_env_names[self.opt.env_index]}')
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.log.info(f"Models will be saved to {self.model_dir}")

        # Initialize agent
        self.agent = SAC_continuous(**OmegaConf.to_container(self.opt, resolve=True))

        # Load a saved model if requested
        if self.opt.load_model:
            self.log.info("Loading pre-trained model")
            params = f"{self.opt.std}_{self.opt.robust}"
            self.agent.load(self.brief_env_names[self.opt.env_index], params)

    def render(self):
        """Run agent in render mode for visualization"""
        self.log.info("Starting render mode")
        while True:
            ep_r = evaluate_policy(self.env, self.agent, self.opt.max_action, turns=1)
            self.log.info(f"Env: {self.env_names[self.opt.env_index]}, Episode Reward: {ep_r}")

    def evaluate(self):
        """Evaluate the agent's performance"""
        eval_num = 100
        self.log.info(f"Evaluating agent across {eval_num} episodes")

        # Setup seed list for reproducibility
        seeds_list = [random.randint(0, 100000) for _ in range(eval_num)] if not hasattr(self.opt, 'seeds_list') else self.opt.seeds_list

        scores = []
        # Use tqdm for evaluation progress
        for i in tqdm(range(eval_num), desc="Evaluation Progress", ncols=100):
            score = evaluate_policy(self.eval_env, self.agent, turns=1, seeds_list=[seeds_list[i]])
            scores.append(score)
            # Update progress bar with current mean score
            if i > 0 and i % 5 == 0:
                current_mean = np.mean(scores[:i])
                tqdm.write(f"Current mean score after {i} episodes: {current_mean:.2f}")
                # Log intermediate results to summary
                self.summary_logger.info(f"Intermediate evaluation ({i}/{eval_num}): Mean score = {current_mean:.2f}")

        # Calculate statistics
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        p90_score = np.quantile(scores, 0.9)
        p10_score = np.quantile(scores, 0.1)

        # Save results to output directory
        results_path = self.output_dir / "results.txt"
        with open(results_path, 'a') as f:
            f.write(f"{[self.brief_env_names[self.opt.env_index], self.opt.std, self.opt.robust, mean_score, std_score, p90_score, p10_score]}\n")

        self.log.info(f"Results: {self.brief_env_names[self.opt.env_index]}, Mean: {mean_score:.2f}, Std: {std_score:.2f}")
        self.log.info(f"90th percentile: {p90_score:.2f}, 10th percentile: {p10_score:.2f}")
        self.log.info(f"Results saved to {results_path}")

        # Log final results to summary file
        self.summary_logger.info("-" * 50)
        self.summary_logger.info("EVALUATION COMPLETED")
        self.summary_logger.info(f"Environment: {self.env_names[self.opt.env_index]}")
        self.summary_logger.info(f"Evaluation over {eval_num} episodes:")
        self.summary_logger.info(f"  Mean reward: {mean_score:.2f} ± {std_score:.2f}")
        self.summary_logger.info(f"  90th percentile: {p90_score:.2f}")
        self.summary_logger.info(f"  10th percentile: {p10_score:.2f}")
        self.summary_logger.info("-" * 50)

        return mean_score, std_score
    
    def train(self):
        """Train the agent"""
        total_steps = 0
        total_episode = 0

        # Create a progress bar for the total training steps
        with tqdm(total=self.opt.max_train_steps, desc="Training Progress", ncols=100) as pbar:
            while total_steps < self.opt.max_train_steps:
                # (a) Reset environment with incremented seed
                state, info = self.env.reset(seed=self.env_seed)
                self.env_seed += 1
                total_episode += 1
                done = False
                ep_reward = 0

                # Create a progress bar for steps within this episode
                episode_pbar = tqdm(total=self.opt.max_e_steps, desc=f"Episode {total_episode}", 
                                    leave=False, ncols=100, position=1)

                # (b) Interact with environment until episode finishes
                episode_steps = 0
                while not done:
                    # Random exploration for first 50 episodes
                    if total_steps < (50 * self.opt.max_e_steps):
                        # Sample action directly from environment's action space
                        action_env = self.env.action_space.sample()
                        # Convert env action back to agent's internal range [-1,1]
                        action_agent = Action_adapter_reverse(action_env, self.opt.max_action)
                    else:
                        # Select action from agent (internal range [-1,1])
                        action_agent = self.agent.select_action(state, deterministic=False)
                        # Convert agent action to environment range
                        action_env = Action_adapter(action_agent, self.opt.max_action)

                    # Step the environment
                    next_state, reward, dw, tr, info = self.env.step(action_env)
                    ep_reward += reward

                    # Custom reward shaping, if needed
                    if self.opt.reward_adapt:
                        reward = Reward_adapter(reward, self.opt.env_index)

                    # Check for terminal state
                    done = (dw or tr)

                    # Store transition in replay buffer
                    self.agent.replay_buffer.add(state, action_agent, reward, next_state, dw)

                    # Move to next step
                    state = next_state
                    total_steps += 1
                    episode_steps += 1

                    # Update progress bars
                    pbar.update(1)
                    episode_pbar.update(1)

                    # Update progress bar description with more info
                    if total_steps % 10 == 0:
                        pbar.set_postfix({
                            'episode': total_episode,
                            'reward': f"{ep_reward:.2f}"
                        })

                    # (c) Train the agent at fixed intervals (batch updates)
                    if (total_steps >= 50 * self.opt.max_e_steps) and (total_steps % self.opt.update_every == 0):
                        writer_copy = self.writer
                        train_bar = tqdm(range(self.opt.update_every), 
                                        desc="Model Update", 
                                        leave=False, ncols=100, position=2)

                        for i in train_bar:
                            self.agent.train(writer_copy, total_steps)
                            writer_copy = False

                        # Learning rate decay
                        self.agent.a_lr *= 0.999
                        self.agent.c_lr *= 0.999

                    # (d) Evaluate and log periodically
                    if total_steps % self.opt.eval_interval == 0:
                        # Temporarily close progress bars for evaluation
                        episode_pbar.close()
                        pbar.set_description("Evaluating...")
                        ep_r = evaluate_policy(self.eval_env, self.agent, turns=10)

                        if self.writer is not None:
                            self.writer.add_scalar('ep_r', ep_r, global_step=total_steps)

                        self.log.info(
                            f"EnvName: {self.brief_env_names[self.opt.env_index]}, "
                            f"Steps: {int(total_steps/1000)}k, "
                            f"Episodes: {total_episode}, "
                            f"Episode Reward: {ep_r}"
                        )

                        # Reset progress bar description
                        pbar.set_description("Training Progress")
                        episode_pbar = tqdm(total=self.opt.max_e_steps, initial=episode_steps,
                                            desc=f"Episode {total_episode}", 
                                            leave=False, ncols=100, position=1)

                    # (e) Save model at fixed intervals
                    if self.opt.save_model and total_steps % self.opt.save_interval == 0:
                        self.agent.save(self.brief_env_names[self.opt.env_index])

                # Close episode progress bar when episode ends
                episode_pbar.close()

                # Log episode stats
                self.log.info(f"Episode {total_episode} completed with reward {ep_reward:.2f} in {episode_steps} steps")

        # Final evaluation after training
        self._evaluate_final(total_steps, total_episode)

        # Save final model
        if self.opt.save_model:
            self.agent.save(self.brief_env_names[self.opt.env_index])
            self.log.info(f"Final model saved to models/SAC_model/{self.brief_env_names[self.opt.env_index]}")

    def _evaluate_final(self, total_steps, total_episode):
        """Evaluate the agent after training is complete"""
        eval_num = 20
        self.log.info(f"Training completed. Evaluating across {eval_num} episodes")
        scores = []

        # Create a progress bar for evaluation
        for i in tqdm(range(eval_num), desc="Final Evaluation", ncols=100):
            score = evaluate_policy(self.eval_env, self.agent, turns=1)
            scores.append(score)

        mean_score = np.mean(scores)
        std_score = np.std(scores)
        p90_score = np.quantile(scores, 0.9)
        p10_score = np.quantile(scores, 0.1)

        self.log.info(f"Final evaluation - Mean: {mean_score:.2f}, Std: {std_score:.2f}")
        self.log.info(f"90th percentile: {p90_score:.2f}, 10th percentile: {p10_score:.2f}")

        # Log final results to summary file
        self.summary_logger.info("-" * 50)
        self.summary_logger.info("TRAINING COMPLETED")
        self.summary_logger.info(f"Environment: {self.env_names[self.opt.env_index]}")
        self.summary_logger.info(f"Total steps: {total_steps}")
        self.summary_logger.info(f"Total episodes: {total_episode}")
        self.summary_logger.info(f"Final evaluation over {eval_num} episodes:")
        self.summary_logger.info(f"  Mean reward: {mean_score:.2f} ± {std_score:.2f}")
        self.summary_logger.info(f"  90th percentile: {p90_score:.2f}")
        self.summary_logger.info(f"  10th percentile: {p10_score:.2f}")
        self.summary_logger.info("-" * 50)

@hydra.main(version_base=None, config_path="config", config_name="sac_config")
def main(cfg: DictConfig):
    """
    Main function to train and evaluate an SAC agent on different environments.
    """
    trainer = DR_SAC(cfg)
    return trainer.run()

if __name__ == '__main__':
    main()
