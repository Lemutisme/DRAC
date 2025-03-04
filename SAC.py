from utils import evaluate_policy_SAC as evaluate_policy
from utils import Action_adapter_symm as Action_adapter
from utils import Action_adapter_symm_reverse as Action_adapter_reverse
from utils import build_net, Reward_adapter, str2bool, register
from continuous_cartpole import register

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import torch.optim.adam
import copy
import gymnasium as gym
import os, shutil
import argparse
import math
from scipy.special import logsumexp
from scipy.optimize import minimize, minimize_scalar

######################################################
## NOTE: Current Template
# Actor, Double_Q_Critic, V_Critic: MLP
# Transition learner: VAE
# Three dual optimiztion options:
# 1. beta, minimize loss.mean()
# 2. functional g, minimize scalar loss, equivalent to replace beta with g(s,a)
# 3. independent optimize, too slow
######################################################


######################################################
## TODO: Add the following imlementation

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
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)  
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
    
class dual(nn.Module):
    def __init__(self, state_dim, action_dim, hid_shape, hid_layers):
        super(dual, self).__init__()  
        layers = [state_dim + action_dim] + list(hid_shape) * hid_layers + [1]

        self.G = build_net(layers, nn.ReLU, ExpActivation)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)          
        return self.G(sa)

class ExpActivation(nn.Module):
    def forward(self, x):
        return torch.exp(x)   
            
class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size, device):
        self.max_size = max_size
        self.device = device
        self.ptr = 0
        self.size = 0

        self.s = torch.zeros((max_size, state_dim) ,dtype=torch.float,device=self.device)
        self.a = torch.zeros((max_size, action_dim) ,dtype=torch.float,device=self.device)
        self.r = torch.zeros((max_size, 1) ,dtype=torch.float,device=self.device)
        self.s_next = torch.zeros((max_size, state_dim) ,dtype=torch.float,device=self.device)
        self.dw = torch.zeros((max_size, 1) ,dtype=torch.bool,device=self.device)

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


class SAC_countinuous():
    def __init__(self, **kwargs):
        # Init hyperparameters for agent, just like "self.gamma = opt.gamma, self.lambd = opt.lambd, ..."
        self.__dict__.update(kwargs)
        self.tau = 0.005

        self.actor = Actor(self.state_dim, self.action_dim, (self.net_width, self.net_width), self.net_layer).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.a_lr)
        
        self.v_critic = V_Critic(self.state_dim, (self.net_width, self.net_width), self.net_layer).to(self.device)
        self.v_critic_optimizer = torch.optim.Adam(self.v_critic.parameters(), lr=self.c_lr)
        self.v_critic_target = copy.deepcopy(self.v_critic)
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.v_critic_target.parameters():
            p.requires_grad = False

        self.q_critic = Double_Q_Critic(self.state_dim, self.action_dim, (self.net_width, self.net_width), self.net_layer).to(self.device)
        self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=self.c_lr)
        
        if self.robust:    
            print('This is a robust policy.')
            self.transition = TransitionVAE(self.state_dim, self.action_dim, self.state_dim, (self.net_width, self.net_width), self.net_layer).to(self.device)
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

    def train(self, robust_update, printer, writer, step):
        s, a, r, s_next, dw = self.replay_buffer.sample(self.batch_size)
        
        #----------------------------- ↓↓↓↓↓ Update R Net ↓↓↓↓↓ ------------------------------#
        if self.robust:
            s_next_recon, mu, logvar = self.transition(s, a, s_next)
            tr_loss = self.vae_loss(s_next, s_next_recon, mu, logvar)
            self.trans_optimizer.zero_grad()
            tr_loss.backward()
            self.trans_optimizer.step()
            if printer:
                print(f"tr_loss: {tr_loss.item()}")
            if writer:
                writer.add_scalar('tr_loss', tr_loss, global_step=step)
                
        if robust_update:   
            with torch.no_grad():
                s_next_sample = self.transition.sample(s, a, 200)
            
            #############################################################		
            ### option1: optimize w.r.t beta ###
            # self.log_beta = nn.Parameter(torch.ones_like(self.log_beta, requires_grad=True, device=self.device) * 0.1)
            # self.beta_optimizer = torch.optim.Adam([self.log_beta], lr=self.b_lr)

            # for _ in range(5):
            #     self.beta = torch.exp(self.log_beta)
            #     opt_loss = -self.dual_func_beta(s_next_sample, self.beta)
            #     self.beta_optimizer.zero_grad()
            #     opt_loss.mean().backward()
            #     if printer:
            #         print(opt_loss.sum().item())
            #     self.beta_optimizer.step() 
            
            # V_next_opt = self.dual_func(s_next_sample, torch.exp(self.log_beta)) 
            #############################################################		

            #############################################################		
            ### option2: optimize w.r.t functional g ###
            for _ in range(5):
                opt_loss = -self.dual_func_g(s, a, s_next_sample)
                self.g_optimizer.zero_grad()
                opt_loss.mean().backward()
                self.g_optimizer.step() 
                # if printer:
                #     print(opt_loss.mean().item())
            
            V_next_opt = self.dual_func_g(s, a, s_next_sample) 
            #############################################################		
            
            #############################################################		
            # option3: Use scipy.optimize to separately optimize
            # s_next_sample = s_next_sample.cpu().numpy()
            # V_next_opt_acc = np.zeros((self.batch_size, 1))
            # for i in range(s_next_sample.shape[0]):
            #     opt = minimize_scalar(fun=lambda beta:-self.dual_func_ind(s_next_sample[i], beta), method='Bounded', bounds=(1e-4, 1.0))
            #     V_next_opt_acc[i] = -opt.fun
            # V_next_opt_acc = torch.from_numpy(V_next_opt_acc).float()
            # V_next_opt_acc = V_next_opt_acc.to('cuda' if torch.cuda.is_available() else 'cpu')
            #############################################################		


        #----------------------------- ↓↓↓↓↓ Update Q Net ↓↓↓↓↓ ------------------------------#
        for params in self.q_critic.parameters():
            params.requires_grad = True
            
        with torch.no_grad():
            V_next = self.v_critic_target(s_next)
            #############################################################		
            ### Q(s, a) = r + γ * (1 - done) * V(s') ###
            if robust_update:
                target_Q = r + (~dw) * self.gamma * V_next_opt
                if printer:
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
                q_loss += param.pow(2).sum() * opt.l2_reg
        
        self.q_critic_optimizer.zero_grad()
        q_loss.backward()
        self.q_critic_optimizer.step()
        if printer:
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
                v_loss += param.pow(2).sum() * opt.l2_reg
        
        self.v_critic_optimizer.zero_grad()
        v_loss.backward()
        self.v_critic_optimizer.step()
        if printer:
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
        if printer:
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
        torch.save(self.actor.state_dict(), "./models/SAC_model/{}/actor_{}.pth".format(EnvName,params))
        torch.save(self.q_critic.state_dict(), "./models/SAC_model/{}/q_{}.pth".format(EnvName,params))
        torch.save(self.v_critic.state_dict(), "./models/SAC_model/{}/v_{}.pth".format(EnvName,params))

    def load(self, EnvName, params):
        self.actor.load_state_dict(torch.load("./models/SAC_model/{}/actor_{}.pth".format(EnvName, params), map_location=self.device, weights_only=True))
        self.q_critic.load_state_dict(torch.load("./models/SAC_model/{}/q_{}.pth".format(EnvName, params), map_location=self.device, weights_only=True))
        self.v_critic.load_state_dict(torch.load("./models/SAC_model/{}/v_{}.pth".format(EnvName, params), map_location=self.device, weights_only=True))

def main(opt):
    """
    Main function to train and evaluate an SAC agent on different environments.
    """

    # 1. Define environment names and abbreviations
    EnvName = [
        'Pendulum-v1',
        "ContinuousCartPole",
        'LunarLanderContinuous-v3',
        'Humanoid-v5',
        'HalfCheetah-v4',
        'BipedalWalker-v3',
        'BipedalWalkerHardcore-v3',
        'FrozenLake-v1'
    ]
    BrifEnvName = [
        'PV1',
        "CPV0",
        'LLdV2',
        'Humanv5',
        'HCv4',
        'BWv3',
        'BWHv3',
        'CRv3'
    ]

    # 2. Create training and evaluation environments
    if not opt.noise:
        env = gym.make(EnvName[opt.EnvIdex])
        eval_env =  gym.make(EnvName[opt.EnvIdex])
    else:
        if opt.EnvIdex == 0:
            env = gym.make("CustomPendulum-v1", std=opt.std) # Add noise when updating angle
            eval_env = gym.make("CustomPendulum-v1", std=2*opt.std) # Add noise when updating angle

    # 3. Extract environment properties
    opt.state_dim = env.observation_space.shape[0]
    opt.action_dim = env.action_space.shape[0]  # Continuous action dimension
    opt.max_action = float(env.action_space.high[0])  # Action range [-max_action, max_action]
    opt.max_e_steps = env._max_episode_steps    

    # 4. Print environment info
    print(
        f"Env: {EnvName[opt.EnvIdex]}  "
        f"state_dim: {opt.state_dim}  "
        f"action_dim: {opt.action_dim}  "
        f"max_a: {opt.max_action}  "
        f"min_a: {env.action_space.low[0]}  "
        f"max_e_steps: {opt.max_e_steps}"
    )

    # 5. Seed everything for reproducibility
    env_seed = opt.seed
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random Seed: {opt.seed}")

    # 6. Set up TensorBoard for logging (if requested)
    writer = None
    if opt.write:
        from torch.utils.tensorboard import SummaryWriter
        # timenow = str(datetime.now())[:-10]    # e.g. 2025-01-10 17:45
        # timenow = ' ' + timenow[:13] + '_' + timenow[-2:]  # e.g. ' 2025-01-10_45'
        writepath = f"runs/SAC/{BrifEnvName[opt.EnvIdex]}"
        if opt.noise:
            writepath += f"_Noise_{opt.std}"
        if opt.robust:
            writepath += f"_Robust"
        if os.path.exists(writepath):
            shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)

    # 7. Create a directory for saving models

    dir = f'models/SAC_model/{BrifEnvName[opt.EnvIdex]}'
    if not os.path.exists(dir):
        os.mkdir(dir)

    # 8. Initialize the SAC agent
    agent = SAC_countinuous(**vars(opt))  # Convert argparse Namespace to dict

    # 9. Load a saved model if requested
    if opt.load_model:
        print("Load Model.")
        params = f"{opt.std}_{opt.robust}"
        agent.load(BrifEnvName[opt.EnvIdex], params)

    # 10. If rendering mode is on, run an infinite evaluation loop
    if opt.render:
        while True:
            ep_r = evaluate_policy(env, agent, opt.max_action, turns=1)
            print(f"Env: {EnvName[opt.EnvIdex]}, Episode Reward: {ep_r}")
    
    # 11. If evaluating only, print result
    elif opt.eval_model:
        eval_num = 100
        print(f"Evaluate {eval_num} episodes.")
        scores = []
        for i in range(eval_num):
            score = evaluate_policy(eval_env, agent, turns=1, seeds_list=[opt.seeds_list[i]])
            scores.append(score)
        # filename = "new-robust.txt" if opt.robust else "new-non-robust.txt"
        # with open(filename, 'a') as f:
        #     f.write(f"{[BrifEnvName[opt.EnvIdex], opt.train_std, opt.eval_std, delta] + [np.mean(scores), np.std(scores), np.quantile(scores, 0.9), np.quantile(scores, 0.1)]}\n")
        print(f"{[BrifEnvName[opt.EnvIdex]] + [np.mean(scores), np.std(scores), np.quantile(scores, 0.9), np.quantile(scores, 0.1)]}\n")    
            

    # 12. Otherwise, proceed with training
    else:
        total_steps = 0
        total_episode = 0

        while total_steps < opt.max_train_steps:
            # (a) Reset environment with incremented seed
            state, info = env.reset(seed=env_seed)
            env_seed += 1
            total_episode += 1
            done = False

            # (b) Interact with environment until episode finishes
            while not done:
                # Random exploration for first 50 episodes (each episode is up to max_e_steps)
                if total_steps < (50 * opt.max_e_steps):
                    # Sample action directly from environment's action space
                    action_env = env.action_space.sample()  # Range: [-max_action, max_action]
                    # Convert env action back to agent's internal range [-1,1]
                    action_agent = Action_adapter_reverse(action_env, opt.max_action)
                else:
                    # Select action from agent (internal range [-1,1])
                    action_agent = agent.select_action(state, deterministic=False)
                    # Convert agent action to environment range
                    action_env = Action_adapter(action_agent, opt.max_action)

                # Step the environment
                next_state, reward, dw, tr, info = env.step(action_env)

                # Custom reward shaping, if needed
                if opt.reward_adapt:
                    reward = Reward_adapter(reward, opt.EnvIdex)

                # Check for terminal state
                done = (dw or tr)

                # Store transition in replay buffer
                agent.replay_buffer.add(state, action_agent, reward, next_state, dw)

                # Move to next step
                state = next_state
                total_steps += 1

                # (c) Train the agent at fixed intervals (batch updates)
                if (total_steps >= 50 * opt.max_e_steps) and (total_steps % opt.update_every == 0):
                    printer = False
                    writer_copy = writer
                    if total_steps % 500 == 0:
                        printer = True
                    for i in range(opt.update_every):
                        # if i % opt.robust_update_every == 0:
                        #     agent.train(agent.robust, printer)
                        # else:
                        #     agent.train(False, printer)
                        agent.train(agent.robust, printer, writer_copy, total_steps)
                        printer = False
                        writer_copy = False
                    
                    agent.a_lr *= 0.999
                    agent.c_lr *= 0.999

                # (d) Evaluate and log periodically
                if total_steps % opt.eval_interval == 0:
                    ep_r = evaluate_policy(eval_env, agent, turns=10)
                    if writer is not None:
                        writer.add_scalar('ep_r', ep_r, global_step=total_steps)
                    print(
                        f"EnvName: {BrifEnvName[opt.EnvIdex]}, "
                        f"Steps: {int(total_steps/1000)}k, "
                        f"Episodes: {total_episode}, "
                        f"Episode Reward: {ep_r}"
                    )

                # (e) Save model at fixed intervals
                if opt.save_model and total_steps % opt.save_interval == 0:
                    agent.save(BrifEnvName[opt.EnvIdex])
        
        # 11.5 Compute score of 20 episodes
        eval_num = 20
        print(f"Train finished. Now generate scores of {eval_num} episodes.")
        scores = []
        for _ in range(eval_num):
            score = evaluate_policy(eval_env, agent, turns=1)
            scores.append(score)
        print(np.mean(scores), np.std(scores))
        # filename = "robust.txt" if opt.robust else "non-robust.txt"
        # with open(filename, 'a') as f:
        #       f.write(f"{[BrifEnvName[opt.EnvIdex], opt.train_std, opt.eval_std] + scores}\n")

        # 12. Save model at last
        if opt.save_model:
            agent.save(BrifEnvName[opt.EnvIdex])

    env.close()
    eval_env.close()

    
if __name__ == '__main__':
    '''Hyperparameter Setting'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda', help='running device: cuda or cpu')
    parser.add_argument('--EnvIdex', type=int, default=0, help='PV1, Lch_Cv2, Humanv4, HCv4, BWv3, BWHv3, CRv3')
    parser.add_argument('--write', type=str2bool, default=False, help='Use SummaryWriter to record the training')
    parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
    parser.add_argument('--load_model', type=str2bool, default=False, help='Load pretrained model or Not')
    parser.add_argument('--eval_model', type=str2bool, default=False, help='Evaluate only')
    parser.add_argument('--save_model', type=str2bool, default=True, help='Save or not')

    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--max_train_steps', type=int, default=int(2e5), help='Max training steps')
    parser.add_argument('--save_interval', type=int, default=int(1e4), help='Model saving interval, in steps.')
    parser.add_argument('--eval_interval', type=int, default=int(1e3), help='Model evaluating interval, in steps.')
    parser.add_argument('--update_every', type=int, default=50, help='Training Fraquency, in stpes')
    # parser.add_argument('--robust_update_every', type=int, default=2, help='Training Fraquency, in stpes')

    parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
    parser.add_argument('--net_width', type=int, default=256, help='Hidden net width')
    parser.add_argument('--net_layer', type=int, default=1, help='Hidden net layers')
    parser.add_argument('--a_lr', type=float, default=5e-3, help='Learning rate of actor')
    parser.add_argument('--c_lr', type=float, default=5e-5, help='Learning rate of critic')
    parser.add_argument('--b_lr', type=float, default=5e-5, help='Learning rate of dual-form optimization')
    parser.add_argument('--g_lr', type=float, default=5e-4, help='Learning rate of dual net')
    parser.add_argument('--r_lr', type=float, default=5e-5, help='Learning rate of reward net')
    parser.add_argument('--l2_reg', type=float, default=1e-4, help='L2 regulization coefficient for Critic')
    parser.add_argument('--batch_size', type=int, default=256, help='batch_size of training')
    parser.add_argument('--alpha', type=float, default=0.12, help='Entropy coefficient')
    parser.add_argument('--adaptive_alpha', type=str2bool, default=True, help='Use adaptive_alpha or Not')
    
    # parser.add_argument('--reg_coef', type=float, default=0.0, help='Regulator of Network Parameters')
    parser.add_argument('--reward_adapt', type=bool, default=True, help='Reward adaptation')
    parser.add_argument('--robust', type=bool, default=False, help='Robust policy')
    parser.add_argument('--noise', type=bool, default=False, help='Evaluation Env Noise')
    parser.add_argument('--std', type=float, default=0.0, help='Evaluation Env Noise')
    parser.add_argument('--delta', type=float, default=0.0, help='Evaluation Env Noise') 

    opt = parser.parse_args()
    opt.device = torch.device(opt.device) # from str to torch.device
    if opt.eval_model:
        opt.seeds_list = [random.randint(0, 100000) for _ in range(100)]
        opt.robust = False
        main(opt)
        print("----------------------------------")
        opt.robust = True
        main(opt)
    else:
        print(opt)
        main(opt)

    # Pen step 1e5
    # LLd step 250k 2.5e5
    # Human about 400k