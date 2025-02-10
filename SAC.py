from utils import build_net, str2bool, evaluate_policy, Reward_adapter, Action_adapter, Action_adapter_reverse

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import copy
from datetime import datetime
import gymnasium as gym
from gymnasium.wrappers import TransformReward
import os, shutil
import argparse
import math
from scipy.special import logsumexp
from scipy.optimize import minimize, minimize_scalar
import time

######################################################
## TODO: Add the following imlementation
# 1. Add the main function to introduce the distribution shift
#    e.g. ai_safety_gym.environments.distributional_shift.py
#
# 2. Implement the Bellman operator of DRSAC
#    e.g. parts in SAC.SAC_continous and utils.Actor
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
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)  #总感觉这里clamp不利于学习
        # we learn log_std rather than std, so that exp(log_std) is always > 0
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

class Double_Q_Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hid_shape, hid_layers):
        super(Double_Q_Critic, self).__init__()
        layers = [state_dim + action_dim] + list(hid_shape) * hid_layers + [1]

        self.Q_1 = build_net(layers, nn.ReLU, nn.Identity)
        self.Q_2 = build_net(layers, nn.ReLU, nn.Identity)   

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = self.Q_1(sa)
        q2 = self.Q_2(sa)            
        return q1, q2
           
class Reward(nn.Module):
    def __init__(self, state_dim, action_dim, hid_shape, hid_layers):
        super(Reward, self).__init__()
        layers = [state_dim + action_dim] + list(hid_shape) * hid_layers
        self.rnet = build_net(layers, nn.ReLU, nn.Identity)
        self.mu_layer = nn.Linear(layers[-1], 1)
        self.log_std_layer = nn.Linear(layers[-1], 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        r_out = self.rnet(sa)
        mu = self.mu_layer(r_out)
        log_std = self.log_std_layer(r_out)
        # FIXME:  I think we still nedd to prevent numerical instability
        log_std = torch.clamp(log_std, min=-20, max=20) 
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        r = dist.rsample()     
        return r
    
    def sample(self, state, action, num):
        sa = torch.cat([state, action], 1)
        r_out = self.rnet(sa) 
        mu = self.mu_layer(r_out)
        log_std = self.log_std_layer(r_out)
        #log_std = torch.clamp(log_std, 2, -20)  #总感觉这里clamp不利于学习
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        r = dist.rsample(sample_shape=(num,)) #shape = (num, batch_size, 1)
        r = r.permute(1, 0, 2).squeeze(-1) #shape = (batch_size, num)
        return r
            
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
        #每次只放入一个时刻的数据
        self.s[self.ptr] = torch.from_numpy(s).to(self.device)
        self.a[self.ptr] = torch.from_numpy(a).to(self.device) # Note that a is numpy.array
        self.r[self.ptr] = r
        self.s_next[self.ptr] = torch.from_numpy(s_next).to(self.device)
        self.dw[self.ptr] = dw

        self.ptr = (self.ptr + 1) % self.max_size #存满了又重头开始存
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = torch.randint(0, self.size, device=self.device, size=(batch_size,))
        return self.s[ind], self.a[ind], self.r[ind], self.s_next[ind], self.dw[ind]

# Env Class with Reward Shift
class NoiseRewardWrapper(gym.RewardWrapper):
    def __init__(self, env, func):
        super().__init__(env)
        self.env = env
        self._max_episode_steps = env._max_episode_steps # I don't why it can't inherit
        self.func = func
        print('Env with Reward Shift Made.')

    def step(self, action):
        obs, reward, dw, tr, info = self.env.step(action)
        modified_obs = obs
        modified_obs[2] = self.func(obs[2])
        return modified_obs, reward, dw, tr, info

class ScalingActionWrapper(gym.ActionWrapper):

    """Assumes that actions are symmetric about zero!!!"""

    def __init__(self, env, scaling_factors: np.array):
        super(ScalingActionWrapper, self).__init__(env)
        self._max_episode_steps = env._max_episode_steps # I don't why it can't inherit
        self.scaling_factors = scaling_factors

    def action(self, action):
        return self.scaling_factors * action


class SAC_countinuous():
    def __init__(self, **kwargs):
        # Init hyperparameters for agent, just like "self.gamma = opt.gamma, self.lambd = opt.lambd, ..."
        self.__dict__.update(kwargs)
        if self.robust:
            print('This is a robust policy.\n')
            if self.train_noise and self.eval_noise:
                train_var = self.train_std ** 2
                eval_var = self.eval_std ** 2
                self.delta = 0.5*(eval_var / train_var + math.log(train_var / eval_var)- 1) # KL divergence between two Gaussian distributions with same mean.
                assert self.delta >= 0
            else:
                self.delta = 0
        self.tau = 0.005

        self.actor = Actor(self.state_dim, self.action_dim, (self.net_width, self.net_width), self.net_layer).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.a_lr)

        self.q_critic = Double_Q_Critic(self.state_dim, self.action_dim, (self.net_width, self.net_width), self.net_layer).to(self.device)
        self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=self.c_lr)
        self.q_critic_target = copy.deepcopy(self.q_critic)
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.q_critic_target.parameters():
            p.requires_grad = False
        
        if self.robust:    
            self.reward = Reward(self.state_dim, self.action_dim, (self.net_width, self.net_width), self.net_layer).to(self.device)
            self.reward_optimizer = torch.optim.Adam(self.reward.parameters(), lr=self.r_lr)
            
            self.log_beta = nn.Parameter(torch.ones((self.batch_size,1), requires_grad=True, device=self.device) * 1.0)
            self.beta_optimizer = torch.optim.Adam([self.log_beta], lr=self.b_lr)

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
    
    # def reward_adapt(self, r, mean):
    #     return r / mean * self.delta   
    
    def dual_func(self, r, beta):
        # Jointly optimize, in tensor
        size = r.shape[1]
        return - beta * (torch.logsumexp(-r/beta, dim=1, keepdim=True) - math.log(size)) - beta * self.delta  

    def dual_func_ind(self, r, beta):
        # Independently optimize, in np.array
        size = len(r)
        return - beta * (logsumexp(-r/beta) - math.log(size)) - beta * self.delta           

    def train(self, robust_update, printer):
        s, a, r, s_next, dw = self.replay_buffer.sample(self.batch_size)
        # time1 = time.time()
        # if self.robust and self.delta >0:
        #     r = self.reward_adapt(r, self.r_mean)
        #----------------------------- ↓↓↓↓↓ Update R Net ↓↓↓↓↓ ------------------------------#
        if self.robust:
            r_pred = self.reward(s, a)
            r_loss = F.mse_loss(r_pred, r)
            self.reward_optimizer.zero_grad()
            r_loss.backward()
            self.reward_optimizer.step()
            if printer:
                print(f"r_loss: {r_loss.item()}")
                
        if robust_update:   
            with torch.no_grad():
                r_sample = self.reward.sample(s, a, 200)
                if printer: 
                    print("r_est:", (torch.abs(r - r_sample.mean(dim=1, keepdim=True)) / r).mean().item())
                
            # Reinitiate variable to optimize.
            # FIXME: Do not reinitialize β. Treat it as a persistent parameter updated via gradient descent, 
            # just like actor/critic weights. This ensures stable convergence and efficient use of learned robustness information.
            # Maybe we can get better results and faster convergence.
            # self.log_beta = nn.Parameter(torch.ones(self.batch_size, requires_grad=True, device=self.device) * initial_beta)

            # self.log_beta = nn.Parameter(torch.ones_like(self.log_beta, requires_grad=True, device=self.device) * 0.1)
            # self.beta_optimizer = torch.optim.Adam([self.log_beta], lr=self.b_lr)

            # for _ in range(20):
            #     self.beta = torch.exp(self.log_beta)
            #     opt_loss = -self.dual_func(r_sample, self.beta)
            #     self.beta_optimizer.zero_grad()
            #     # FIXME: opt_loss.mean()
            #     opt_loss.mean().backward()
            #     # if printer:
            #     #     print(opt_loss.sum().item())
            #     self.beta_optimizer.step() 
            
            # r_opt = self.dual_func(r_sample, torch.exp(self.log_beta)) 
            
            # Use scipy.optimize to independently optimize
            r_sample = r_sample.cpu().numpy()
            r_opt = np.zeros((self.batch_size, 1))
            for i in range(r_sample.shape[0]):
                opt = minimize_scalar(fun=lambda beta:-self.dual_func_ind(r_sample[i], beta), method='Bounded', bounds=(1e-4, 1.0))
                r_opt[i] = -opt.fun
            r_opt = torch.from_numpy(r_opt).float()
            r_opt = r_opt.to('cuda' if torch.cuda.is_available() else 'cpu')
            # if printer:
            #     print((r-r_opt).norm(p=1).item(), r_opt.norm(p=1).item())
            # if printer:
            #     print((r_opt1 - r_opt2).norm(p=1).item(), r_opt2.norm(p=1).item())
            # time2 = time.time()
            # print(time2 - time1)

        #----------------------------- ↓↓↓↓↓ Update Q Net ↓↓↓↓↓ ------------------------------#
        with torch.no_grad():
            a_next, log_pi_a_next = self.actor(s_next, deterministic=False, with_logprob=True)
            target_Q1, target_Q2 = self.q_critic_target(s_next, a_next)
            target_Q = torch.min(target_Q1, target_Q2)
            #############################################################		
            ### r + γ * (1 - done) * E_pi(Q(s',a') - α * logπ(a'|s')) ###
            if robust_update:
                target_Q = r_opt + (~dw) * self.gamma * (target_Q - self.alpha * log_pi_a_next)
            else:
                target_Q = r + (~dw) * self.gamma * (target_Q - self.alpha * log_pi_a_next) 
            if printer:
                print(torch.max(target_Q).item(), torch.min(target_Q).item())
            #############################################################

        # Get current Q estimates
        current_Q1, current_Q2 = self.q_critic(s, a)

        # JQ(θ)
        q_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q) 
        
        if self.robust:
            for name, param in self.q_critic.named_parameters():
                if 'weight' in name:
                    q_loss += param.pow(2).sum() * self.reg_coef
        
        self.q_critic_optimizer.zero_grad()
        q_loss.backward()
        self.q_critic_optimizer.step()
        if printer:
            print(f"q_loss: {q_loss.item()}")
        # time3 = time.time()
        # print(time3 - time2)

        #----------------------------- ↓↓↓↓↓ Update Actor Net ↓↓↓↓↓ ------------------------------#
        # Freeze critic so you don't waste computational effort computing gradients for them when update actor
        for params in self.q_critic.parameters():
            params.requires_grad = False

        a, log_pi_a = self.actor(s, deterministic=False, with_logprob=True)
        current_Q1, current_Q2 = self.q_critic(s, a)
        Q = torch.min(current_Q1, current_Q2)

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
            print(f"a_loss: {a_loss.item()}")
        
        for params in self.q_critic.parameters():
            params.requires_grad = True
        # time4 = time.time()
        # print(time4 - time3)

        #----------------------------- ↓↓↓↓↓ Update alpha ↓↓↓↓↓ ------------------------------#
        if self.adaptive_alpha: # Adaptive alpha SAC
            # We learn log_alpha instead of alpha to ensure alpha>0
            alpha_loss = -(self.log_alpha * (log_pi_a + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp()
        
        if printer:
            print("alpha = ", self.alpha.item())

        #----------------------------- ↓↓↓↓↓ Update Target Net ↓↓↓↓↓ ------------------------------#
        for param, target_param in zip(self.q_critic.parameters(), self.q_critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        time5 = time.time()
        # print(time5 - time4)
        # print("\n")

    def save(self, EnvName):
        params = f"{self.train_std}_{self.eval_std}_{self.robust}_new"
        torch.save(self.actor.state_dict(), "./SAC_model/{}_actor{}.pth".format(EnvName,params))
        torch.save(self.q_critic.state_dict(), "./SAC_model/{}_q_critic{}.pth".format(EnvName,params))

    def load(self, EnvName, params):
        self.actor.load_state_dict(torch.load("./SAC_model/{}_actor{}_new.pth".format(EnvName, params), map_location=self.device, weights_only=True))
        self.q_critic.load_state_dict(torch.load("./SAC_model/{}_q_critic{}_new.pth".format(EnvName, params), map_location=self.device, weights_only=True))

def main(opt):
    """
    Main function to train and evaluate an SAC agent on different environments.
    """

    # 1. Define environment names and abbreviations
    EnvName = [
        'Pendulum-v1',
        'LunarLanderContinuous-v3',
        'Humanoid-v5',
        'HalfCheetah-v4',
        'BipedalWalker-v3',
        'BipedalWalkerHardcore-v3',
        'FrozenLake-v1'
    ]
    BrifEnvName = [
        'PV1',
        'LLdV2',
        'Humanv5',
        'HCv4',
        'BWv3',
        'BWHv3',
        'CRv3'
    ]

    # 2. Create training and evaluation environments
    env = gym.make(
        EnvName[opt.EnvIdex]
    )
    # env = ScalingActionWrapper(env, scaling_factors=env.action_space.high)
    if opt.train_noise:
        train_dist = Normal(0, opt.train_std)
        env = NoiseRewardWrapper(env, lambda r: r + train_dist.sample())

    eval_env = gym.make(EnvName[opt.EnvIdex])
    # eval_env = ScalingActionWrapper(eval_env, scaling_factors=env.action_space.high)
    if opt.eval_noise:
        eval_dist = Normal(0, opt.eval_std)
        eval_env = NoiseRewardWrapper(eval_env, lambda r: r + eval_dist.sample())

    # 3. Extract environment properties
    opt.state_dim = env.observation_space.shape[0]
    opt.action_dim = env.action_space.shape[0]  # Continuous action dimension
    opt.max_action = float(env.action_space.high[0])  # Action range [-max_action, max_action]
    opt.max_e_steps = env._max_episode_steps
    # if opt.EnvIdex == 0:
    #     opt.Max_train_steps = 1e5
    # elif opt.EnvIdex == 1:
    #     opt.Max_train_steps = 5e5
    # elif opt.EnvIdex == 2:
    #     opt.Max_train_steps = 1e6       

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
        if opt.train_noise:
            writepath += f"/Train Noise {opt.train_std}"
        if opt.eval_noise:
            writepath += f"/Eval Noise {opt.eval_std}"
        if opt.robust:
             writepath += f"/Robust"
        if os.path.exists(writepath):
            shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)

    # 7. Create a directory for saving models
    if not os.path.exists('model'):
        os.mkdir('model')

    # 8. Initialize the SAC agent
    agent = SAC_countinuous(**vars(opt))  # Convert argparse Namespace to dict

    # 9. Load a saved model if requested
    if opt.Loadmodel:
        params = f"{opt.train_std}_{opt.eval_std}_{opt.robust}"
        agent.load(BrifEnvName[opt.EnvIdex], params)

    # 10. If rendering mode is on, run an infinite evaluation loop
    if opt.render:
        eval_num = 100
        scores = []
        train_var = opt.train_std ** 2
        eval_var = opt.eval_std ** 2
        delta = 0.5*(eval_var / train_var + math.log(train_var / eval_var)- 1)
        for _ in range(eval_num):
            score = evaluate_policy(eval_env, agent, turns=1)
            scores.append(score)
            # print(score, noise_score)
        # filename = "new-robust.txt" if opt.robust else "new-non-robust.txt"
        # with open(filename, 'a') as f:
        #     f.write(f"{[BrifEnvName[opt.EnvIdex], opt.train_std, opt.eval_std, delta] + [np.mean(scores), np.std(scores), np.quantile(scores, 0.9), np.quantile(scores, 0.1)]}\n")
        print(f"{[BrifEnvName[opt.EnvIdex], opt.train_std, opt.eval_std, delta] + [np.mean(scores), np.std(scores), np.quantile(scores, 0.9), np.quantile(scores, 0.1)]}\n")    
        env.close()
        eval_env.close()
            

    # 11. Otherwise, proceed with training
    else:
        total_steps = 0
        total_episode = 0

        while total_steps < opt.Max_train_steps:
            # (a) Reset environment with incremented seed
            state, info = env.reset(seed=env_seed)
            env_seed += 1
            total_episode += 1
            done = False

            # (b) Interact with environment until episode finishes
            while not done:
                # Random exploration for first 5 episodes (each episode is up to max_e_steps)
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
                # if (total_steps < 50 * opt.max_e_steps) and (total_steps % opt.update_every == 0):
                #     agent.r_mean = torch.max(agent.replay_buffer.r[:agent.replay_buffer.ptr]), torch.min(agent.replay_buffer.r[:agent.replay_buffer.ptr])
                #     print(agent.r_mean)
                    
                if (total_steps >= 50 * opt.max_e_steps) and (total_steps % opt.update_every == 0):
                    printer = False
                    if total_steps % 1000 == 0:
                        printer = True
                    for i in range(opt.update_every):
                        if i % opt.robust_update_every == 0:
                            agent.train(agent.robust, printer)
                        else:
                            agent.train(False, printer)
                        # agent.train(agent.robust, False)
                        printer = False
                    
                    if opt.robust: 
                       agent.delta *= 0.999
                       
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
                if total_steps % opt.save_interval == 0:
                    agent.save(BrifEnvName[opt.EnvIdex])
        
        # 11.5 Compute score of 20 episodes
        eval_num = 20
        print(f"Train finished. Now generate scores of {eval_num} episodes.")
        scores = []
        for _ in range(eval_num):
            score = evaluate_policy(eval_env, agent, turns=1)
            scores.append(score)
        filename = "robust.txt" if opt.robust else "non-robust.txt"
        with open(filename, 'a') as f:
              f.write(f"{[BrifEnvName[opt.EnvIdex], opt.train_std, opt.eval_std] + scores}\n")

        # 11.55 Save model at last
        agent.save(BrifEnvName[opt.EnvIdex])

        # 12. Close environments after training
        env.close()
        eval_env.close()

    
if __name__ == '__main__':
    '''Hyperparameter Setting'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda', help='running device: cuda or cpu')
    parser.add_argument('--EnvIdex', type=int, default=0, help='PV1, Lch_Cv2, Humanv4, HCv4, BWv3, BWHv3, CRv3')
    parser.add_argument('--write', type=str2bool, default=False, help='Use SummaryWriter to record the training')
    parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
    parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
    #parser.add_argument('--ModelIdex', type=int, default=100, help='which model to load')

    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--Max_train_steps', type=int, default=int(5e5), help='Max training steps')
    parser.add_argument('--save_interval', type=int, default=int(1e4), help='Model saving interval, in steps.')
    parser.add_argument('--eval_interval', type=int, default=int(2e3), help='Model evaluating interval, in steps.')
    parser.add_argument('--update_every', type=int, default=50, help='Training Fraquency, in stpes')
    parser.add_argument('--robust_update_every', type=int, default=2, help='Training Fraquency, in stpes')

    parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
    parser.add_argument('--net_width', type=int, default=256, help='Hidden net width')
    parser.add_argument('--net_layer', type=int, default=1, help='Hidden net layers')
    parser.add_argument('--a_lr', type=float, default=5e-3, help='Learning rate of actor')
    parser.add_argument('--c_lr', type=float, default=5e-3, help='Learning rate of critic')
    parser.add_argument('--b_lr', type=float, default=3e-4, help='Learning rate of dual-form optimization')
    parser.add_argument('--r_lr', type=float, default=3e-5, help='Learning rate of reward net')
    parser.add_argument('--batch_size', type=int, default=256, help='batch_size of training')
    parser.add_argument('--alpha', type=float, default=0.12, help='Entropy coefficient')
    parser.add_argument('--adaptive_alpha', type=str2bool, default=True, help='Use adaptive_alpha or Not')
    
    parser.add_argument('--reg_coef', type=float, default=0.0, help='Regulator of Network Parameters')
    parser.add_argument('--reward_adapt', type=bool, default=True, help='Reward adaptation')
    parser.add_argument('--robust', type=bool, default=False, help='Robust policy')
    parser.add_argument('--train_noise', type=bool, default=False, help='Train Env is Noisy')
    parser.add_argument('--train_std', type=float, default=1.0, help='Standard Deviation of Train Env Reward')
    parser.add_argument('--eval_noise', type=bool, default=False, help='Evaluation Env is Noisy')
    parser.add_argument('--eval_std', type=float, default=1.0, help='Standard Deviation of Eval Env Reward')
    opt = parser.parse_args()
    opt.device = torch.device(opt.device) # from str to torch.device

    print(opt)
    main(opt)
    
    # Pen step 5e4
    # LLd step 250k 2.5e5
    # Human about 400k