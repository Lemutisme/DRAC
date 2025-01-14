import torch.optim.adam
from utils import build_net, evaluate_policy, str2bool, discretize

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.nn.utils.rnn import pad_sequence
import copy
from datetime import datetime
import gymnasium as gym
import os, shutil
import argparse
from collections import defaultdict


class Policy_Net(nn.Module):
    def __init__(self, state_dim, action_dim, hid_shape):
        super(Policy_Net, self).__init__()
        layers = [state_dim] + list(hid_shape) + [action_dim]
        self.P = build_net(layers, nn.ReLU, nn.Identity)

    def forward(self, s):
        logits = self.P(s)
        probs = F.softmax(logits, dim=1)
        return probs

class Double_Q_Net(nn.Module):
    def __init__(self, state_dim, action_dim, hid_shape):
        super(Double_Q_Net, self).__init__()
        layers = [state_dim] + list(hid_shape) + [action_dim]

        self.Q1 = build_net(layers, nn.ReLU, nn.Identity)
        self.Q2 = build_net(layers, nn.ReLU, nn.Identity)

    def forward(self, s):
        q1 = self.Q1(s)
        q2 = self.Q2(s)
        return q1,q2

class ReplayBuffer(object):
    def __init__(self, state_dim, discrete, bins, max_size, device):
        self.state_dim = state_dim
        self.discrete = discrete
        self.bins = bins
        self.max_size = max_size
        self.device = device
        self.ptr = 0
        self.size = 0
        self.grid_made = False

        self.s = torch.zeros((max_size, state_dim),dtype=torch.float,device=self.device)
        self.a = torch.zeros((max_size, 1),dtype=torch.long,device=self.device)
        self.r = torch.zeros((max_size, 1),dtype=torch.float,device=self.device)
        self.s_next = torch.zeros((max_size, state_dim),dtype=torch.float,device=self.device)
        self.dw = torch.zeros((max_size, 1),dtype=torch.bool,device=self.device)
        self.r_dic= defaultdict(list)
        self.s_dic = defaultdict(list)

    def add(self, s, a, r, s_next, dw):
        if self.discrete and self.grid_made:
            s = torch.from_numpy(s).to(self.device)
            s = discretize(s, self.state_grid)
            self.s[self.ptr] = s
            a = torch.tensor([a], device=self.device)
            self.a[self.ptr] = a
            self.r[self.ptr] = r
            s_next = torch.from_numpy(s_next).to(self.device)
            s_next = discretize(s_next, self.state_grid)
            self.s_next[self.ptr] = s_next
            self.dw[self.ptr] = dw
            
            sa = torch.concat((s, a))
            sa_tuple = tuple(sa.tolist())
            self.r_dic[sa_tuple] += [r]
            # self.s_dic[sa] += [s_next]
        else:
            self.s[self.ptr] = torch.from_numpy(s).to(self.device)
            self.a[self.ptr] = a
            self.r[self.ptr] = r
            self.s_next[self.ptr] = torch.from_numpy(s_next).to(self.device)
            self.dw[self.ptr] = dw

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample(self, batch_size):
        ind = torch.randint(0, self.size, device=self.device, size=(batch_size,))
        return self.s[ind], self.a[ind], self.r[ind], self.s_next[ind], self.dw[ind]
    
    def make_grid(self):
        # To make finer discretization, estimate upper and lower bound from exploration.
        obs_high = torch.max(self.s, dim = 0).values
        obs_low = torch.min(self.s, dim = 0).values
        self.state_grid = [torch.linspace(obs_low[dim], obs_high[dim], self.bins + 1, device=self.device) for dim in range(self.state_dim)]
        self.grid_made = True
        print("Grid made.")
        # Repalce the states with approximation.
        for i in range(self.ptr):
            s, s_next = self.s[i], self.s_next[i]
            s, s_next = discretize(self.s[i], self.state_grid), discretize(s_next, self.state_grid)
            self.s[i], self.s_next[i] = s, s_next
            sa = torch.concat((s.flatten(), self.a[i]))
            sa_tuple = tuple(sa.tolist())
            self.r_dic[sa_tuple] += [self.r[i]]

    def empirical(self, s, a):
        result = []
        for i in range(s.shape[0]):
            sa = torch.concat((s[i], a[i]))
            sa_tuple = tuple(sa.tolist())
            if len(self.r_dic[sa_tuple]) == 0:
                print('not seen before.\n')
            result.append(torch.tensor(self.r_dic[sa_tuple]).to(self.device))
        return result
        # return pad_sequence(result, batch_first=True).to(self.device)

class SACD_agent():
    def __init__(self, **kwargs):
        # Init hyperparameters for agent, just like "self.gamma = opt.gamma, self.lambd = opt.lambd, ..."
        self.__dict__.update(kwargs)
        self.tau = 0.005
        self.H_mean = 0
        self.replay_buffer = ReplayBuffer(self.state_dim, self.discrete, self.bins, max_size=int(1e6), device=self.device)

        self.actor = Policy_Net(self.state_dim, self.action_dim, self.hid_shape).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)

        self.q_critic = Double_Q_Net(self.state_dim, self.action_dim, self.hid_shape).to(self.device)
        self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=self.lr)
        self.q_critic_target = copy.deepcopy(self.q_critic)
        for p in self.q_critic_target.parameters(): p.requires_grad = False

        self.beta = torch.ones((self.batch_size, 1), requires_grad=True, device=self.device)
        self.beta_optimizer = torch.optim.Adam([self.beta], lr=self.lr)

        if self.adaptive_alpha:
            # We use 0.6 because the recommended 0.98 will cause alpha explosion.
            self.target_entropy = 0.6 * (-np.log(1 / self.action_dim))  # H(discrete)>0
            self.log_alpha = torch.tensor(np.log(self.alpha), dtype=float, requires_grad=True, device=self.device)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.lr)

    def select_action(self, state, deterministic):
        with torch.no_grad():
            state = torch.FloatTensor(state[np.newaxis,:]).to(self.device) #from (s_dim,) to (1, s_dim)
            probs = self.actor(state)
            if deterministic:
                a = probs.argmax(-1).item()
            else:
                a = Categorical(probs).sample().item()
            return a

    def train(self):
        s, a, r, s_next, dw = self.replay_buffer.sample(self.batch_size)

        #------------------------------------------ Train Critic ----------------------------------------#
        '''Compute the target soft Q value'''
        with torch.no_grad():
            next_probs = self.actor(s_next) #[b,a_dim]
            next_log_probs = torch.log(next_probs+1e-8) #[b,a_dim]
            next_q1_all, next_q2_all = self.q_critic_target(s_next)  # [b,a_dim]
            min_next_q_all = torch.min(next_q1_all, next_q2_all)
            v_next = torch.sum(next_probs * (min_next_q_all - self.alpha * next_log_probs), dim=1, keepdim=True) # [b,1]
            # target_Q = r + (~dw) * self.gamma * v_next
        
        r_opt = r
        if self.robust:
            r_history = self.replay_buffer.empirical(s,a)
            for _ in range(5):
                loss1 = torch.zeros((self.batch_size, 1), device=self.device)
                for i in range(self.batch_size):
                    loss1[i] = self.beta[i] * torch.log(torch.mean(torch.exp(r_history[i]/self.beta[i]), dim=0, keepdim=True)) + self.beta[i] * self.delta
                self.beta_optimizer.zero_grad()
                loss1.sum().backward()
                self.beta_optimizer.step()
            r_opt = -loss1.detach()
            
        target_Q = r_opt + (~dw) * self.gamma * v_next


        '''Update soft Q net'''
        q1_all, q2_all = self.q_critic(s) #[b,a_dim]
        q1, q2 = q1_all.gather(1, a), q2_all.gather(1, a) #[b,1]
        q_loss = F.mse_loss(q1, target_Q) + F.mse_loss(q2, target_Q)
        self.q_critic_optimizer.zero_grad()
        q_loss.backward()
        self.q_critic_optimizer.step()

        #------------------------------------------ Train Actor ----------------------------------------#
        probs = self.actor(s) #[b,a_dim]
        log_probs = torch.log(probs + 1e-8) #[b,a_dim]
        with torch.no_grad():
            q1_all, q2_all = self.q_critic(s)  #[b,a_dim]
        min_q_all = torch.min(q1_all, q2_all)

        a_loss = torch.sum(probs * (self.alpha*log_probs - min_q_all), dim=1, keepdim=False) #[b,]

        self.actor_optimizer.zero_grad()
        a_loss.mean().backward()
        self.actor_optimizer.step()

        #------------------------------------------ Train Alpha ----------------------------------------#
        if self.adaptive_alpha:
            with torch.no_grad():
                self.H_mean = -torch.sum(probs * log_probs, dim=1).mean()
            alpha_loss = self.log_alpha * (self.H_mean - self.target_entropy)

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp().item()

        #------------------------------------------ Update Target Net ----------------------------------#
        for param, target_param in zip(self.q_critic.parameters(), self.q_critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, timestep, EnvName):
        torch.save(self.actor.state_dict(), f"./model/sacd_actor_{timestep}_{EnvName}.pth")
        torch.save(self.q_critic.state_dict(), f"./model/sacd_critic_{timestep}_{EnvName}.pth")


    def load(self, timestep, EnvName):
        self.actor.load_state_dict(torch.load(f"./model/sacd_actor_{timestep}_{EnvName}.pth", map_location=self.device))
        self.q_critic.load_state_dict(torch.load(f"./model/sacd_critic_{timestep}_{EnvName}.pth", map_location=self.device))

def main(opt):
    """
    Main function to run SACD training or evaluation on CartPole-v1 or LunarLander-v2.
    """

    # 1. Define environment names and abbreviations
    EnvName = ['CartPole-v1', 'LunarLander-v2']
    BriefEnvName = ['CPV1', 'LLdV2']

    # 2. Create training and evaluation environments
    env = gym.make(
        EnvName[opt.EnvIdex], 
        render_mode="human" if opt.render else None
    )
    eval_env = gym.make(EnvName[opt.EnvIdex])

    # 3. Extract environment properties and store them in opt
    opt.state_dim = env.observation_space.shape[0]
    # opt.state_high = [env.observation_space.high[dim] if not math.isinf(env.observation_space.high[dim]) else 1e6 for dim in range(opt.state_dim)]
    # opt.state_low = [env.observation_space.low[dim] if not math.isinf(-env.observation_space.low[dim]) else -1e6 for dim in range(opt.state_dim)]
    # opt.state_grid = [np.linspace(opt.state_low[dim], opt.state_high[dim], opt.bins + 1)[1:-1] for dim in range(opt.state_dim)]
    opt.action_dim = env.action_space.n
    opt.max_e_steps = env._max_episode_steps

    # 4. Seed everything for reproducibility
    env_seed = opt.seed
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Random Seed: {opt.seed}")
    print(
        f"Algorithm: SACD  "
        f"Env: {BriefEnvName[opt.EnvIdex]}  "
        f"state_dim: {opt.state_dim}  "
        f"action_dim: {opt.action_dim}  "
        f"Random Seed: {opt.seed}  "
        f"max_e_steps: {opt.max_e_steps} "
        f"Device: {opt.device}\n"
    )

    # 5. Optionally set up TensorBoard logging
    writer = None
    if opt.write:
        from torch.utils.tensorboard import SummaryWriter
        timenow = str(datetime.now())[:-10]     # e.g. 2025-01-10 17:45
        timenow = ' ' + timenow[:13] + '_' + timenow[-2:]  # e.g. ' 2025-01-10_45'
        write_path = f"runs/SACD_{BriefEnvName[opt.EnvIdex]}{timenow}"
        if os.path.exists(write_path):
            shutil.rmtree(write_path)
        writer = SummaryWriter(log_dir=write_path)

    # 6. Ensure a directory for model saving
    if not os.path.exists('model'):
        os.mkdir('model')

    # 7. Initialize the SACD agent
    agent = SACD_agent(**vars(opt))  # Convert argparse Namespace to dict

    # 8. Optionally load a saved model
    if opt.Loadmodel:
        agent.load(opt.ModelIdex, BriefEnvName[opt.EnvIdex])

    # 9. If rendering is requested, run an infinite evaluation loop
    if opt.render:
        while True:
            score = evaluate_policy(env, agent, turns=1)
            print(
                f"EnvName: {BriefEnvName[opt.EnvIdex]}, "
                f"Seed: {opt.seed}, "
                f"Score: {score}"
            )

    # 10. Otherwise, proceed with training
    else:
        total_steps = 0
        
        # 10-a. Training until reaching Max_train_steps
        while total_steps < opt.Max_train_steps:
            state, info = env.reset(seed=env_seed)
            env_seed += 1
            done = False

            # (i) Interact & train for one episode
            while not done:
                # (A) E-greedy exploration for initial steps
                if total_steps < opt.random_steps:
                    action = env.action_space.sample()
                else:
                    action = agent.select_action(state, deterministic=False)

                # (B) Step the environment
                next_state, reward, dw, tr, info = env.step(action)
                done = (dw or tr)

                # (C) LunarLander-specific reward shaping
                if opt.EnvIdex == 1 and reward <= -100:
                    reward = -10

                # (D) Store experience in replay buffer
                agent.replay_buffer.add(state, action, reward, next_state, dw)
                state = next_state

                # (D+) Before trainging start, discretize the obervation space based on experience
                if (total_steps == opt.random_steps) and agent.discrete:
                    agent.replay_buffer.make_grid()

                # (E) Periodic training: train `opt.update_every` times every `opt.update_every` steps
                if (total_steps >= opt.random_steps) and (total_steps % opt.update_every == 0):
                    for _ in range(opt.update_every):
                        agent.train()

                # (F) Evaluation & logging
                if total_steps % opt.eval_interval == 0:
                    score = evaluate_policy(eval_env, agent, turns=3)
                    if writer is not None:
                        writer.add_scalar('ep_r', score, global_step=total_steps)
                        writer.add_scalar('alpha', agent.alpha, global_step=total_steps)
                        writer.add_scalar('H_mean', agent.H_mean, global_step=total_steps)
                    print(
                        f"EnvName: {BriefEnvName[opt.EnvIdex]}, "
                        f"Seed: {opt.seed}, "
                        f"Steps: {int(total_steps / 1000)}k, "
                        f"Score: {int(score)}"
                    )

                total_steps += 1

                # (G) Save model periodically
                if total_steps % opt.save_interval == 0:
                    agent.save(int(total_steps / 1000), BriefEnvName[opt.EnvIdex])

        # 11. Close environments after training
        env.close()
        eval_env.close()

if __name__ == '__main__':
    '''Hyperparameter Setting'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda', help='running device: cuda or cpu')
    parser.add_argument('--EnvIdex', type=int, default=0, help='CP-v1, LLd-v2')
    parser.add_argument('--write', type=str2bool, default=False, help='Use SummaryWriter to record the training')
    parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
    parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
    parser.add_argument('--ModelIdex', type=int, default=50, help='which model to load')

    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--Max_train_steps', type=int, default=1e5, help='Max training steps')
    parser.add_argument('--save_interval', type=int, default=1e5, help='Model saving interval, in steps.')
    parser.add_argument('--eval_interval', type=int, default=2e3, help='Model evaluating interval, in steps.')
    parser.add_argument('--random_steps', type=int, default=2e4, help='steps for random policy to explore')
    parser.add_argument('--update_every', type=int, default=50, help='training frequency')
    parser.add_argument('--bins', type=int, default=100, help='grid number')

    parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
    parser.add_argument('--hid_shape', type=list, default=[200,200], help='Hidden net shape')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--alpha', type=float, default=0.2, help='init alpha')
    parser.add_argument('--adaptive_alpha', type=str2bool, default=True, help='Use adaptive alpha turning')
    parser.add_argument('--delta', type=float, default=0.0, help='Pertubation Distance')
    parser.add_argument('--robust', type=bool, default=False, help='robust or non-robust policy')
    parser.add_argument('--discrete', type=bool, default=True, help='discretize the state space')
    opt = parser.parse_args()
    opt.device = torch.device(opt.device) # from str to torch.device
    print(opt)

    main(opt)