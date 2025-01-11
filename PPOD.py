
from utils import evaluate_policy_PPOD, str2bool

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import copy
from datetime import datetime
import gymnasium as gym
import os, shutil
import argparse

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, net_width):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, net_width)
        self.l2 = nn.Linear(net_width, net_width)
        self.l3 = nn.Linear(net_width, action_dim)

    def forward(self, state):
        n = torch.tanh(self.l1(state))
        n = torch.tanh(self.l2(n))
        return n

    def pi(self, state, softmax_dim = 0):
        n = self.forward(state)
        prob = F.softmax(self.l3(n), dim=softmax_dim)
        return prob

class Critic(nn.Module):
    def __init__(self, state_dim,net_width):
        super(Critic, self).__init__()

        self.C1 = nn.Linear(state_dim, net_width)
        self.C2 = nn.Linear(net_width, net_width)
        self.C3 = nn.Linear(net_width, 1)

    def forward(self, state):
        v = torch.relu(self.C1(state))
        v = torch.relu(self.C2(v))
        v = self.C3(v)
        return v

class PPO_discrete():
    def __init__(self, **kwargs):
        # Init hyperparameters for PPO agent, just like "self.gamma = opt.gamma, self.lambd = opt.lambd, ..."
        self.__dict__.update(kwargs)

        '''Build Actor and Critic'''
        self.actor = Actor(self.state_dim, self.action_dim, self.net_width).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic = Critic(self.state_dim, self.net_width).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

        '''Build Trajectory holder'''
        self.s_hoder = np.zeros((self.T_horizon, self.state_dim), dtype=np.float32)
        self.a_hoder = np.zeros((self.T_horizon, 1), dtype=np.int64)
        self.r_hoder = np.zeros((self.T_horizon, 1), dtype=np.float32)
        self.s_next_hoder = np.zeros((self.T_horizon, self.state_dim), dtype=np.float32)
        self.logprob_a_hoder = np.zeros((self.T_horizon, 1), dtype=np.float32)
        self.done_hoder = np.zeros((self.T_horizon, 1), dtype=np.bool_)
        self.dw_hoder = np.zeros((self.T_horizon, 1), dtype=np.bool_)

    def select_action(self, s, deterministic):
        s = torch.from_numpy(s).float().to(self.device)
        with torch.no_grad():
            pi = self.actor.pi(s, softmax_dim=0)
            if deterministic:
                a = torch.argmax(pi).item()
                return a, None
            else:
                m = Categorical(pi)
                a = m.sample().item()
                pi_a = pi[a].item()
                return a, pi_a

    def train(self):
        self.entropy_coef *= self.entropy_coef_decay #exploring decay
        '''Prepare PyTorch data from Numpy data'''
        s = torch.from_numpy(self.s_hoder).to(self.device)
        a = torch.from_numpy(self.a_hoder).to(self.device)
        r = torch.from_numpy(self.r_hoder).to(self.device)
        s_next = torch.from_numpy(self.s_next_hoder).to(self.device)
        old_prob_a = torch.from_numpy(self.logprob_a_hoder).to(self.device)
        done = torch.from_numpy(self.done_hoder).to(self.device)
        dw = torch.from_numpy(self.dw_hoder).to(self.device)

        ''' Use TD+GAE+LongTrajectory to compute Advantage and TD target'''
        with torch.no_grad():
            vs = self.critic(s)
            vs_ = self.critic(s_next)

            '''dw(dead and win) for TD_target and Adv'''
            deltas = r + self.gamma * vs_ * (~dw) - vs
            deltas = deltas.cpu().flatten().numpy()
            adv = [0]

            '''done for GAE'''
            for dlt, done in zip(deltas[::-1], done.cpu().flatten().numpy()[::-1]):
                advantage = dlt + self.gamma * self.lambd * adv[-1] * (~done)
                adv.append(advantage)
            adv.reverse()
            adv = copy.deepcopy(adv[0:-1])
            adv = torch.tensor(adv).unsqueeze(1).float().to(self.device)
            td_target = adv + vs
            if self.adv_normalization:
                adv = (adv - adv.mean()) / ((adv.std() + 1e-4))  #sometimes helps

        """PPO update"""
        #Slice long trajectopy into short trajectory and perform mini-batch PPO update
        optim_iter_num = int(math.ceil(s.shape[0] / self.batch_size))

        for _ in range(self.K_epochs):
            #Shuffle the trajectory, Good for training
            perm = np.arange(s.shape[0])
            np.random.shuffle(perm)
            perm = torch.LongTensor(perm).to(self.device)
            s, a, td_target, adv, old_prob_a = \
                s[perm].clone(), a[perm].clone(), td_target[perm].clone(), adv[perm].clone(), old_prob_a[perm].clone()

            '''mini-batch PPO update'''
            for i in range(optim_iter_num):
                index = slice(i * self.batch_size, min((i + 1) * self.batch_size, s.shape[0]))

                '''actor update'''
                prob = self.actor.pi(s[index], softmax_dim=1)
                entropy = Categorical(prob).entropy().sum(0, keepdim=True)
                prob_a = prob.gather(1, a[index])
                ratio = torch.exp(torch.log(prob_a) - torch.log(old_prob_a[index]))  # a/b == exp(log(a)-log(b))

                surr1 = ratio * adv[index]
                surr2 = torch.clamp(ratio, 1 - self.clip_rate, 1 + self.clip_rate) * adv[index]
                a_loss = -torch.min(surr1, surr2) - self.entropy_coef * entropy

                self.actor_optimizer.zero_grad()
                a_loss.mean().backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 40)
                self.actor_optimizer.step()

                '''critic update'''
                c_loss = (self.critic(s[index]) - td_target[index]).pow(2).mean()
                for name, param in self.critic.named_parameters():
                    if 'weight' in name:
                        c_loss += param.pow(2).sum() * self.l2_reg

                self.critic_optimizer.zero_grad()
                c_loss.backward()
                self.critic_optimizer.step()

    def put_data(self, s, a, r, s_next, logprob_a, done, dw, idx):
        self.s_hoder[idx] = s
        self.a_hoder[idx] = a
        self.r_hoder[idx] = r
        self.s_next_hoder[idx] = s_next
        self.logprob_a_hoder[idx] = logprob_a
        self.done_hoder[idx] = done
        self.dw_hoder[idx] = dw

    def save(self, episode):
        torch.save(self.critic.state_dict(), "./model/ppo_critic{}.pth".format(episode))
        torch.save(self.actor.state_dict(), "./model/ppo_actor{}.pth".format(episode))

    def load(self, episode):
        self.critic.load_state_dict(torch.load("./model/ppo_critic{}.pth".format(episode), map_location=self.device))
        self.actor.load_state_dict(torch.load("./model/ppo_actor{}.pth".format(episode), map_location=self.device))

def main(opt):
    """
    Main training function for discrete PPO on specified environments.
    """

    # 1. Define environment names and abbreviations
    env_names = ['CartPole-v1', 'LunarLander-v2']
    brief_env_names = ['CP-v1', 'LLd-v2']
    
    # 2. Create training and evaluation environments
    env = gym.make(
        env_names[opt.EnvIdex],
        render_mode="human" if opt.render else None
    )
    eval_env = gym.make(env_names[opt.EnvIdex])
    
    # 3. Extract and set environment properties in 'opt'
    opt.state_dim = env.observation_space.shape[0]
    opt.action_dim = env.action_space.n
    opt.max_e_steps = env._max_episode_steps
    
    # 4. Seed everything
    env_seed = opt.seed
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Random Seed: {opt.seed}")
    print(
        f"Env: {brief_env_names[opt.EnvIdex]}  "
        f"state_dim: {opt.state_dim}  "
        f"action_dim: {opt.action_dim}  "
        f"Random Seed: {opt.seed}  "
        f"max_e_steps: {opt.max_e_steps}\n"
    )
    
    # 5. TensorBoard for logging (optional)
    writer = None
    if opt.write:
        from torch.utils.tensorboard import SummaryWriter
        timenow = str(datetime.now())[:-10]     # e.g. 2025-01-10 17:45
        timenow = ' ' + timenow[:13] + '_' + timenow[-2:]  # e.g. ' 2025-01-10_45'
        write_path = f"runs/{brief_env_names[opt.EnvIdex]}{timenow}"
        
        if os.path.exists(write_path):
            shutil.rmtree(write_path)
        
        writer = SummaryWriter(log_dir=write_path)
    
    # 6. Ensure a directory exists for saving models
    if not os.path.exists('model'):
        os.mkdir('model')
    
    # 7. Create the PPO agent
    agent = PPO_discrete(**vars(opt))
    
    # 8. Load existing model if requested
    if opt.Loadmodel:
        agent.load(opt.ModelIdex)
    
    # 9. If render is enabled, enter an evaluation loop
    if opt.render:
        while True:
            episode_reward = evaluate_policy_PPOD(env, agent, turns=1)
            print(f"Env: {env_names[opt.EnvIdex]}, Episode Reward: {episode_reward}")
    else:
        # 10. Otherwise, start training
        traj_length = 0
        total_steps = 0
        
        while total_steps < opt.Max_train_steps:
            # (a) Reset environment each episode with an incremented seed
            state, info = env.reset(seed=env_seed)
            env_seed += 1
            done = False
            
            # (b) Interact and train for one episode
            while not done:
                # (i) Select action stochastically
                action, logprob = agent.select_action(state, deterministic=False)
                
                # (ii) Step the environment
                next_state, reward, dw, tr, info = env.step(action)
                
                # Reward shaping for LunarLander
                if reward <= -100:
                    reward = -30
                
                # Check for episode termination
                done = (dw or tr)
                
                # (iii) Store transition in the agent’s buffer
                agent.put_data(
                    state, action, reward, next_state,
                    logprob, done, dw, idx=traj_length
                )
                
                # (iv) Move to the next state
                state = next_state
                traj_length += 1
                total_steps += 1
                
                # (v) Train if we have enough experience
                if traj_length % opt.T_horizon == 0:
                    agent.train()
                    traj_length = 0
                
                # (vi) Evaluate periodically
                if total_steps % opt.eval_interval == 0:
                    score = evaluate_policy_PPOD(eval_env, agent, turns=3)
                    
                    if writer is not None:
                        writer.add_scalar('ep_r', score, global_step=total_steps)
                    
                    print(
                        f"EnvName: {env_names[opt.EnvIdex]}, "
                        f"Seed: {opt.seed}, "
                        f"Steps: {int(total_steps/1000)}k, "
                        f"Score: {score}"
                    )
                
                # (vii) Save model periodically
                if total_steps % opt.save_interval == 0:
                    agent.save(total_steps)
        
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
    parser.add_argument('--ModelIdex', type=int, default=300000, help='which model to load')

    parser.add_argument('--seed', type=int, default=209, help='random seed')
    parser.add_argument('--T_horizon', type=int, default=2048, help='lenth of long trajectory')
    parser.add_argument('--Max_train_steps', type=int, default=5e7, help='Max training steps')
    parser.add_argument('--save_interval', type=int, default=1e5, help='Model saving interval, in steps.')
    parser.add_argument('--eval_interval', type=int, default=5e3, help='Model evaluating interval, in steps.')

    parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
    parser.add_argument('--lambd', type=float, default=0.95, help='GAE Factor')
    parser.add_argument('--clip_rate', type=float, default=0.2, help='PPO Clip rate')
    parser.add_argument('--K_epochs', type=int, default=10, help='PPO update times')
    parser.add_argument('--net_width', type=int, default=64, help='Hidden net width')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--l2_reg', type=float, default=0, help='L2 regulization coefficient for Critic')
    parser.add_argument('--batch_size', type=int, default=64, help='lenth of sliced trajectory')
    parser.add_argument('--entropy_coef', type=float, default=0, help='Entropy coefficient of Actor')
    parser.add_argument('--entropy_coef_decay', type=float, default=0.99, help='Decay rate of entropy_coef')
    parser.add_argument('--adv_normalization', type=str2bool, default=False, help='Advantage normalization')
    opt = parser.parse_args()
    opt.device = torch.device(opt.device) # from str to torch.device
    print(opt)

    main(opt)
