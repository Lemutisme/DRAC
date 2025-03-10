from utils import evaluate_policy_PPO as evaluate_policy
from utils import Action_adapter_pos as Action_adapter
from utils import Reward_adapter, str2bool, build_net, register
from continuous_cartpole import register

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Beta

import copy
import gymnasium as gym
import os, shutil
import hydra
import argparse
from scipy.special import logsumexp

class BetaActor(nn.Module):
    def __init__(self, state_dim, action_dim, net_width):
        super(BetaActor, self).__init__()

        self.l1 = nn.Linear(state_dim, net_width)
        self.l2 = nn.Linear(net_width, net_width)
        self.alpha_head = nn.Linear(net_width, action_dim)
        self.beta_head = nn.Linear(net_width, action_dim)

    def forward(self, state):
        a = torch.tanh(self.l1(state))
        a = torch.tanh(self.l2(a))

        alpha = F.softplus(self.alpha_head(a)) + 1.0
        beta = F.softplus(self.beta_head(a)) + 1.0

        return alpha, beta

    def get_dist(self,state):
        alpha,beta = self.forward(state)
        dist = Beta(alpha, beta)
        return dist

    def deterministic_act(self, state):
        alpha, beta = self.forward(state)
        mode = (alpha) / (alpha + beta)
        return mode

class GaussianActor_musigma(nn.Module):
    def __init__(self, state_dim, action_dim, net_width):
        super(GaussianActor_musigma, self).__init__()

        self.l1 = nn.Linear(state_dim, net_width)
        self.l2 = nn.Linear(net_width, net_width)
        self.mu_head = nn.Linear(net_width, action_dim)
        self.sigma_head = nn.Linear(net_width, action_dim)

    def forward(self, state):
        a = torch.tanh(self.l1(state))
        a = torch.tanh(self.l2(a))
        mu = torch.sigmoid(self.mu_head(a))
        sigma = F.softplus( self.sigma_head(a) )
        return mu, sigma

    def get_dist(self, state):
        mu, sigma = self.forward(state)
        dist = Normal(mu,sigma)
        return dist

    def deterministic_act(self, state):
        mu, sigma = self.forward(state)
        return mu

class GaussianActor_mu(nn.Module):
    def __init__(self, state_dim, action_dim, net_width, log_std=0):
        super(GaussianActor_mu, self).__init__()

        self.l1 = nn.Linear(state_dim, net_width)
        self.l2 = nn.Linear(net_width, net_width)
        self.mu_head = nn.Linear(net_width, action_dim)
        self.mu_head.weight.data.mul_(0.1)
        self.mu_head.bias.data.mul_(0.0)

        self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * log_std)

    def forward(self, state):
        a = torch.relu(self.l1(state))
        a = torch.relu(self.l2(a))
        mu = torch.sigmoid(self.mu_head(a))
        return mu

    def get_dist(self,state):
        mu = self.forward(state)
        action_log_std = self.action_log_std.expand_as(mu)
        action_std = torch.exp(action_log_std)

        dist = Normal(mu, action_std)
        return dist

    def deterministic_act(self, state):
        return self.forward(state)

class Critic(nn.Module):
    def __init__(self, state_dim, hid_shape, hid_layer):
        super(Critic, self).__init__()

        layers = [state_dim] + list(hid_shape) * hid_layer + [1]
        self.cnet = build_net(layers, nn.Tanh, nn.Identity)

    def forward(self, state):
        return self.cnet(state)	

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
            

class PPO_agent(object):
    def __init__(self, **kwargs):
        # Init hyperparameters for PPO agent, just like "self.gamma = opt.gamma, self.lambd = opt.lambd, ..."
        self.__dict__.update(kwargs)

        # Choose distribution for the actor
        if self.Distribution == 'Beta':
            self.actor = BetaActor(self.state_dim, self.action_dim, self.net_width).to(self.device)
        elif self.Distribution == 'GS_ms':
            self.actor = GaussianActor_musigma(self.state_dim, self.action_dim, self.net_width).to(self.device)
        elif self.Distribution == 'GS_m':
            self.actor = GaussianActor_mu(self.state_dim, self.action_dim, self.net_width).to(self.device)
        else: 
            print('Dist Error')
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.a_lr)

        # Build Critic
        self.critic = Critic(self.state_dim, (self.net_width, self.net_width), self.net_layer).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.c_lr)

        # Build Reward Net and Variable for OPT
        if self.robust:    
            print('This is a robust policy.\n')
            self.transition = TransitionVAE(self.state_dim, self.action_dim, self.state_dim, (self.net_width, self.net_width), self.net_layer).to(self.device)
            self.trans_optimizer = torch.optim.Adam(self.transition.parameters(), lr=self.r_lr)
            
            # self.log_beta = nn.Parameter(torch.ones((self.batch_size,1), requires_grad=True, device=self.device) * 1.0)
            # self.beta_optimizer = torch.optim.Adam([self.log_beta], lr=self.b_lr)
            
            self.g = dual(self.state_dim, self.action_dim, (self.net_width, self.net_width), self.net_layer).to(self.device)
            self.g_optimizer = torch.optim.Adam(self.g.parameters(), lr=self.g_lr)

        # Build Trajectory holder
        self.s_holder = np.zeros((self.T_horizon, self.state_dim),dtype=np.float32)
        self.a_holder = np.zeros((self.T_horizon, self.action_dim),dtype=np.float32)
        self.r_holder = np.zeros((self.T_horizon, 1),dtype=np.float32)
        self.s_next_holder = np.zeros((self.T_horizon, self.state_dim),dtype=np.float32)
        self.logprob_a_holder = np.zeros((self.T_horizon, self.action_dim),dtype=np.float32)
        self.done_holder = np.zeros((self.T_horizon, 1),dtype=np.bool_)
        self.dw_holder = np.zeros((self.T_horizon, 1),dtype=np.bool_)

    def select_action(self, state, deterministic):
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            if deterministic:
                # only used when evaluate the policy. Making the performance more stable
                a = self.actor.deterministic_act(state)
                return a.cpu().numpy()[0], None  # action is in shape (adim, 0)
            else:
                # only used when interact with the env
                dist = self.actor.get_dist(state)
                a = dist.sample()
                a = torch.clamp(a, 0, 1)
                logprob_a = dist.log_prob(a).cpu().numpy().flatten()
                return a.cpu().numpy()[0], logprob_a # both are in shape (adim, 0)

    def vae_loss(self, s_next, s_next_recon, mu, logvar):
        recon_loss = F.mse_loss(s_next_recon, s_next, reduction='sum')
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_div

    def dual_func_g(self, s, a, s_next):
        size = s_next.shape[1]
        dual_sa = self.g(s,a)
        return - dual_sa * (torch.logsumexp(-self.critic(s_next).squeeze(-1)/dual_sa, dim=1, keepdim=True) - math.log(size)) - dual_sa * self.delta  

    def dual_func_beta(self, s_next, beta):
        # Jointly optimize, in tensor
        size = s_next.shape[1]
        return - beta * (torch.logsumexp(-self.critic(s_next).squeeze(-1)/beta, dim=1, keepdim=True) - math.log(size)) - beta * self.delta     

    def dual_func_ind(self, s_next, beta):
        # Independently optimize, in np.array
        size = s_next.shape[-1]
        v_next = self.critic(s_next)
        v_next = v_next.cpu().numpy()
        return - beta * (logsumexp(-v_next/beta) - math.log(size)) - beta * self.delta    

    def train(self, printer, writer, step):
        self.entropy_coef *= self.entropy_coef_decay

        '''Prepare PyTorch data from Numpy data'''
        s = torch.from_numpy(self.s_holder).to(self.device)
        a = torch.from_numpy(self.a_holder).to(self.device)
        r = torch.from_numpy(self.r_holder).to(self.device)
        s_next = torch.from_numpy(self.s_next_holder).to(self.device)
        logprob_a = torch.from_numpy(self.logprob_a_holder).to(self.device)
        done = torch.from_numpy(self.done_holder).to(self.device)
        dw = torch.from_numpy(self.dw_holder).to(self.device)

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
            
            with torch.no_grad():
                s_next_sample = self.transition.sample(s, a, 200)

            for _ in range(5):
                opt_loss = -self.dual_func_g(s, a, s_next_sample)
                self.g_optimizer.zero_grad()
                opt_loss.mean().backward()
                if printer:
                    print(opt_loss.mean().item())    
                self.g_optimizer.step() 

            vs_opt = self.dual_func_g(s, a, s_next_sample) 

        ''' Use TD+GAE+LongTrajectory to compute Advantage and TD target'''
        with torch.no_grad():
            vs = self.critic(s)
            vs_ = self.critic(s_next)

            '''dw for TD_target and Adv'''
            if self.robust:
                deltas = r + self.gamma * vs_opt * (~dw) - vs
            else:
                deltas = r + self.gamma * vs_ * (~dw) - vs
            deltas = deltas.cpu().flatten().numpy()
            adv = [0]

            '''done for GAE'''
            for dlt, mask in zip(deltas[::-1], done.cpu().flatten().numpy()[::-1]):
                advantage = dlt + self.gamma * self.lambd * adv[-1] * (~mask)
                adv.append(advantage)
            adv.reverse()
            adv = copy.deepcopy(adv[0:-1])
            adv = torch.tensor(adv).unsqueeze(1).float().to(self.device)
            td_target = adv + vs
            adv = (adv - adv.mean()) / ((adv.std()+1e-4))  #sometimes helps

        """Slice long trajectopy into short trajectory and perform mini-batch PPO update"""
        a_optim_iter_num = int(math.ceil(s.shape[0] / self.a_optim_batch_size))
        c_optim_iter_num = int(math.ceil(s.shape[0] / self.c_optim_batch_size))
        for i in range(self.K_epochs):

            #Shuffle the trajectory, Good for training
            perm = np.arange(s.shape[0])
            np.random.shuffle(perm)
            perm = torch.LongTensor(perm).to(self.device)
            s, a, td_target, adv, logprob_a = \
                s[perm].clone(), a[perm].clone(), td_target[perm].clone(), adv[perm].clone(), logprob_a[perm].clone()

            '''update the actor'''
            for i in range(a_optim_iter_num):
                index = slice(i * self.a_optim_batch_size, min((i + 1) * self.a_optim_batch_size, s.shape[0]))
                distribution = self.actor.get_dist(s[index])
                dist_entropy = distribution.entropy().sum(1, keepdim=True)
                logprob_a_now = distribution.log_prob(a[index])
                ratio = torch.exp(logprob_a_now.sum(1,keepdim=True) - logprob_a[index].sum(1,keepdim=True))  # a/b == exp(log(a)-log(b))

                surr1 = ratio * adv[index]
                surr2 = torch.clamp(ratio, 1 - self.clip_rate, 1 + self.clip_rate) * adv[index]
                a_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy

                self.actor_optimizer.zero_grad()
                a_loss.mean().backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 40)
                self.actor_optimizer.step()
                if writer:
                    writer.add_scalar('a_loss', a_loss.mean(), global_step=step)

            '''update the critic'''
            for i in range(c_optim_iter_num):
                index = slice(i * self.c_optim_batch_size, min((i + 1) * self.c_optim_batch_size, s.shape[0]))
                c_loss = (self.critic(s[index]) - td_target[index]).pow(2).mean()
                for name,param in self.critic.named_parameters():
                    if 'weight' in name:
                        c_loss += param.pow(2).sum() * self.l2_reg

                self.critic_optimizer.zero_grad()
                c_loss.backward()
                self.critic_optimizer.step()
                if writer:
                    writer.add_scalar('c_loss', c_loss, global_step=step)

    def put_data(self, s, a, r, s_next, logprob_a, done, dw, idx):
        self.s_holder[idx] = s
        self.a_holder[idx] = a
        self.r_holder[idx] = r
        self.s_next_holder[idx] = s_next
        self.logprob_a_holder[idx] = logprob_a
        self.done_holder[idx] = done
        self.dw_holder[idx] = dw

    def save(self):
        params = f"{self.std}_{self.robust}"
        torch.save(self.actor.state_dict(), f"{dir}/actor_{params}.pth")
        torch.save(self.critic.state_dict(), f"{dir}/q_critic_{params}.pth")

    def load(self, params):
        self.actor.load_state_dict(torch.load(f"{dir}/actor_{params}.pth", map_location=self.device, weights_only=True))
        self.critic.load_state_dict(torch.load(f"{dir}/q_critic_{params}.pth", map_location=self.device, weights_only=True))

#@hydra.main(version_base=None, config_path="config/PPO", config_name="base")
def main(opt):
    # 1. Define environment names and their abbreviations
    EnvName = [
        'Pendulum-v1',
        "ContinuousCartPole",
        'LunarLanderContinuous-v3',
        'Humanoid-v4',
        'HalfCheetah-v4',
        'BipedalWalker-v3',
        'BipedalWalkerHardcore-v3'
    ]
    BrifEnvName = [
        'PV1',
        'CPV0',
        'LLdV2',
        'Humanv4',
        'HCv4',
        'BWv3',
        'BWHv3'
    ]
    
    # 2. Build Training and Evaluation Environments
    env = gym.make(EnvName[opt.EnvIdex])
    
    if not opt.noise:
        eval_env =  gym.make(EnvName[opt.EnvIdex])
    else:
        if opt.EnvIdex == 0:
            eval_env = gym.make("CustomPendulum-v1", std=opt.std) # Add noise when updating angle
        elif opt.EnvIdex == 1:
            eval_env = gym.make("CustomCartPole", std=opt.std) # Add noise when updating angle


    # 3. Extract environment/state/action info
    opt.state_dim = env.observation_space.shape[0]
    opt.action_dim = env.action_space.shape[0]
    opt.max_action = float(env.action_space.high[0])
    opt.max_steps = env._max_episode_steps
    
    print(
        f"Env: {EnvName[opt.EnvIdex]}  "
        f"state_dim: {opt.state_dim}  "
        f"action_dim: {opt.action_dim}  "
        f"max_action: {opt.max_action}  "
        f"min_action: {env.action_space.low[0]}  "
        f"max_steps: {opt.max_steps}"
    )

    # 4. Seed everything
    env_seed = opt.seed
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random Seed: {opt.seed}")

    # 5. Set up TensorBoard (optional)
    writer = None
    if opt.write:
        from torch.utils.tensorboard import SummaryWriter
        # timenow = str(datetime.now())[:-10]  # e.g. 2022-11-24 17:45
        # timenow = ' ' + timenow[:13] + '_' + timenow[-2:]
        writepath = f"runs/PPO/{BrifEnvName[opt.EnvIdex]}"
        if opt.noise:
            writepath += f"_Noise_{opt.std}"
        if opt.robust:
            writepath += f"_Robust"
        if os.path.exists(writepath):
            shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)
    
    # 6. (Optional) Adjust learning rates for Beta distribution — commented-out example
    # if Dist[distnum] == 'Beta':
    #     kwargs["a_lr"] *= 2
    #     kwargs["c_lr"] *= 4

    # 7. Ensure a directory for saving models
    opt.dir = f'./models/PPO_model/{BrifEnvName[opt.EnvIdex]}'
    if not os.path.exists(opt.dir):
        os.mkdir(opt.dir)

    # 8. Create the PPO agent
    agent = PPO_agent(**vars(opt))  # Convert opt to dict and pass to PPO_agent

    # 9. Load existing model if requested
    if opt.load_model:
        params = f"{opt.std}_{opt.robust}"
        agent.load(params)

    # 10. If rendering is enabled, just evaluate in a loop
    if opt.render:
        while True:
            ep_r = evaluate_policy(env, agent, opt.max_action, turns=1)
            print(f"Env: {EnvName[opt.EnvIdex]}, Episode Reward: {ep_r}")
    
    # 11. If evaluating only, print result
    elif opt.eval_model:
        eval_num = 100
        print(f"Evaluate {eval_num} policies.")
        scores = []
        for _ in range(eval_num):
            score = evaluate_policy(eval_env, agent, opt.max_action, turns=1)
            scores.append(score)
        # filename = "new-robust.txt" if opt.robust else "new-non-robust.txt"
        # with open(filename, 'a') as f:
        #     f.write(f"{[BrifEnvName[opt.EnvIdex], opt.train_std, opt.eval_std, delta] + [np.mean(scores), np.std(scores), np.quantile(scores, 0.9), np.quantile(scores, 0.1)]}\n")
        print(f"{[BrifEnvName[opt.EnvIdex]] + [np.mean(scores), np.std(scores), np.quantile(scores, 0.9), np.quantile(scores, 0.1)]}\n")    

    # 12. Otherwise, proceed with training
    else:
        traj_length = 0
        total_steps = 0

        while total_steps < opt.Max_train_steps:
            # Reset environment each episode with an incremented seed 
            s, info = env.reset(seed=env_seed)
            env_seed += 1
            done = False

            # 11-a. Interact & train for one episode
            while not done:
                # (i) Select action (stochastic when training)
                a, logprob_a = agent.select_action(s, deterministic=False)

                # (ii) Convert action range if needed
                act = Action_adapter(a, opt.max_action)  # [0,1] -> [-max_action, max_action]

                # (iii) Step environment
                s_next, r, dw, tr, info = env.step(act)
                r = Reward_adapter(r, opt.EnvIdex)       # Custom reward adapter
                done = (dw or tr)

                # (iv) Store transition for PPO
                agent.put_data(
                    s, a, r, s_next,
                    logprob_a, done, dw, idx=traj_length
                )

                # Move to next step
                s = s_next
                traj_length += 1
                total_steps += 1

                # (v) Update agent if horizon reached
                printer = traj_length % (10 * opt.T_horizon) == 0
                if traj_length % opt.T_horizon == 0:
                    agent.train(printer, writer, total_steps)
                    traj_length = 0
                
                # if total_steps >= 4e5 and total_steps % 5e3 == 0:
                #     agent.a_lr *= 0.95
                #     agent.c_lr *= 0.95

                # (vi) Periodically evaluate and log
                if total_steps % opt.eval_interval == 0:
                    score = evaluate_policy(eval_env, agent, opt.max_action, turns=20)
                    if writer is not None:
                        writer.add_scalar('ep_r', score, global_step=total_steps)
                    print(
                        f"EnvName: {EnvName[opt.EnvIdex]} | "
                        f"Steps: {int(total_steps/1000)}k | "
                        f"Score: {score}"
                    )

                # (vii) Periodically save model
                if opt.save_model and total_steps % opt.save_interval == 0:
                    agent.save(BrifEnvName[opt.EnvIdex])

    # Close environments
    env.close()
    eval_env.close()


if __name__ == '__main__':
    '''Hyperparameter Setting'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda', help='running device: cuda or cpu')
    parser.add_argument('--EnvIdex', type=int, default=0, help='PV1, Lch_Cv2, Humanv4, HCv4, BWv3, BWHv3')
    parser.add_argument('--write', type=str2bool, default=False, help='Use SummaryWriter to record the training')
    parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
    parser.add_argument('--load_model', type=str2bool, default=False, help='Load pretrained model or Not')
    parser.add_argument('--eval_model', type=str2bool, default=False, help='Evaluate only')
    parser.add_argument('--save_model', type=str2bool, default=False, help='Evaluate only')

    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--T_horizon', type=int, default=2048, help='lenth of long trajectory')
    parser.add_argument('--Distribution', type=str, default='Beta', help='Should be one of Beta ; GS_ms  ;  GS_m')
    parser.add_argument('--Max_train_steps', type=int, default=int(1e6), help='Max training steps')
    parser.add_argument('--save_interval', type=int, default=int(3e5), help='Model saving interval, in steps.')
    parser.add_argument('--eval_interval', type=int, default=int(5e3), help='Model evaluating interval, in steps.')

    parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
    parser.add_argument('--lambd', type=float, default=0.95, help='GAE Factor')
    parser.add_argument('--clip_rate', type=float, default=0.2, help='PPO Clip rate')
    parser.add_argument('--K_epochs', type=int, default=10, help='PPO update times')
    parser.add_argument('--net_width', type=int, default=128, help='Hidden net width')
    parser.add_argument('--net_layer', type=int, default=1, help='Hidden net layers')
    parser.add_argument('--a_lr', type=float, default=2e-5, help='Learning rate of actor')
    parser.add_argument('--c_lr', type=float, default=2e-5, help='Learning rate of critic')
    parser.add_argument('--b_lr', type=float, default=2e-4, help='Learning rate of dual optimization problem')
    parser.add_argument('--g_lr', type=float, default=2e-4, help='Learning rate of dual optimization problem')
    parser.add_argument('--r_lr', type=float, default=2e-4, help='Learning rate of reward')
    parser.add_argument('--l2_reg', type=float, default=1e-3, help='L2 regulization coefficient for Critic')
    parser.add_argument('--a_optim_batch_size', type=int, default=64, help='lenth of sliced trajectory of actor')
    parser.add_argument('--c_optim_batch_size', type=int, default=64, help='lenth of sliced trajectory of critic')
    parser.add_argument('--entropy_coef', type=float, default=1e-3, help='Entropy coefficient of Actor')
    parser.add_argument('--entropy_coef_decay', type=float, default=0.99, help='Decay rate of entropy_coef')

    parser.add_argument('--robust', type=bool, default=False, help='Robust policy')
    parser.add_argument('--noise', type=bool, default=False, help='Env with noise')
    parser.add_argument('--std', type=float, default=0.0, help='Noise std')
    parser.add_argument('--delta', type=float, default=0.0, help='Noise std')

    opt = parser.parse_args()
    opt.device = torch.device(opt.device) # from str to torch.device
    if not opt.eval_model:
        print(opt)

    main(opt)
