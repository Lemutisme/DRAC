import copy
import torch
import numpy as np
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from utils import Double_Q_Net, Policy_Net, ReplayBuffer, evaluate_policy, str2bool
from datetime import datetime
import gymnasium as gym
import os, shutil
import argparse

class SACD_agent():
	def __init__(self, **kwargs):
		# Init hyperparameters for agent, just like "self.gamma = opt.gamma, self.lambd = opt.lambd, ..."
		self.__dict__.update(kwargs)
		self.tau = 0.005
		self.H_mean = 0
		self.replay_buffer = ReplayBuffer(self.state_dim, self.action_dim, max_size=int(1e6), device=self.device)

		self.actor = Policy_Net(self.state_dim, self.action_dim, self.hid_shape).to(self.device)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)

		self.q_critic = Double_Q_Net(self.state_dim, self.action_dim, self.hid_shape).to(self.device)
		self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=self.lr)
		self.q_critic_target = copy.deepcopy(self.q_critic)
		for p in self.q_critic_target.parameters(): p.requires_grad = False

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
			target_Q = r + (~dw) * self.gamma * v_next

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
	#Create Env
	EnvName = ['CartPole-v1', 'LunarLander-v2']
	BriefEnvName = ['CPV1', 'LLdV2']
	env = gym.make(EnvName[opt.EnvIdex], render_mode="human" if opt.render else None)
	eval_env = gym.make(EnvName[opt.EnvIdex])
	opt.state_dim = env.observation_space.shape[0]
	opt.action_dim = env.action_space.n
	opt.max_e_steps = env._max_episode_steps

	# Seed Everything
	env_seed = opt.seed
	torch.manual_seed(opt.seed)
	torch.cuda.manual_seed(opt.seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	print("Random Seed: {}".format(opt.seed))

	print('Algorithm: SACD','  Env:',BriefEnvName[opt.EnvIdex],'  state_dim:',opt.state_dim,
		  '  action_dim:',opt.action_dim,'  Random Seed:',opt.seed, '  max_e_steps:',opt.max_e_steps, '\n')

	if opt.write:
		from torch.utils.tensorboard import SummaryWriter
		timenow = str(datetime.now())[0:-10]
		timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
		writepath = 'runs/SACD_{}'.format(BriefEnvName[opt.EnvIdex]) + timenow
		if os.path.exists(writepath): shutil.rmtree(writepath)
		writer = SummaryWriter(log_dir=writepath)

	#Build model
	if not os.path.exists('model'): os.mkdir('model')
	agent = SACD_agent(**vars(opt))
	if opt.Loadmodel: agent.load(opt.ModelIdex, BriefEnvName[opt.EnvIdex])

	if opt.render:
		while True:
			score = evaluate_policy(env, agent, 1)
			print('EnvName:', BriefEnvName[opt.EnvIdex], 'seed:', opt.seed, 'score:', score)
	else:
		total_steps = 0
		while total_steps < opt.Max_train_steps:
			s, info = env.reset(seed=env_seed)  # Do not use opt.seed directly, or it can overfit to opt.seed
			env_seed += 1
			done = False

			'''Interact & trian'''
			while not done:
				#e-greedy exploration
				if total_steps < opt.random_steps: a = env.action_space.sample()
				else: a = agent.select_action(s, deterministic=False)
				s_next, r, dw, tr, info = env.step(a) # dw: dead&win; tr: truncated
				done = (dw or tr)

				if opt.EnvIdex == 1:
					if r <= -100: r = -10  # good for LunarLander

				agent.replay_buffer.add(s, a, r, s_next, dw)
				s = s_next

				'''update if its time'''
				# train 50 times every 50 steps rather than 1 training per step. Better!
				if total_steps >= opt.random_steps and total_steps % opt.update_every == 0:
					for j in range(opt.update_every):
						agent.train()

				'''record & log'''
				if total_steps % opt.eval_interval == 0:
					score = evaluate_policy(eval_env, agent, turns=3)
					if opt.write:
						writer.add_scalar('ep_r', score, global_step=total_steps)
						writer.add_scalar('alpha', agent.alpha, global_step=total_steps)
						writer.add_scalar('H_mean', agent.H_mean, global_step=total_steps)
					print('EnvName:', BriefEnvName[opt.EnvIdex], 'seed:', opt.seed,
						  'steps: {}k'.format(int(total_steps / 1000)), 'score:', int(score))
				total_steps += 1

				'''save model'''
				if total_steps % opt.save_interval == 0:
					agent.save(int(total_steps/1000), BriefEnvName[opt.EnvIdex])
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
    parser.add_argument('--Max_train_steps', type=int, default=4e5, help='Max training steps')
    parser.add_argument('--save_interval', type=int, default=1e5, help='Model saving interval, in steps.')
    parser.add_argument('--eval_interval', type=int, default=2e3, help='Model evaluating interval, in steps.')
    parser.add_argument('--random_steps', type=int, default=1e4, help='steps for random policy to explore')
    parser.add_argument('--update_every', type=int, default=50, help='training frequency')

    parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
    parser.add_argument('--hid_shape', type=list, default=[200,200], help='Hidden net shape')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--alpha', type=float, default=0.2, help='init alpha')
    parser.add_argument('--adaptive_alpha', type=str2bool, default=True, help='Use adaptive alpha turning')
    opt = parser.parse_args()
    opt.device = torch.device(opt.device) # from str to torch.device
    print(opt)

    main(opt)