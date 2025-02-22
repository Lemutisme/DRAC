import argparse
import torch.nn as nn
import numpy as np
from numpy.random import normal
from gymnasium.envs.classic_control.pendulum import PendulumEnv, angle_normalize
from gymnasium.envs.registration import register


def build_net(layer_shape, hidden_activation, output_activation):
    '''Build net with for loop'''
    layers = []
    for j in range(len(layer_shape)-1):
        act = hidden_activation if j < len(layer_shape)-2 else output_activation
        layers += [nn.Linear(layer_shape[j], layer_shape[j+1]), act()]
    return nn.Sequential(*layers)

# Disrupted Env
class CustomPendulumEnv(PendulumEnv):
    def __init__(self, render_mode=None, std=0.0):
        super().__init__(render_mode=render_mode)
        self.noise_std = std
        print(f"Penludum Env with Observation Noise STD={std}.")
        
    def step(self, u):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering
        costs = angle_normalize(th) ** 2 + 0.1 * thdot**2 + 0.001 * (u**2)

        newthdot = thdot + (3 * g / (2 * l) * np.sin(th) + 3.0 / (m * l**2) * u) * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * dt + normal(0, self.noise_std) # theta computation is not accurate.

        self.state = np.array([newth, newthdot])

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return self._get_obs(), -costs, False, False, {}

register(
    id="CustomPendulum-v1",
    entry_point="utils:CustomPendulumEnv",
    max_episode_steps=200,
    kwargs={'std': 0.0}
)


# Reward engineering for better training
def Reward_adapter(r, EnvIdex):
    # For Pendulum-v0
    if EnvIdex == 0:
        r = (r + 8) / 8
    # For LunarLander
    elif EnvIdex == 1:
        if r <= -100: r = -10
    elif EnvIdex == 2:
        r = r / 5
    # For BipedalWalker
    elif EnvIdex == 4 or EnvIdex == 5:
        if r <= -100: r = -1
    return r

def Action_adapter_symm(a,max_action):
    #from [-1,1] to [-max,max]
    return  a*max_action

def Action_adapter_symm_reverse(act,max_action):
    #from [-max,max] to [-1,1]
    return  act/max_action

def Action_adapter_pos(a, max_action):
    #from [0,1] to [-max,max]
    return  2 * (a - 0.5) * max_action

def evaluate_policy_SAC(env, agent, turns = 5):
    total_scores = 0
    for j in range(turns):
        s, info = env.reset()
        done = False
        while not done:
            # Take deterministic actions at test time
            a = agent.select_action(s, deterministic=True)
            a_env = Action_adapter_symm(a, agent.max_action)
            s_next, r, dw, tr, info = env.step(a_env)
            done = (dw or tr)

            total_scores += r
            s = s_next
    return int(total_scores/turns)

def evaluate_policy_PPO(env, agent, max_action, turns=3):
    total_scores = 0
    for j in range(turns):
        s, info = env.reset()
        done = False
        while not done:
            a, logprob_a = agent.select_action(s, deterministic=True) # Take deterministic actions when evaluation
            act = Action_adapter_pos(a, max_action)  # [0,1] to [-max,max]
            s_next, r, dw, tr, info = env.step(act)
            done = (dw or tr)

            total_scores += r
            s = s_next

    return total_scores/turns

def str2bool(v):
    '''transfer str to bool for argparse'''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'True','true','TRUE', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False','false','FALSE', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
