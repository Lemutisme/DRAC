import argparse
import torch
import torch.nn as nn
import numpy as np
import random
from torch.distributions import Normal
from math import exp
import gymnasium as gym
from gymnasium.envs.classic_control.pendulum import PendulumEnv

def build_net(layer_shape, hidden_activation, output_activation):
    '''Build net with for loop'''
    layers = []
    for j in range(len(layer_shape)-1):
        act = hidden_activation if j < len(layer_shape)-2 else output_activation
        layers += [nn.Linear(layer_shape[j], layer_shape[j+1]), act()]
    return nn.Sequential(*layers)

# def discretize(state, grid):
#     # if len(state.shape) < 2:
#     # 	state = state.view(1,-1)
#     assert state.shape[0] == len(grid)
#     dis_state = []
#     for i in range(state.shape[0]):
#         idx = torch.searchsorted(grid[i], state[i], right=True)
#         idx = (idx - 1).clamp(min = 0)
#         dis_state.append(grid[i][idx])
#     return torch.stack(dis_state, dim=-1)

# Disrupted Env
class CustomPendulumEnv(PendulumEnv):
    def __init__(self, render_mode=None, g=10.0, length=1.0):
        super().__init__(render_mode=render_mode, g=g)
        self.l = length  # override the pendulum length

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

def Action_adapter(a,max_action):
    #from [-1,1] to [-max,max]
    return  a*max_action

def Action_adapter_reverse(act,max_action):
    #from [-max,max] to [-1,1]
    return  act/max_action

def evaluate_policy(env, agent, turns = 5):
    total_scores = 0
    dist = Normal(0.0, 0.1)
    for j in range(turns):
        s, info = env.reset()
        done = False
        while not done:
            # Take deterministic actions at test time
            a = agent.select_action(s, deterministic=True)
            a_env = Action_adapter(a, agent.max_action)
            s_next, r, dw, tr, info = env.step(a_env)
            done = (dw or tr)

            total_scores += r
            s = s_next
    return int(total_scores/turns)

# def evaluate_policy_PPOD(env, agent, turns = 3):
#     total_scores = 0
#     for j in range(turns):
#         s, info = env.reset()
#         done = False
#         while not done:
#             # Take deterministic actions at test time
#             a = agent.select_action(s, deterministic=True)
#             s_next, r, dw, tr, info = env.step(a)
#             done = (dw or tr)

#             total_scores += r
#             s = s_next
#     return int(total_scores/turns)

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
