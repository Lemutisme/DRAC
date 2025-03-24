import argparse
import torch.nn as nn

def build_net(layer_shape, hidden_activation, output_activation):
    '''Build net with for loop'''
    layers = []
    for j in range(len(layer_shape)-1):
        act = hidden_activation if j < len(layer_shape)-2 else output_activation
        layers += [nn.Linear(layer_shape[j], layer_shape[j+1]), act()]
    return nn.Sequential(*layers)

# Reward engineering for better training
def Reward_adapter(r, EnvIdex):
    # For Pendulum-v0
    if EnvIdex == 0:
        r = (r + 8) / 8
    # For LunarLander
    elif EnvIdex == 2:
        if r <= -100: r = -10
    # elif EnvIdex == 2:
    #     r = r / 5
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

def evaluate_policy_SAC(env, agent, turns = 1, seeds_list = []):
    total_scores = 0
    for j in range(turns):
        if len(seeds_list) > 0:
            s, info = env.reset(seed=seeds_list[j])
        else:
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
