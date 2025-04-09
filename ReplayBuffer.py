import numpy as np
import torch
from pathlib import Path

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
    
    def save(self, path=None): 
        if not path:
            path = Path(f"./dataset")
        path.mkdir(parents=True, exist_ok=True)
            
        # Store size in the front of reward buffer
        r_np = self.r[:self.size].cpu().numpy()
        r_with_size = np.insert(r_np, 0, self.size)
        np.save(f"{path}/r.npy", r_with_size)
        
        np.save(f"{path}/s.npy", self.s[:self.size].cpu().numpy())
        np.save(f"{path}/a.npy", self.a[:self.size].cpu().numpy())
        np.save(f"{path}/s_next.npy", self.s_next[:self.size].cpu().numpy())
        np.save(f"{path}/dw.npy", self.dw[:self.size].cpu().numpy())
        
    def load(self, path=None, d4rl_dataset=False):
        if d4rl_dataset:  
            self.size = int(d4rl_dataset.rewards.shape[0])
            self.s[:self.size,] = torch.from_numpy(d4rl_dataset.observations).to(self.device)
            self.a[:self.size,] = torch.from_numpy(d4rl_dataset.actions).to(self.device)
            self.r[:self.size,] = torch.from_numpy(d4rl_dataset.rewards).to(self.device)
            self.s_next[:self.size,] = torch.from_numpy(d4rl_dataset.next_observations).to(self.device)
            self.dw[:self.size,] = torch.from_numpy(d4rl_dataset.terminals).to(self.device)
        else:
            self.size = int(np.load(f"{path}/r.npy")[0]) 
            self.s[:self.size,] = torch.from_numpy(np.load(f"{path}/s.npy")).to(self.device)
            self.a[:self.size,] = torch.from_numpy(np.load(f"{path}/a.npy")).to(self.device)
            self.r[:self.size,] = torch.from_numpy(np.load(f"{path}/r.npy")[1:]).reshape(-1,1).to(self.device)
            self.s_next[:self.size,] = torch.from_numpy(np.load(f"{path}/s_next.npy")).to(self.device)
            self.dw[:self.size,] = torch.from_numpy(np.load(f"{path}/dw.npy")).to(self.device)
            