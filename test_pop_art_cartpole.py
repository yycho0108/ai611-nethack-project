#!/usr/bin/env python3

import gym
import numpy as np
import torch as th
nn = th.nn
F = nn.functional
from functools import partial
from pop_art_agent import PopArtAgent
from typing import Dict, Tuple

# NOTE(ycho): only for temporary testing
from stable_baselines3.common.vec_env import SubprocVecEnv


def senc(x):
    return th.as_tensor(x, dtype=th.float32)
    # return th.as_tensor(x, dtype=th.float32)[None]


class DummyStateEncoder(nn.Module):
    def __init__(self, h_dim: int, device: th.device):
        super().__init__()
        self.h_dim = h_dim
        self.device = device
        self.linear = nn.Linear(4, h_dim)

    def forward(self, inputs: Dict[str, th.Tensor],
                core_state: Tuple[th.Tensor, th.Tensor],
                done: th.Tensor) -> Tuple[th.Tensor,
                                          Tuple[th.Tensor, th.Tensor]]:
        x = th.as_tensor(inputs, device=self.device)
        y = self.linear(x)
        print(F'{x.shape} -> {y.shape}')
        # core_state = th.as_tensor(s, device=self.device)
        return (y, core_state)

    def initial_state(self, batch_size: int) -> Tuple[th.Tensor, th.Tensor]:
        return (th.zeros((batch_size, 0), device=self.device),
                th.zeros((batch_size, 0), device=self.device))


def main():
    # env = gym.make('CartPole-v1')
    # env.reset()
    # env.step(th.as_tensor(0)) # --> error
    # env.step(0) # --> ok
    env_fns = [partial(gym.make, 'CartPole-v1') for _ in range(16)]
    env = SubprocVecEnv(env_fns)
    #device: th.device = (
    #    th.device('cuda') if th.cuda.is_available() else th.device('cpu'))
    device = th.device('cpu')
    state_encoder = DummyStateEncoder(
        h_dim=np.prod(env.observation_space.shape),
        device=device).to(device)
    agent = PopArtAgent(device, state_encoder, env,
                        num_env=len(env_fns),
                        hidden_dim=64).to(device)
    agent.learn()


if __name__ == '__main__':
    main()
