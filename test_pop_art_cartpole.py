#!/usr/bin/env python3

from functools import partial
from typing import Dict, Tuple, Union
from pathlib import Path

import gym
import numpy as np
import torch as th
nn = th.nn
F = nn.functional

# NOTE(ycho): only for temporary testing
from util import ensure_dir, get_new_dir, get_device
from pop_art_agent import PopArtAgent
from stable_baselines3.common.vec_env import SubprocVecEnv


class CartpoleStateEncoder(nn.Module):
    """lightweight state encoder for CartPole environment."""

    def __init__(self, h_dim: int, device: th.device):
        super().__init__()
        self.h_dim = h_dim
        self.device = device
        self.encoder = nn.Sequential(
            nn.Linear(4, h_dim),
            nn.Tanh(),
            nn.Linear(h_dim, h_dim),
            nn.Tanh()
        )

    def forward(self, inputs: th.Tensor,
                core_state: Tuple[th.Tensor, th.Tensor],
                done: th.Tensor) -> Tuple[th.Tensor,
                                          Tuple[th.Tensor, th.Tensor]]:
        x = th.as_tensor(inputs, device=self.device)
        y = self.encoder(x)
        return (y, core_state)

    def initial_state(self, batch_size: int) -> Tuple[th.Tensor, th.Tensor]:
        return (th.zeros((batch_size, 0), device=self.device),
                th.zeros((batch_size, 0), device=self.device))


def make_env(index: int) -> gym.Env:
    env = gym.make('CartPole-v1')
    # env = gym.make('MountainCar-v0')
    env.seed(index)
    return env


def train():
    # env_fns = [partial(gym.make, 'CartPole-v1') for _ in range(64)]
    env_fns = [partial(make_env, i) for i in range(32)]
    env = SubprocVecEnv(env_fns)
    device: th.device = get_device()

    state_dim: int = 64
    hidden_dim: int = 64
    state_encoder = CartpoleStateEncoder(
        h_dim=state_dim,
        device=device).to(device)
    agent = PopArtAgent(device, state_encoder, env,
                        hidden_dim=hidden_dim).to(device)

    path = get_new_dir('/tmp/cartpole')
    log_path = ensure_dir(path / 'log')
    agent.train(True)
    agent.learn(100000, log_dir=log_path,
                save_steps=10000)


def test():
    env_fns = [partial(gym.make, 'CartPole-v1') for _ in range(1)]
    # env_fns = [partial(gym.make, 'MountainCar-v0') for _ in range(1)]
    env = SubprocVecEnv(env_fns)
    device: th.device = get_device()

    state_dim: int = 64
    hidden_dim: int = 64
    state_encoder = CartpoleStateEncoder(
        h_dim=state_dim,
        device=device).to(device)
    agent = PopArtAgent(device, state_encoder, env,
                        hidden_dim=hidden_dim).to(device)
    # agent.load_ckpt('/tmp/cartpole/run-013/log/nh-pa-00900.pt')
    agent.load_ckpt('/tmp/cartpole/run-033/log/nh-pa-90000.pt')
    agent.eval()

    obs = env.reset()
    core_state = tuple(s.to(device) for s in state_encoder.initial_state(1))
    done = th.zeros(1, dtype=bool, device=device)
    cum_rew = 0.0
    for _ in range(1000):
        obs = th.as_tensor(obs, device=device)
        done = th.as_tensor(done, dtype=th.bool, device=device)
        state, core_state = state_encoder(obs, core_state, done[None])
        with th.autograd.detect_anomaly():
            action = agent.get_action(state, deterministic=True)
            # print('action', action.shape)
            # action = th.mean(action)
            # action.backward()
        obs, rew, done, info = env.step(action.detach().cpu().numpy())
        cum_rew += rew
        if done:
            print(cum_rew, done)
            cum_rew = 0
        # env.unwrapped.render(mode='human')


if __name__ == '__main__':
    train()
    # test()
