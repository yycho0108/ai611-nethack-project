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


def senc(x):
    return th.as_tensor(x, dtype=th.float32)
    # return th.as_tensor(x, dtype=th.float32)[None]


class DummyStateEncoder(nn.Module):
    def __init__(self, h_dim: int, device: th.device):
        super().__init__()
        self.h_dim = h_dim
        self.device = device
        # self.linear = nn.Linear(4, h_dim)

    def forward(self, inputs: Dict[str, th.Tensor],
                core_state: Tuple[th.Tensor, th.Tensor],
                done: th.Tensor) -> Tuple[th.Tensor,
                                          Tuple[th.Tensor, th.Tensor]]:
        x = th.as_tensor(inputs, device=self.device)
        # y = self.linear(x)
        # print(F'{x.shape} -> {y.shape}')
        # core_state = th.as_tensor(s, device=self.device)
        return (x, core_state)

    def initial_state(self, batch_size: int) -> Tuple[th.Tensor, th.Tensor]:
        return (th.zeros((batch_size, 0), device=self.device),
                th.zeros((batch_size, 0), device=self.device))


def train():
    env_fns = [partial(gym.make, 'CartPole-v1') for _ in range(16)]
    env = SubprocVecEnv(env_fns)
    device: th.device = (
        th.device('cuda') if th.cuda.is_available() else th.device('cpu'))
    h_dim: int = np.prod(env.observation_space.shape)
    state_encoder = DummyStateEncoder(
        h_dim=h_dim,
        device=device).to(device)
    agent = PopArtAgent(device, state_encoder, env,
                        num_env=len(env_fns),
                        hidden_dim=h_dim).to(device)

    path = get_new_dir('/tmp/cartpole')
    log_path = ensure_dir(path / 'log')
    agent.learn(log_dir=log_path)


def test():
    env_fns = [partial(gym.make, 'CartPole-v1') for _ in range(1)]
    env = SubprocVecEnv(env_fns)
    device: th.device = get_device()
    h_dim: int = np.prod(env.observation_space.shape)

    state_encoder = DummyStateEncoder(
        h_dim=h_dim,
        device=device).to(device)
    agent = PopArtAgent(device, state_encoder, env,
                        num_env=len(env_fns),
                        hidden_dim=h_dim).to(device)
    agent.load_ckpt('/tmp/cartpole/run-007/log/nh-pa-00900.pt')
    # agent.eval()

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
