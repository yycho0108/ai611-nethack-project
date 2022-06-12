#!/usr/bin/env python3

import gym
import nle
import torch as th
nn = th.nn
F = nn.functional
from functools import partial
from pop_art_agent import PopArtAgent
from feature import NetHackEncoder
from typing import Tuple, Dict

from stable_baselines3.common.vec_env import SubprocVecEnv


class FormatObservationWrapper(gym.ObservationWrapper):
    def __init__(self,
                 env: gym.Env,
                 keys: Tuple[str, ...] = ('glyphs', 'blstats'),
                 ):
        super().__init__(env)
        self.keys = keys
        self.observation_space = gym.spaces.Dict(
            {k: env.observation_space[k] for k in keys})

    def step_wait(self, *args, **kwds):
        observation, reward, done, info = self.env.step_wait(*args, **kwds)
        return self.observation(observation), reward, done, info

    def observation(self, obs_in):
        obs_out: Dict[str, np.ndarray] = dict()
        for key in self.keys:
            entry = obs_in[key]
            # entry = th.from_numpy(entry).unsqueeze(dim=0)
            obs_out[key] = entry[None]
        return obs_out


def make_env(*args, **kwds) -> gym.Env:
    env = gym.make('NetHackScore-v0')
    env = FormatObservationWrapper(env)
    return env


def main():
    num_env = 16
    env_fns = [make_env for _ in range(num_env)]
    env = SubprocVecEnv(env_fns)
    device: th.device = (
        th.device('cuda') if th.cuda.is_available() else th.device('cpu'))
    state_encoder = NetHackEncoder(
        observation_shape=env.observation_space,
        use_lstm=True
    ).to(device)
    agent = PopArtAgent(
        device,
        state_encoder,
        env,
        num_env=num_env,
        hidden_dim=512).to(device)
    agent.learn()


if __name__ == '__main__':
    main()
