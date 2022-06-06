#!/usr/bin/env python3

import gym
import nle
import torch as th
nn = th.nn
F = nn.functional
from functools import partial
from pop_art_agent import PopArtAgent
from feature_extractor import NetHackNet

from stable_baselines3.common.vec_env import SubprocVecEnv


def main():
    env_fns = [partial(gym.make, 'NetHackScore-v0') for _ in range(2)]
    env = SubprocVecEnv(env_fns)
    state_encoder = NetHackNet(use_lstm=True)
    agent = PopArtAgent(state_encoder, env,
                        action_dim=2)
    agent.learn()


if __name__ == '__main__':
    main()
