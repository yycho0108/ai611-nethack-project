#!/usr/bin/env python3

import gym
import torch as th
nn = th.nn
F = nn.functional
from functools import partial
from pop_art_agent import PopArtAgent

# NOTE(ycho): only for temporary testing
from stable_baselines3.common.vec_env import SubprocVecEnv


def senc(x):
    return th.as_tensor(x, dtype=th.float32)
    # return th.as_tensor(x, dtype=th.float32)[None]


def main():
    # env = gym.make('CartPole-v1')
    # env.reset()
    # env.step(th.as_tensor(0)) # --> error
    # env.step(0) # --> ok

    env_fns = [partial(gym.make, 'CartPole-v1') for _ in range(8)]
    env = SubprocVecEnv(env_fns)
    state_encoder = senc
    agent = PopArtAgent(state_encoder, env,
                        action_dim=2)
    agent.learn()


if __name__ == '__main__':
    main()
