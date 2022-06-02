#!/usr/bin/env python3

import gym
import nle

env = gym.make('NetHackScore-v0')
obs = env.reset()
for _ in range(128):
    obs = env.reset()
    print(obs['blstats'])
    # print(obs['blstats'].min(), obs['blstats'].max(), obs['blstats'].shape)
# print(env.observation_space['features'])
# print(list(env.observation_space))
print(env.observation_space['glyphs']) # (0, 5976, (21,79), int16)
print(env.observation_space['blstats']) # (0, 5976, (21,79), int16)
print(env.observation_space['message']) # (0, 5976, (21,79), int16)
print(env.action_space)
env.step(1)
env.render()
