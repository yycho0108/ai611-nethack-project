#!/usr/bin/env python3

import gym
import nle
import torch as th
from gym.wrappers.transform_observation import TransformObservation

from feature import NetHackEncoder


def transform_observation(obs):
    """Transform the environment observation.

    * (1) add channel dimension to `glyphs`
    * (2) add batch dimension to all relevant inputs.
    """
    return {'glyphs': th.as_tensor(obs['glyphs'][None, None]),
            'blstats': th.as_tensor(obs['blstats'][None, None]),
            'message': th.as_tensor(obs['message'][None])}


def main():
    env = gym.make('NetHackScore-v0')
    env = TransformObservation(env, transform_observation)
    obs = env.reset()
    enc = NetHackEncoder(env.observation_space)
    core_state = enc.initial_state()
    done = th.zeros((1, 1), dtype=th.int)
    feat, _ = enc(obs, core_state, done)
    print(feat.shape)


if __name__ == '__main__':
    main()
