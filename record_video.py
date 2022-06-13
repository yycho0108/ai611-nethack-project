#!/usr/bin/env python3

import gym
import nle
import torch as th
import numpy as np
nn = th.nn
F = nn.functional
from functools import partial
from pop_art_agent import PopArtAgent
from feature import NetHackEncoder
from typing import Tuple, Dict
from tqdm.auto import tqdm
from pathlib import Path

from stable_baselines3.common.vec_env import DummyVecEnv

from util import ensure_dir, get_new_dir, get_device
from matplotlib import pyplot as plt
import argparse


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
            obs_out[key] = entry  # [None]
        return obs_out


def make_env(*args, **kwds) -> gym.Env:
    env = gym.make('NetHackScore-v0', savedir='/tmp/record')
    env = FormatObservationWrapper(env)
    return env


def main():
    # Argument parsing...
    parser = argparse.ArgumentParser(description='Record video.')
    parser.add_argument('load_ckpt', type=str,
                        help='Checkpoint file to load.')
    args = parser.parse_args()
    if not Path(args.load_ckpt).is_file():
        raise FileNotFoundError(F'{args.load_ckpt} not found!')

    num_env: int = 1
    env_fns = [make_env for _ in range(num_env)]
    env = DummyVecEnv(env_fns)
    device: th.device = get_device()

    state_encoder = NetHackEncoder(
        observation_shape=env.observation_space,
        device=device,
        use_lstm=True
    ).to(device)
    agent = PopArtAgent(
        device,
        state_encoder,
        env,
        hidden_dim=512).to(device)
    agent.load_ckpt(args.load_ckpt)
    agent.eval()
    state_encoder.eval()

    obs = env.reset()
    core_state = state_encoder.initial_state(batch_size=1,
                                             device=device)
    done: np.ndarray = np.zeros(shape=(1,), dtype=bool)
    logits = []
    with th.no_grad():
        steps: int = 0
        with tqdm() as pbar:
            while True:
                # Add time dimension.
                obs = {k: v[None] for k, v in obs.items()}
                state, core_state = state_encoder(
                    obs, core_state,
                    th.as_tensor(done, dtype=th.bool, device=device)[None]
                )
                # Remove time dimension.
                state = state.squeeze(dim=0)
                dist = agent.get_action_distribution(state)
                # action = th.argmax(dist.logits, dim=-1)
                action = dist.sample()
                logits.append(
                    th.softmax(
                        dist.logits,
                        dim=-1).ravel().detach().cpu().numpy())
                # NOTE(ycho): to see random behavior,
                # uncomment below block.
                #action = np.reshape(
                #    env.envs[0].action_space.sample(), (1,))
                obs, rew, done, info = env.step(action)
                # env.envs[0].unwrapped.render()
                steps += 1
                pbar.update(1)
                if done.any():
                    break
        # plt.hist(np.reshape(actions, -1))
        plt.bar(
            np.arange(env.action_space.n),
            np.mean(logits, axis=0),
            yerr=np.std(logits, axis=0),
            alpha=0.8)
        plt.title(F'Action probability distribution | {steps} steps')
        plt.xlabel('action')
        plt.ylabel('probability')
        plt.grid()
        plt.savefig('/tmp/actions.png')


if __name__ == '__main__':
    main()
