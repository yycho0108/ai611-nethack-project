#!/usr/bin/env python3

from typing import Tuple, Optional, Union, Iterable, Any, Dict, List
from abc import ABC, abstractmethod, abstractproperty
import numpy as np
import torch as th
nn = th.nn
F = nn.functional
import gym
import einops
from torch.distributions import Normal, Independent, Categorical

from vtrace import vtrace_from_importance_weights

from feature_extractor import NetHackNet


class PopArtModule(nn.Module):
    def __init__(self, c_in: int, c_out: int, beta: float = 4e-4):
        super().__init__()
        self.beta = beta
        self.c_in = c_in
        self.c_out = c_out

        self.linear = nn.Linear(c_in, c_out, True)
        self.register_buffer('mu', th.zeros(c_out,
                                            requires_grad=False,
                                            dtype=th.float32))
        self.register_buffer('sigma', th.ones(c_out,
                                              requires_grad=False,
                                              dtype=th.float32))
        # self.linear.reset_parameters()

    def forward(self, x: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        y_hat = self.linear(x)
        with th.no_grad():
            y = y_hat * self.sigma * self.mu
        # FIXME(ycho): hack with single-task assumption !!!!
        return (y.squeeze(-1), y_hat.squeeze(-1))

    def update_parameters(self, vs, task):
        # TODO(ycho): figure out the following:
        # vs=?
        # task=?
        mu0 = self.mu
        sigma0 = self.sigma

        vs = vs * task
        n = task.sum((0, 1))
        mu = vs.sum((0, 1)) / n
        nu = th.sum(vs**2, (0, 1)) / n
        sigma = th.sqrt(nu - mu**2)
        sigma = th.clamp(sigma, min=1e-4, max=1e+6)

        # nan values are replaced with old values.
        mu[th.isnan(mu)] = mu0[th.isnan(mu)]
        sigma[th.isnan(sigma)] = sigma0[th.isnan(sigma)]

        # polyak average, I guess.
        self.mu = (1 - self.beta) * self.mu + self.beta * mu
        self.sigma = (1 - self.beta) * self.sigma + self.beta * sigma

        # Update nn.Linear params
        self.linear.weight.data = (
            self.linear.weight.t() * sigma0 / self.sigma).t()
        self.linear.bias.data = (
            sigma0 * self.linear.bias + oldmu - self.mu) / self.sigma


class PopArtAgent(nn.Module):
    def __init__(self,
                 state_encoder: NetHackNet,
                 env: gym.Env, 
                 num_env: int = 8, 
                 num_interactions = 5, 
                 hidden_dim: int = 4,
                 use_continuous_actions: bool = False):
        super().__init__()
        self.env = env
        self.num_env = num_env
        self.num_interactions = num_interactions
        self.state_encoder = state_encoder
        self.is_encoder_intialized = False
        self.action_dim = self.env.action_space.n

        if use_continuous_actions:
            self.policy = nn.Sequential(
                nn.Linear(self.state_encoder.h_dim, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, 2 * self.action_dim, bias=False)
            )
        else:
            self.policy = nn.Sequential(
                nn.Linear(self.state_encoder.h_dim, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, self.action_dim, bias=False)
            )
        self.pop_art = PopArtModule(
            hidden_dim,
            1)
        self.optimizer = th.optim.Adam(self.parameters())

    def get_action_distribution(
            self, state: th.Tensor) -> th.distributions.Distribution:
        params = self.policy(state)
        params = params.reshape(-1, self.num_env, self.action_dim)
        return Categorical(logits=params)
        # mu, std = einops.rearrange(params, '... (k d) -> k ... d', k=2)
        # return Independent(Normal(mu, std), 1)

    def get_action(self, state: th.Tensor,
                   deterministic: bool = True):
        dist = self.get_action_distribution(state)
        if deterministic:
            # NOTE(ycho): does not generally work
            # for distribution variants, such as
            # Independent / Transformed / Cauchy ...
            return dist.mean
        else:
            return dist.sample()

    def interact(self):
        env = self.env
        buf = []

        done: np.ndarray = np.ones(self.num_env, dtype=np.bool8)
        prv_obs: Dict[str, th.Tensor] = None
        prv_core_state: Tuple[th.Tensor, th.Tensor] = None

        for _ in range(self.num_interactions):
            # obs = env.reset() if done else prv_obs
            if prv_obs is None:
                prv_obs = self._format_observations(env.reset())
            # obs = env.reset() if (prv_obs is None) else prv_obs

            with th.no_grad():
                if not self.is_encoder_intialized:
                    prv_core_state = self.state_encoder.initial_state(batch_size=self.num_env)
                    self.is_encoder_initialized = True
                state, core_state = self.state_encoder(
                    prv_obs, prv_core_state, th.tensor(done, dtype=th.bool).unsqueeze(dim=0)
                )
                dist = self.get_action_distribution(state)
                action = dist.sample().squeeze()
                # NOTE(ycho): store `log_prob` for vtrace calculation
                log_prob = dist.log_prob(action).squeeze()

            obs, rew, done, info = env.step(action.cpu().numpy())
            obs = self._format_observations(obs)
            rew = np.asarray(rew, dtype=np.float32)
            buf.append((prv_obs, action, log_prob, obs, rew, done))
            prv_obs = obs
            prv_core_state = core_state
        return buf, core_state

    def _format_observations(self, observation, keys=("glyphs", "blstats")) -> Dict[str, th.Tensor]:
        observations: Dict[str, th.Tensor] = dict()
        for key in keys:
            entry = observation[key]
            entry = th.from_numpy(entry).unsqueeze(dim=0)
            observations[key] = entry
        return observations

    def sample_steps(self, buf):
        buf_t = []
        for x in buf:
            buf_step = []
            for e in x:
                if isinstance(e, np.ndarray):
                    buf_step.append(th.as_tensor(e))
                else:
                    buf_step.append(e)
            buf_t.append(buf_step)
        return buf_t

    def _learn_step(self):
        # -- collect-rollouts --
        buf, core_state = self.interact()
        samples = self.sample_steps(buf)
        obs0s, actions, lp0, obs1s, rewards, dones = zip(*samples)

        # T, B, ...
        obs0s = self._stack_observations(obs0s)
        actions = th.stack(actions, axis=0)
        lp0 = th.stack(lp0, axis=0)
        obs1s = self._stack_observations(obs1s)
        rewards = th.stack(rewards, axis=0)
        dones = th.stack(dones, axis=0)

        state0s, _ = self.state_encoder(obs0s, core_state, dones)
        action_dist = self.get_action_distribution(state0s)
        lp1 = action_dist.log_prob(actions)
        baseline, normalized_baseline = self.pop_art(state0s)
        baseline = baseline.reshape((-1, self.num_env))
        normalized_baseline = normalized_baseline.reshape((-1, self.num_env))
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        log_rho = (lp1 - lp0)  # log importance ratio and stuff
        # log_rho = th.clamp(log_rho, max=0.0)
        rho = th.exp(log_rho)

        # Derive bootstrap / discount values
        bootstrap_value = baseline[-1]
        discounts = (~dones).to(th.float32) * 0.99

        # Vtrace calculation
        vs, pg_adv = vtrace_from_importance_weights(
            log_rho,
            discounts,
            rewards,
            baseline,
            dim_t=1)  # NO idea, to be honest

        # normalized v_s
        nvs = (vs - self.pop_art.mu) / self.pop_art.sigma
        # policy gradient loss, valid_mask=?
        pg_loss = th.mean(-log_prob * pg_adv).to(th.float32)
        # value baseline loss, valid_mask=?
        vb_loss = F.mse_loss(nvs, normalized_baseline).to(th.float32)
        # entropy loss [...]
        ent_loss = -th.mean(action_dist.entropy())

        loss = (pg_loss + vb_loss + ent_loss)
        print('loss', loss)

        self.optimizer.zero_grad()
        loss.backward()
        # clip_grad_norm_(...)
        self.optimizer.step()

        # TODO(ycho): `task` labels are required for pop-art.
        # Should we supply this in some way??
        # self.pop_art.update_parameters(vs, task)

    def _stack_observations(self, observations: Iterable[Dict[str, th.Tensor]]) -> Dict[str, th.Tensor]:
        _temp_obs_stack: Dict[str, List[th.Tensor]] = dict()
        keys: List[str] = list()
        for observation in observations:
            for key, value in observation.items():
                if key not in _temp_obs_stack: 
                    _temp_obs_stack[key] = list()
                    keys.append(key)
                _temp_obs_stack[key].append(value)
        return {key: th.cat(_temp_obs_stack[key], dim=0) for key in keys}

    def learn(self):
        for _ in range(1):
            self._learn_step()
