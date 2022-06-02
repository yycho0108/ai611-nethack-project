#!/usr/bin/env python3

from typing import Tuple, Optional, Union, Iterable, Any, Dict
from abc import ABC, abstractmethod, abstractproperty
import torch as th
nn = th.nn
F = nn.functional
import gym
import einops
from torch.distributions import Normal, Independent, Categorical

from vtrace import vtrace_from_importance_weights


class PopArtModule(nn.Module):
    def __init__(self, c_in: int, c_out: int, beta: float = 4e-4):
        super().__init__()
        self.beta = beta
        self.c_in = c_in
        self.c_out = c_out

        self.linear = nn.Linear(c_in, c_out, True)
        self.register_buffer('mu', th.zeros(c_out,
                                            requires_grad=False))
        self.register_buffer('sigma', th.ones(c_out,
                                              requires_grad=False))
        # self.linear.reset_parameters()

    def forward(self, x: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        y_hat = self.linear(x)
        with th.no_grad():
            y = y_hat * self.sigma * self.mu
        return y

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
                 state_encoder: nn.Module,
                 env: gym.Env,
                 input_dim: int = 4,
                 hidden_dim: int = 8,
                 action_dim: int = 2):
        super().__init__()
        use_continuous_actions: bool = False
        self.env = env
        self.state_encoder = state_encoder

        if use_continuous_actions:
            self.policy = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, 2 * action_dim, bias=False)
            )
        else:
            self.policy = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, action_dim, bias=False)
            )
        self.pop_art = PopArtModule(
            hidden_dim,
            action_dim)

    def get_action_distribution(
            self, state: th.Tensor) -> th.distributions.Distribution:
        params = self.policy(state)
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

        done: bool = True
        prv_obs: th.Tensor = None

        for _ in range(128):
            # obs = env.reset() if done else prv_obs
            if prv_obs is None:
                prv_obs = env.reset()
            # obs = env.reset() if (prv_obs is None) else prv_obs

            with th.no_grad():
                state = self.state_encoder(prv_obs)
                dist = self.get_action_distribution(state)
                action = dist.sample()
                # NOTE(ycho): store `log_prob` for vtrace calculation
                log_prob = dist.log_prob(action)

            # NOTE(ycho): list(int(x)) is obviously a hack...
            obs, rew, done, info = env.step(list(int(x) for x in action))
            buf.append((prv_obs, action, log_prob, obs, rew, done))
            prv_obs = obs
        return buf

    def sample_steps(self, buf):
        return buf

    def _learn_step(self):
        # -- collect-rollouts --
        buf = self.interact()
        samples = self.sample_steps(buf)
        obs0s, actions, lp0, obs1s, rews, dones = zip(*samples)

        obs0s = th.stack(obs0s, axis=0)
        actions = th.stack(actions, axis=0)
        lp0 = th.stack(lp0, axis=0)
        obs1s = th.stack(obs1s, axis=0)
        rews = th.stack(rews, axis=0)
        dones = th.stack(dones, axis=0)

        state0s = self.state_encoder(obs0s)
        action_dist = self.get_action_distribution(state0s)
        lp1 = action_dist.log_prob(actions)
        baseline, normalized_baseline = self.pop_art(state0s)
        action = action_dist.rsample()
        log_prob = action_dist.log_prob(action)
        log_rho = (lp1 - lp0)  # log importance ratio and stuff
        # log_rho = th.clamp(log_rho, max=0.0)
        rho = th.exp(log_rho)

        bootstrap_value = baseline[-1]  # ???
        discounts = (~dones).float() * 0.99  # ???

        # Looks a lot like GAE
        #log_rhos: th.Tensor,
        #discounts: th.Tensor,
        #rewards: th.Tensor,
        #values: th.Tensor,
        #dim_t: int = 0,
        vs, pg_adv = vtrace_from_importance_weights(
            log_rho,
            (~dones).float() * 0.99,
            rewards,
            baseline,
            dim_t=1)  # NO idea, to be honest

        # normalized v_s
        nvs = (vs - self.pop_art.mu) / self.pop_art.sigma
        # policy gradient loss, valid_mask=?
        pg_loss = th.mean(-log_prob * pg_adv)
        # value baseline loss, valid_mask=?
        vb_loss = F.mse_loss(nvs, normalized_baseline)
        # entropy loss [...]
        ent_loss = -action_dist.entropy()

    def learn(self):
        for _ in range(16):
            self._learn_step()
