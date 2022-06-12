#!/usr/bin/env python3
"""Pop-Art + IMPALA agent implementation.

References:
    https://github.com/deepmind/scalable_agent
    https://github.com/fanyun-sun/pytorch-a2c-ppo-acktr
    https://github.com/aluscher/torchbeastpopart
    https://github.com/steffenvan/IMPALA-PopArt
"""

from typing import Tuple, Optional, Union, Iterable, Any, Dict, List
from abc import ABC, abstractmethod, abstractproperty
import numpy as np
import torch as th
import itertools
nn = th.nn
F = nn.functional
import gym
import einops
from torch.distributions import Normal, Independent, Categorical
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from pathlib import Path

from vtrace import vtrace_from_importance_weights
from feature import NetHackEncoder
from util import ensure_dir
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from collections import deque


def compute_policy_gradient_loss(logits, actions, advantages):
    """pg loss computation taken from reference."""
    cross_entropy = F.nll_loss(
        F.log_softmax(th.flatten(logits, 0, 1), dim=-1),
        # target=torch.flatten(actions, 0, 1),
        target=th.flatten(actions, 0, 2),
        reduction="none")
    # cross_entropy = cross_entropy.view_as(advantages)
    cross_entropy = cross_entropy.view_as(actions)
    return th.mean(cross_entropy * advantages.detach())


class PopArtModule(nn.Module):
    def __init__(self, c_in: int, c_out: int, beta: float = 4e-4):
        # NOTE(ycho): beta = update rate
        super().__init__()
        self.beta = beta
        self.c_in = c_in
        self.c_out = c_out

        self.baseline = nn.Linear(c_in, c_out, True)
        self.register_buffer('mu', th.zeros(c_out,
                                            requires_grad=False,
                                            dtype=th.float32))
        self.register_buffer('sigma', th.ones(c_out,
                                              requires_grad=False,
                                              dtype=th.float32))

    def forward(self, x: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        y_hat = self.baseline(x)
        with th.no_grad():
            y = y_hat * self.sigma + self.mu
        return (y, y_hat)

    def update_parameters(self,
                          vs: th.Tensor,
                          task: th.Tensor):
        """
        task: task ID
        """
        mu0 = self.mu
        sigma0 = self.sigma

        vs = vs * task  # TxBxK, K=num tasks
        n = task.sum((0, 1))
        mu = vs.sum((0, 1)) / n  # K, I guess per task
        nu = th.sum(th.square(vs), (0, 1)) / n
        sigma = th.sqrt(nu - mu**2)
        sigma = th.clamp(sigma, min=1e-4, max=1e+6)

        # NaN values are replaced with old values.
        mu = th.where(th.isnan(mu), mu0, mu)
        sigma = th.where(th.isnan(sigma), sigma0, sigma)

        # polyak average, I guess.
        self.mu = (1 - self.beta) * self.mu + self.beta * mu
        self.sigma = (1 - self.beta) * self.sigma + self.beta * sigma
        # th.lerp(self.mu, mu, self.beta, out=self.mu)
        # th.lerp(self.sigma, sigma, self.beta, out=self.sigma)

        # Update nn.Linear params
        self.baseline.weight.data = (
            self.baseline.weight.t() * sigma0 / self.sigma).t()
        self.baseline.bias.data = (
            sigma0 * self.baseline.bias + mu0 - self.mu) / self.sigma


class PopArtAgent(nn.Module):
    def __init__(self,
                 device: th.device,
                 state_encoder: NetHackEncoder,
                 env: gym.Env,
                 # == rollout length
                 num_interactions: int = 32,
                 hidden_dim: int = 64,
                 use_continuous_actions: bool = False,
                 gamma: float = 0.99):
        super().__init__()
        self.device = device
        self.env = env
        if isinstance(env, VecEnv):
            self.num_env = env.num_envs
        else:
            raise ValueError(
                'only vectorized envs are supported at the moment.')
        self.num_interactions = num_interactions
        self.state_encoder = state_encoder
        self.action_dim = self.env.action_space.n
        self.num_tasks = 1

        self.gamma = gamma

        # FIXME(ycho): should be able to make
        # batch_size != num_env.
        self.batch_size: int = self.num_env

        if use_continuous_actions:
            self.policy = nn.Sequential(
                nn.Linear(self.state_encoder.h_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 2 * self.action_dim, bias=False)
            )
        else:
            self.policy = nn.Sequential(
                nn.Linear(self.state_encoder.h_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, self.action_dim, bias=False)
            )
        self.pop_art = PopArtModule(self.state_encoder.h_dim, self.num_tasks)
        self.optimizer = th.optim.Adam(self.parameters(),
                                       weight_decay=1e-4)

        # NOTE(ycho): Replay buffer.
        self.buf_s0 = deque(maxlen=4 * self.num_interactions)
        self.buf_a = deque(maxlen=4 * self.num_interactions)
        self.buf_lp = deque(maxlen=4 * self.num_interactions)
        self.buf_s1 = deque(maxlen=4 * self.num_interactions)
        self.buf_r = deque(maxlen=4 * self.num_interactions)
        self.buf_d = deque(maxlen=4 * self.num_interactions)

        # NOTE(ycho): episode cache.
        self._cache = {}
        self.episode_rewards = deque(maxlen=64)

    def save_ckpt(self, ckpt_file: str):
        ckpt_file = Path(ckpt_file)
        ensure_dir(ckpt_file.parent)
        save_dict: Dict[str, Any] = {}
        save_dict['model'] = self.state_dict()
        if self.optimizer is not None:
            save_dict['optimizer'] = self.optimizer.state_dict()
        th.save(save_dict, str(ckpt_file))

    def load_ckpt(self, ckpt_file: str):
        ckpt_file = Path(ckpt_file)
        save_dict = th.load(str(ckpt_file))
        self.load_state_dict(save_dict['model'])
        if self.optimizer is not None:
            self.optimizer.load_state_dict(save_dict['optimizer'])

    def get_action_distribution(
            self, state: th.Tensor) -> th.distributions.Distribution:
        shape = state.shape
        state = state.reshape([-1, self.state_encoder.h_dim])
        params = self.policy(state)
        params = params.reshape(shape[:-1] + (self.action_dim,))
        # NOTE(ycho): below block is for continuous actions.
        # mu, std = einops.rearrange(params, '... (k d) -> k ... d', k=2)
        # return Independent(Normal(mu, std), 1)
        return Categorical(logits=params)

    def get_action(self, state: th.Tensor,
                   deterministic: bool = True):
        dist = self.get_action_distribution(state)
        if deterministic:
            # NOTE(ycho): does not generally work
            # for distribution variants, such as
            # Independent / Transformed / Cauchy ...
            if isinstance(dist, th.distributions.Categorical):
                return th.argmax(dist.logits, dim=-1)
            else:
                raise ValueError('unsuppored distribution')
        else:
            return dist.rsample()

    def reset(self):
        done = th.ones(self.num_env, dtype=bool, device=self.device)
        prv_obs = self.env.reset()
        core_state = self.state_encoder.initial_state(
            batch_size=self.num_env)
        core_state = tuple(s.to(self.device) for s in core_state)
        self._save_values(
            done=done,
            prv_obs=prv_obs,
            prv_core_state=core_state,
            cum_rew=np.zeros(done.shape, np.float32)
        )

    def _save_values(self, **kwds):
        self._cache.update(kwds)

    def _retrieve_values(self):
        return self._cache

    def interact(self):
        cache = self._retrieve_values()
        done, prv_obs, prv_core_state, cum_rew = (
            cache['done'], cache['prv_obs'],
            cache['prv_core_state'], cache['cum_rew'])

        # NOTE(ycho): save pre-rollout core state.
        initial_core_state = prv_core_state

        s0 = self.state_encoder.initial_state(
            batch_size=self.num_env)
        # FIXME(ycho):
        # I really don't like these tuple-based
        # state representations...
        s0 = tuple(s.to(self.device) for s in s0)

        for _ in range(self.num_interactions):
            with th.no_grad():
                state, core_state = self.state_encoder(
                    prv_obs, prv_core_state, th.as_tensor(
                        done, dtype=th.bool, device=self.device)[None])
                dist = self.get_action_distribution(state)
                action = dist.sample()
                # NOTE(ycho): store `log_prob` for vtrace calculation
                log_prob = dist.log_prob(action).squeeze()

                action = action.detach().cpu().numpy()
                log_prob = log_prob.detach().cpu().numpy()

            obs, rew, done, info = self.env.step(action)
            rew = np.asarray(rew, dtype=np.float32)
            cum_rew += rew
            # buf.append((prv_obs, action, log_prob, obs, rew, done))
            self.buf_s0.append(prv_obs)
            self.buf_a.append(action)
            self.buf_lp.append(log_prob)
            self.buf_s1.append(obs)
            self.buf_r.append(rew)
            self.buf_d.append(done)
            prv_obs = obs

            # NOTE(ycho): Need to reset initial states
            # for all environments that have terminated.
            # Since this operation might not be needed,
            if np.any(done):
                prv_core_state = tuple(
                    th.where(
                        th.as_tensor(
                            done[None, :, None],
                            device=s0[i].device), s0[i],
                        core_state[i]) for i in range(2))
                self.episode_rewards.extend(cum_rew[done])
                cum_rew = np.where(done, 0.0, cum_rew)
            else:
                prv_core_state = core_state

        self._save_values(
            done=done,
            prv_obs=prv_obs,
            prv_core_state=prv_core_state,
            cum_rew=cum_rew)
        return initial_core_state

    def sample_steps(self):
        index = np.random.randint(0,
                                  len(self.buf_s0) - self.num_interactions + 1)
        i0 = index
        i1 = index + self.num_interactions
        out = tuple(list(itertools.islice(d, i0, i1)) for d in (
            self.buf_s0, self.buf_a, self.buf_lp,
            self.buf_s1, self.buf_r, self.buf_d))
        return out
        #buf_t = []
        #for x in buf:
        #    buf_step = []
        #    for e in x:
        #        if isinstance(e, np.ndarray):
        #            buf_step.append(th.as_tensor(e))
        #        else:
        #            buf_step.append(e)
        #    buf_t.append(buf_step)
        #return buf_t

    def _learn_step(self):
        # -- collect-rollouts --
        initial_core_state = self.interact()
        samples = self.sample_steps()
        obs0s, actions, lp0, obs1s, rewards, dones = samples  # zip(*samples)

        # Format rollouts.
        # T, B, ...
        obs0s = self._stack_observations(obs0s)
        # actions = th.stack(actions, axis=0)
        actions = th.as_tensor(actions,  # dtype=th.int32,
                               device=self.device)
        # lp0 = th.stack(lp0, axis=0)
        lp0 = th.as_tensor(lp0, device=self.device)
        obs1s = self._stack_observations(obs1s)
        if not isinstance(rewards, th.Tensor):
            rewards = th.utils.data.dataloader.default_collate(
                rewards).to(dtype=th.float32, device=self.device)
        if not isinstance(dones, th.Tensor):
            dones = th.utils.data.dataloader.default_collate(
                dones).to(dtype=bool, device=self.device)

        state0s, _ = self.state_encoder(obs0s, initial_core_state, dones)
        action_dist = self.get_action_distribution(state0s)
        lp1 = action_dist.log_prob(actions)
        baseline, normalized_baseline = self.pop_art(state0s)

        # action = action_dist.sample()
        # action = action_dist.rsample()
        #logits = action_dist.logits
        #logits2d = einops.rearrange(logits, '... d -> (...) d')
        #action = th.multinomial(F.softmax(logits2d, dim=1),
        #                        num_samples=1).reshape(logits.shape[:-1])
        log_prob = action_dist.log_prob(actions)
        log_rho = (lp1 - lp0)  # log of importance ratio.

        # Derive discount factors at each step.
        # done => 0, not done => gamma
        # discounts = (~dones).to(th.float32) * self.gamma
        discounts = th.where(dones, 0.0, self.gamma).to(th.float32)

        # Vtrace calculation
        vs, pg_adv = vtrace_from_importance_weights(
            log_rho[..., None],
            discounts[..., None],
            rewards[..., None],
            baseline,
            0,
            normalized_baseline,
            self.pop_art.mu,
            self.pop_art.sigma
        )

        # normalized v_s
        nvs = (vs - self.pop_art.mu) / self.pop_art.sigma
        # policy gradient loss, valid_mask=?
        # pg_loss_1 = compute_policy_gradient_loss(
        # action_dist.logits, action[..., None], pg_adv)
        pg_loss = th.mean(-log_prob[..., None] * pg_adv)
        # print(F'{pg_loss_1.item()} != {pg_loss_2.item()}')

        # value baseline loss, valid_mask=?
        vb_loss = 0.5 * F.mse_loss(nvs, normalized_baseline)
        # entropy loss [...]
        ent_loss = 0.01 * -th.mean(action_dist.entropy())

        loss = (pg_loss + vb_loss + ent_loss)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()

        # TODO(ycho): `task` labels are required for pop-art.
        # Should we supply this in some way??
        if self.num_tasks == 1:
            tasks = th.ones_like(dones, dtype=th.float32)[..., None]
        else:
            tasks = F.one_hot(th.zeros_like(dones, dtype=th.long),
                              self.num_tasks).float()
        self.pop_art.update_parameters(vs, tasks)
        avg_eplen = (1.0 / th.mean(dones.to(th.float32)))
        scalars = {
            'pg_loss': pg_loss,
            'vb_loss': vb_loss,
            'ent_loss': ent_loss,
            'loss': loss,
            'reward': th.mean(rewards),
            'avg_eplen': avg_eplen,
            'episode_reward': th.mean(th.as_tensor(self.episode_rewards))
        }
        histos = {tag: value.grad.detach().cpu()
                  for (tag, value) in self.named_parameters()}
        return {
            'scalars': scalars,
            'histograms': histos
        }

    def _stack_observations(self, observations: Iterable
                            [Dict[str, th.Tensor]]) -> Dict[str, th.Tensor]:
        return th.utils.data.dataloader.default_collate(observations)
        #_temp_obs_stack: Dict[str, List[th.Tensor]] = dict()
        #keys: List[str] = list()
        #for observation in observations:
        #    for key, value in observation.items():
        #        if key not in _temp_obs_stack:
        #            _temp_obs_stack[key] = list()
        #            keys.append(key)
        #        _temp_obs_stack[key].append(value)
        #return {key: th.cat(_temp_obs_stack[key], dim=0).to(
        #    self.device) for key in keys}

    def learn(self,
              num_steps: int = 1000,
              log_dir: str = './log',
              save_steps: int = 100):
        """Learn for `num_steps` iterations.

        NOTE(ycho): actual number of env-steps
        = num_steps X num_interactions X num_envs.
        """
        self.reset()
        writer = SummaryWriter(log_dir)
        try:
            with tqdm(range(num_steps)) as pbar:
                for i in pbar:
                    # with th.autograd.detect_anomaly():
                    outputs = self._learn_step()
                    scalars: Dict[str, th.Tensor] = outputs['scalars']

                    loss = scalars['loss'].item()
                    pbar.set_description(F'loss={loss:.3f}')

                    global_step = i * self.num_env * self.num_interactions
                    for k, v in scalars.items():
                        writer.add_scalar(k, v.item(), global_step)
                    writer.add_scalar('reward', scalars['reward'].item(),
                                      global_step)

                    for k, v in outputs['histograms'].items():
                        writer.add_histogram(F'{k}/grad', v, global_step)

                    if (i % save_steps == 0):
                        self.save_ckpt(log_dir / F'nh-pa-{i:05d}.pt')
        finally:
            self.save_ckpt(log_dir / F'nh-pa-last.pt')
