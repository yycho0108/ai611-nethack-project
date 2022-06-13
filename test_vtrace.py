# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for V-trace.

For details and theory see:

"IMPALA: Scalable Distributed Deep-RL with
Importance Weighted Actor-Learner Architectures"
by Espeholt, Soyer, Munos et al.
"""

from typing import Optional
import numpy as np
import torch as th
import unittest

from vtrace import vtrace_from_importance_weights


@th.no_grad()
def vtrace_compat(
        log_rhos: th.Tensor,
        discounts: th.Tensor,
        rewards: th.Tensor,
        values: th.Tensor,
        dim_t: int = 0,
        bootstrap_value: Optional[th.Tensor] = None,
        max_rho: Optional[float] = 1.0,
        max_pg_rho: Optional[float] = 1.0):
    """vtrace function with compatible test arguments."""
    log_rhos = th.as_tensor(log_rhos)
    discounts = th.as_tensor(discounts)
    rewards = th.as_tensor(rewards)
    values = th.as_tensor(values)
    if bootstrap_value is not None:
        bootstrap_value = th.as_tensor(bootstrap_value)
    return vtrace_from_importance_weights(
        log_rhos, discounts, rewards, values, dim_t=0,
        bootstrap_value=bootstrap_value, max_rho=max_rho,
        max_pg_rho=max_pg_rho,
        use_popart=False)


def _shaped_arange(*shape):
    """Runs np.arange, converts to float and reshapes."""
    return np.arange(np.prod(shape), dtype=np.float32).reshape(*shape)


def _softmax(logits):
    """Applies softmax non-linearity on inputs."""
    return np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)


def _ground_truth_calculation(discounts, log_rhos, rewards, values,
                              bootstrap_value, max_rho,
                              max_pg_rho):
    """Calculates the ground truth for V-trace in Python/Numpy."""
    vs = []
    seq_len = len(discounts)
    rhos = np.exp(log_rhos)
    cs = np.minimum(rhos, 1.0)
    clipped_rhos = rhos
    if max_rho:
        clipped_rhos = np.minimum(rhos, max_rho)
    clipped_pg_rhos = rhos
    if max_pg_rho:
        clipped_pg_rhos = np.minimum(rhos, max_pg_rho)

    # This is a very inefficient way to calculate the V-trace ground truth.
    # We calculate it this way because it is close to the mathematical notation of
    # V-trace.
    # v_s = V(x_s)
    #       + \sum^{T-1}_{t=s} \gamma^{t-s}
    #         * \prod_{i=s}^{t-1} c_i
    #         * \rho_t (r_t + \gamma V(x_{t+1}) - V(x_t))
    # Note that when we take the product over c_i, we write `s:t` as the notation
    # of the paper is inclusive of the `t-1`, but Python is exclusive.
    # Also note that np.prod([]) == 1.
    values_t_plus_1 = np.concatenate(
        [values, bootstrap_value[:]], axis=0)
    for s in range(seq_len):
        v_s = np.copy(values[s])  # Very important copy.
        for t in range(s, seq_len):
          v_s += (np.prod(discounts[s:t], axis=0) *
                  np.prod(cs[s:t], axis=0) *
                  clipped_rhos[t] *
                  (rewards[t] +
                   discounts[t] *
                   values_t_plus_1[t +
                                   1] -
                   values[t]))
        vs.append(v_s)
    vs = np.stack(vs, axis=0)
    pg_advantages = (
        clipped_pg_rhos * (rewards + discounts * np.concatenate(
            [vs[1:], bootstrap_value[:]], axis=0) - values))
    return (vs, pg_advantages)


class VtraceTest(unittest.TestCase):

  def test_vtrace(self):
      for batch_size in [1, 5]:
        """Tests V-trace against ground truth data calculated in python."""
        seq_len = 5

        # Create log_rhos such that rho will span from near-zero to above the
        # clipping thresholds. In particular, calculate log_rhos in [-2.5, 2.5),
        # so that rho is in approx [0.08, 12.2).
        log_rhos = _shaped_arange(seq_len, batch_size) / (batch_size * seq_len)
        log_rhos = 5 * (log_rhos - 0.5)  # [0.0, 1.0) -> [-2.5, 2.5).
        values = {
            'log_rhos': log_rhos,
            # T, B where B_i: [0.9 / (i+1)] * T
            'discounts':
                np.array([[0.9 / (b + 1)
                           for b in range(batch_size)]
                          for _ in range(seq_len)]),
            'rewards':
                _shaped_arange(seq_len, batch_size),
            'values':
                _shaped_arange(seq_len, batch_size) / batch_size,
            'bootstrap_value':
                _shaped_arange(batch_size)[None] + 1.0,
            'max_rho': 3.7,
            'max_pg_rho': 2.2,
        }

        output_v = vtrace_compat(**values)

        ground_truth_v = _ground_truth_calculation(**values)
        for a, b in zip(ground_truth_v, output_v):
            b = b.detach().cpu().numpy()
            d = np.max(np.abs(a - b))
            # self.assertAlmostEquals(a, b)
            self.assertLess(d, 1e-5)

  def test_higher_rank_inputs_for_importance_weights(self):
      """Checks support for additional dimensions in inputs."""
      T: int = 4
      B: int = 2
      placeholders = {
          'log_rhos': th.zeros(dtype=th.float32, size=[T, B, 1]),
          'discounts': th.zeros(dtype=th.float32, size=[T, B, 1]),
          'rewards': th.zeros(dtype=th.float32, size=[T, B, 42]),
          'values': th.zeros(dtype=th.float32, size=[T, B, 42]),
          'bootstrap_value': th.zeros(dtype=th.float32, size=[1, B, 42])
      }
      output = vtrace_compat(**placeholders)
      vs, pg_adv = output
      self.assertEqual(vs.shape[-1], 42)

  def test_inconsistent_rank_inputs_for_importance_weights(self):
      """Test one of many possible errors in shape of inputs."""
      T: int = 4
      B: int = 2
      placeholders = {
          'log_rhos': th.zeros(dtype=th.float32, size=[T, B, 1]),
          'discounts': th.zeros(dtype=th.float32, size=[T, B, 1]),
          'rewards': th.zeros(dtype=th.float32, size=[T, B, 42]),
          'values': th.zeros(dtype=th.float32, size=[T, B, 42]),
          # Should be [None, 42].
          'bootstrap_value': th.zeros(dtype=th.float32, size=[B])
      }
      with self.assertRaises(RuntimeError):
          vtrace_compat(**placeholders)


if __name__ == '__main__':
  unittest.main()
