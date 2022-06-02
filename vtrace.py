#!/usr/bin/env python3

import torch as th
from typing import Optional, Tuple


@th.no_grad()
def vtrace_from_importance_weights(
        log_rhos: th.Tensor,
        discounts: th.Tensor,
        rewards: th.Tensor,
        values: th.Tensor,
        dim_t: int = 0,

        bootstrap_value: Optional[th.Tensor] = None,
        max_rho: Optional[float] = 1.0,
        max_pg_rho: Optional[float] = 1.0) -> Tuple[th.Tensor, th.Tensor]:

    if bootstrap_value is None:
        bootstrap_value = th.narrow(values, dim_t,
                                    values.shape[dim_t] - 1, 1)

    # Clamping importance ratios here and there
    # TODO(ycho): Figure out the difference between
    # `cs` and `clipped_rhos`.
    rhos = th.exp(log_rhos)
    if max_rho is not None:
        clipped_rhos = th.clamp_max(rhos, max_rho)
    else:
        clipped_rhos = rhos
    cs = th.clamp_max(rhos, 1.0)

    # Compute TD target
    v_prv = values
    v_nxt = th.cat((
        th.narrow(values, dim_t, 1, values.shape[dim_t] - 1),
        bootstrap_value), dim_t)
    deltas = clipped_rhos * (rewards + discounts * v_nxt - v_prv)

    # Accumulate.
    acc = th.zeros_like(bootstrap_value)
    result = []
    for t in range(discounts.shape[dim_t] - 1, -1, -1):
        delta = th.narrow(deltas, dim_t, t, 1)
        discount = th.narrow(discounts, dim_t, t, 1)
        cs = th.narrow(cs, dim_t, t, 1)
        acc = delta + discount * cs * acc
        result.append(acc)
    result.reverse()
    vs = values + th.cat(result, dim=dim_t)

    # Advantage...
    vs_nxt = th.cat((
        th.narrow(vs, dim_t, 1, values.shape[dim_t] - 1),
        bootstrap_value), dim_t)
    if max_pg_rho is not None:
        clipped_pg_rhos = th.clamp_max(rhos, max_pg_rho)
    else:
        clipped_pg_rhos = rhos
    pg_adv = clipped_pg_rhos * (rewards + discounts * vs_nxt - v_prv)
    return (vs, pg_adv)
