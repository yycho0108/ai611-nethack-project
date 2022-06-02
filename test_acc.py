import torch as th


def vtrace1(cs, deltas, discounts, bootstrap_value):
    acc = th.zeros_like(bootstrap_value)
    result = []
    for t in range(discounts.shape[0] - 1, -1, -1):
        acc = deltas[t] + discounts[t] * cs[t] * acc
        result.append(acc)
    result.reverse()
    return result


def vtrace2(cs, deltas, discounts, bootstrap_value):
    th.cumprod(discounts*cs, dim=0) + delta


def main():
    th.manual_seed(0)
    th.cuda.manual_seed(0)

    T: int = 8  # rollout horizon
    B: int = 1

    values = th.rand(size=(T, B))
    rewards = th.rand(size=(T, B))
    bootstrap_value = th.rand((B,)).ravel()
    rhos = th.rand(size=(T, B))
    discounts = th.randint(2, size=(T, B), dtype=th.bool).float() * 0.99

    v0 = values
    v1 = th.cat([values[1:], bootstrap_value[None]], dim=0)
    delta = rhos * (rewards + discounts * v1 - v0)
    out1 = vtrace1(rhos, delta, discounts, bootstrap_value)
    print(out1)


if __name__ == '__main__':
    main()
