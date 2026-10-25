"""配对统计:McNemar 精确检验与 bootstrap 置信区间。"""
from __future__ import annotations

import math
import random


def _binom_cdf(k: int, n: int, p: float = 0.5) -> float:
    return sum(math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i)) for i in range(k + 1))


def mcnemar_exact(baseline: list[bool], candidate: list[bool]) -> dict:
    if len(baseline) != len(candidate):
        raise ValueError("baseline/candidate 长度不一致")
    b = sum(1 for left, right in zip(baseline, candidate) if not left and right)
    c = sum(1 for left, right in zip(baseline, candidate) if left and not right)
    n = b + c
    p_value = 1.0 if n == 0 else min(1.0, 2 * _binom_cdf(min(b, c), n, 0.5))
    return {"b": b, "c": c, "n": n, "statistic": min(b, c), "p_value": p_value}


def bootstrap_ci(
    values: list[float],
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 20261024,
) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(sample) / len(sample))
    means.sort()
    lower = means[int((alpha / 2) * (len(means) - 1))]
    upper = means[int((1 - alpha / 2) * (len(means) - 1))]
    return (lower, upper)
