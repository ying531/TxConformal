from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TopKEstimate:
    """Point estimates for a shortlist containing the K smallest p-values."""

    k: int
    m: int
    p_boundary: float
    fp_count: float
    fdp: float


def _validated_pvalues(p: np.ndarray) -> np.ndarray:
    values = np.asarray(p, dtype=float).ravel()
    if values.size == 0:
        raise ValueError("p must contain at least one p-value")
    if np.any(~np.isfinite(values)):
        raise ValueError("p-values must be finite")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p-values must lie in [0, 1]")
    return values


def _integer_valued_k(K, *, name: str = "K") -> int:
    """Coerce integer-valued NumPy/Python scalars while rejecting fractions."""

    if isinstance(K, (bool, np.bool_)) or not isinstance(
        K, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be an integer or integer-valued float")
    numeric = float(K)
    if not np.isfinite(numeric) or not numeric.is_integer():
        raise TypeError(f"{name} must be an integer or integer-valued float")
    return int(numeric)


def _validated_k(K: int, m: int) -> int:
    k = _integer_valued_k(K)
    if not 1 <= k <= m:
        raise ValueError(f"K must satisfy 1 <= K <= {m}")
    return k


def bh_threshold(p: np.ndarray, alpha: float):
    values = _validated_pvalues(p)
    if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    m = values.size
    order = np.argsort(values, kind="mergesort")
    ps = values[order]
    thresh = alpha * (np.arange(1, m + 1) / m)
    passed = ps <= thresh
    if not np.any(passed):
        return 0.0, np.array([], dtype=int)
    k = np.max(np.where(passed)[0]) + 1
    return float(thresh[k - 1]), order[:k]


def fp_budget_threshold(p: np.ndarray, K: int):
    values = _validated_pvalues(p)
    k_budget = _integer_valued_k(K)
    if k_budget < 0:
        raise ValueError("K must be nonnegative")
    order = np.argsort(values, kind="mergesort")
    ps = values[order]
    ok = np.where(values.size * ps <= k_budget)[0]
    if ok.size == 0:
        return 0.0, np.array([], dtype=int)
    k = ok[-1]
    return float(ps[k]), order[: k + 1]


def tp_min_threshold(p: np.ndarray, K: int):
    values = _validated_pvalues(p)
    k_target = _integer_valued_k(K)
    if k_target < 0:
        raise ValueError("K must be nonnegative")
    m = values.size
    order = np.argsort(values, kind="mergesort")
    ps = values[order]
    est_tp = np.arange(1, m + 1) - m * ps
    reached = np.flatnonzero(est_tp >= k_target)
    if reached.size == 0:
        return 1.0, order
    k = int(reached[0])
    return float(ps[k]), order[: k + 1]


def estimate_topk(p: np.ndarray, K: int) -> TopKEstimate:
    """Estimate FP count and FDP for the K smallest candidate p-values.

    This function deliberately returns point estimates only. A generic uncertainty
    interval requires an estimand-specific variance derivation and is not supplied
    by the package.
    """

    values = _validated_pvalues(p)
    k = _validated_k(K, values.size)
    order = np.argsort(values, kind="mergesort")
    p_boundary = float(values[order[k - 1]])
    fp_count = float(values.size * p_boundary)
    return TopKEstimate(
        k=k,
        m=int(values.size),
        p_boundary=p_boundary,
        fp_count=fp_count,
        fdp=float(fp_count / k),
    )


def estimate_topk_fp(
    p: np.ndarray,
    K: int,
    CI: bool = False,
    **_ignored,
) -> float:
    """Backward-compatible FDP point estimator.

    ``CI=True`` previously returned a scale-inconsistent interval and is now
    rejected. Use :func:`estimate_topk` to obtain both the FP-count and FDP
    point estimates explicitly.
    """

    if CI:
        raise NotImplementedError(
            "Generic top-K confidence intervals are not implemented; "
            "the former interval mixed FP-count and FDP scales."
        )
    return estimate_topk(p, K).fdp
