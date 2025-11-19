import numpy as np 
from typing import Optional

def _validate_and_prepare(V_cal, w_cal, V_test, w_test):
    V_cal = np.asarray(V_cal, float).ravel()
    V_test = np.asarray(V_test, float).ravel()
    w_cal = np.asarray(w_cal, float).ravel()
    if V_cal.size != w_cal.size:
        raise ValueError("V_cal and w_cal must have the same length")
    if w_test is None:
        w_test = np.ones(V_test.shape[0], dtype=float)
    else:
        w_test = np.asarray(w_test, float).ravel()
        if w_test.size != V_test.size:
            raise ValueError("w_test must have same length as V_test")
    if np.any(~np.isfinite(V_cal)) or np.any(~np.isfinite(V_test)):
        raise ValueError("scores must be finite")
    if np.any(~np.isfinite(w_cal)) or np.any(w_cal <= 0):
        raise ValueError("w_cal must be positive and finite")
    if np.any(~np.isfinite(w_test)) or np.any(w_test < 0):
        raise ValueError("w_test must be nonnegative and finite")
    return V_cal, w_cal, V_test, w_test

def _split_conformal_left_tail(V_cal, w_cal, V_test, w_test, u):
    """
    p_j = [ w_cal(< t_j) + u_j * ( w_cal(<= t_j) - w_cal(< t_j) + w_test[j] ) ] / [ sum(w_cal) + w_test[j] ]
    Matches your formula exactly, vectorized.
    """
    order = np.argsort(V_cal, kind="mergesort")
    v = V_cal[order]
    w = w_cal[order]
    W = w.sum()
    pref = np.cumsum(w)

    # indices for strict (<) and weak (<=) comparisons
    idx_lt = np.searchsorted(v, V_test, side="left")  - 1
    idx_le = np.searchsorted(v, V_test, side="right") - 1
    idx_lt = np.clip(idx_lt, -1, v.size - 1)
    idx_le = np.clip(idx_le, -1, v.size - 1)

    w_lt = np.where(idx_lt >= 0, pref[idx_lt], 0.0)
    w_le = np.where(idx_le >= 0, pref[idx_le], 0.0)

    num = w_lt + u * ((w_le - w_lt) + w_test)
    den = W + w_test
    p = num / den
    return np.clip(p, 0.0, 1.0)

def weighted_pvalues_individual(V_cal_c, w_cal, V_test_c, w_test=None, randomize=False, random_state=0):
    """
    Left-tail split-conformal with per-test denominator sum(w_cal)+w_test[j].
    EXACT same formula as your loop; just vectorized & fast.
    """
    V_cal, w_cal, V_test, w_test = _validate_and_prepare(V_cal_c, w_cal, V_test_c, w_test)
    if randomize:
        u = np.random.default_rng(random_state).random(V_test.size)
    else:
        u = np.ones(V_test.size, dtype=float)
    return _split_conformal_left_tail(V_cal, w_cal, V_test, w_test, u)

def weighted_pvalues_for_selection(V_cal, w_cal, V_test_c, w_test=None, randomize=False, random_state=0):
    """
    Identical to weighted_pvalues_individual by your definition (same formula).
    Kept as a separate name for clarity in your pipeline.
    """
    V_cal, w_cal, V_test, w_test = _validate_and_prepare(V_cal, w_cal, V_test_c, w_test)
    if randomize:
        u = np.random.default_rng(random_state).random(V_test.size)
    else:
        u = np.ones(V_test.size, dtype=float)
    return _split_conformal_left_tail(V_cal, w_cal, V_test, w_test, u)
 

 