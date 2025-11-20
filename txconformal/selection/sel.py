import numpy as np
import pandas as pd

def bh_threshold(p: np.ndarray, alpha: float):
    m = p.size
    order = np.argsort(p, kind="mergesort")
    ps = p[order]
    thresh = alpha * (np.arange(1, m+1) / m)
    passed = ps <= thresh
    if not np.any(passed):
        return 0.0, np.array([], dtype=int)
    k = np.max(np.where(passed)[0]) + 1
    t = thresh[k-1]
    idx = order[:k]
    return float(t), idx

def fp_budget_threshold(p: np.ndarray, K: int):
    m = p.size
    order = np.argsort(p, kind="mergesort")
    ps = p[order]
    fp_est = m * ps
    ok = np.where(fp_est <= K)[0]
    if ok.size == 0:
        return 0.0, np.array([], dtype=int)
    k = ok[-1]
    return float(ps[k]), order[:k+1]


def tp_min_threshold(p: np.ndarray, K: int):
    m = p.size
    order = np.argsort(p, kind="mergesort") # indices in ascending order
    ps = p[order]
    est_tp = np.arange(1, m+1) - m * ps
    k = np.argmax(est_tp >= K)
    if est_tp[k] < K:
        return 1.0, order  # select all if target unmet
    return float(ps[k]), order[:k+1]


def estimate_topk_fp(p: np.ndarray, K: int, CI: bool = False, 
                     w_cal: np.ndarray = None, hat_mu_cal: np.ndarray = None, hat_mu_test: np.ndarray = None):
    order = np.sort(p, kind="mergesort")
    kth = order[K-1]
    est_fp = len(p) * kth / K

    if CI:
        df = pd.DataFrame({'p': p, 'hat_mu_test': hat_mu_test})
        df.sort_values('p', inplace=True)
        threshold = df.iloc[K-1]['hat_mu_test']
        std_cal = np.std(w_cal * (1*(hat_mu_cal >= threshold )- K /len(hat_mu_test)) ) * len(hat_mu_test) / K /np.sqrt(len(w_cal))
        std_test = np.sqrt(est_fp * K * (1- est_fp * K/ len(p)))
        std_both = np.sqrt(std_cal**2 + std_test**2)
        ci_low = est_fp  - 1.6 * std_both / K 
        ci_high = est_fp  + 1.6 * std_both / K

    return float(est_fp), (ci_low, ci_high) if CI else float(est_fp)