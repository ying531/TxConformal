import numpy as np

def make_bins(f_calib: np.ndarray, f_test: np.ndarray, n_bins: int = 10, edges=None):
    """
    Legacy quantile bins matching legacy.utils.create_qt:
      - compute pooled quantiles at {0.1, 0.2, ..., 0.9}
      - bin j indicator equals 1 if value lies in (q_{j-1}, q_j], with q_0=-inf
      - values above q_{n_bins-1} receive all-zero row (consistent with legacy code)
    """
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2 to build quantile bins")

    if edges is None:
        probs = np.linspace(1 / n_bins, (n_bins - 1) / n_bins, n_bins - 1)
        f_all = np.concatenate([f_calib, f_test])
        edges = np.quantile(f_all, probs)

    def one_hot(f):
        bins = []
        prev = -np.inf
        for q in edges:
            bins.append(((f > prev) & (f <= q)).astype(float))
            prev = q
        return np.column_stack(bins) if bins else np.zeros((f.shape[0], 0), dtype=float)

    return one_hot(np.asarray(f_calib, float)), one_hot(np.asarray(f_test, float)), edges
