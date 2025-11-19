from typing import Iterable, Tuple, Dict, Any, Optional
import warnings
import numpy as np
from .ebal import ebal

def _log(msg, verbose, level=0):
    if verbose >= level:
        print(msg)

def retry_entropy_balancing(
    phi_calib: np.ndarray,
    phi_test: np.ndarray,
    *,
    force_features_calib: Optional[np.ndarray] = None,
    force_features_test: Optional[np.ndarray] = None,
    backup_features_calib: Optional[list[np.ndarray]] = None,
    backup_features_test: Optional[list[np.ndarray]] = None,
    # grid over (PCA_var_bound, PCA_ratio_bound)
    grid: Iterable[Tuple[float, float, float]] = (
        (0.1, 1e-5), (0.5, 1e-5), (1.0, 1e-5),
        (0.1, 1e-4), (0.5, 1e-4), (1.0, 1e-4),
        (0.1, 1e-3), (0.5, 1e-3), (1.0, 1e-3),
    ),
    tolerances: Iterable[float] = (1e-5, 1e-4, 1e-3),
    max_iterations: int = 500, 
    step_cap: float = 1.0,
    print_level: Optional[int] = None,
    random_state: Optional[int] = None,
    add_no_pca_fallback: bool = True,   # final attempt without PCA if all pairs fail
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Returns (weights_calib, meta). Raises RuntimeError if all attempts fail.
    """
    last_err: Optional[Exception] = None
    n_cal = phi_calib.shape[0]
    n_tst = phi_test.shape[0]

    if print_level is None:
        print_level = -1

    if print_level <= 0:
        warnings.filterwarnings("ignore", category=RuntimeWarning)

    # Prepare pooled design blocks to match the legacy API of ebal
    X_soft = np.vstack([phi_calib, phi_test])
    X_force = None
    if force_features_calib is not None and force_features_test is not None:
        X_force = np.vstack([force_features_calib, force_features_test])
    if_test = np.r_[np.zeros(n_cal, dtype=int), np.ones(n_tst, dtype=int)]
    
    # X_backup = None
    if backup_features_calib is not None and backup_features_test is not None: 
        if len(backup_features_calib) != len(backup_features_test):
            raise ValueError("backup_features_calib and backup_features_test must have the same length!")
        # X_backup = np.vstack([backup_features_calib, backup_features_test])

    success = False

    for tol in tolerances:
        for var_b, ratio_b in grid:
            try:
                _log(f"Running ({var_b}, {ratio_b}) at tolerance {tol}", print_level, level=1)
                eb = ebal(
                    PCA=True,
                    PCA_var_bound=float(var_b),
                    PCA_ratio_bound=float(ratio_b), 
                    max_iterations=int(max_iterations),
                    constraint_tolerance=float(tol),
                    step_cap=float(step_cap),
                    print_level=print_level,
                    random_state=random_state,
                )
                out = eb.ebalance(if_test, X_soft, X_force)
                if out["converged"]:
                    _log(f"Converged with ({var_b}, {ratio_b}) at tolerance {tol}", print_level, level=0)
                    success = True
                    w_cal = out["w"][:n_cal]
                    meta = {
                        "converged": True,
                        "maxdiff": float(out["maxdiff"]),
                        "pca_enabled": True,
                        "pca_var_bound": float(var_b),
                        "pca_ratio_bound": float(ratio_b), 
                        "design_cols": int(out["calib_x"].shape[1]),
                        "random_state": random_state,
                        "calib_x": out["calib_x"],
                        "test_x": out["test_x"],
                    }
                    return w_cal, meta
                # treat non-convergence as failure and continue
                last_err = RuntimeError(f"not converged at ({var_b}, {ratio_b}); maxdiff={out['maxdiff']:.2e}")
            except Exception as e:
                last_err = e
                continue
    
    if not success:
        _log("All PCA attempts failed, trying fallback...", print_level, level=0)
        if backup_features_calib is None:
            raise RuntimeError("No backup features available for fallback")
        
        for ii in range(len(backup_features_calib)):
            backup_features_calib = backup_features_calib[ii]
            backup_features_test = backup_features_test[ii]

            X_backup = np.vstack([backup_features_calib, backup_features_test])
            for var_b, ratio_b in grid:
                for tol in tolerances:
                    eb = ebal(
                            PCA=True,
                            PCA_var_bound=float(var_b),
                            PCA_ratio_bound=float(ratio_b), 
                            max_iterations=int(max_iterations),
                            constraint_tolerance=float(tol),
                            step_cap=float(step_cap),
                            print_level=int(print_level),
                            random_state=random_state,
                            )
                    out = eb.ebalance(if_test, X_backup, None)
                    if out["converged"]:
                        _log(f"Converged with ({var_b}, {ratio_b}) at tolerance {tol}", print_level, level=0)
                        success = True
                        w_cal = out["w"][:n_cal]
                        meta = {
                            "converged": True,
                            "maxdiff": float(out["maxdiff"]),
                            "pca_enabled": True,
                            "pca_var_bound": float(var_b),
                            "pca_ratio_bound": float(ratio_b), 
                            "design_cols": int(out["calib_x"].shape[1]),
                            "random_state": random_state,
                            "calib_x": out["calib_x"],
                            "test_x": out["test_x"],
                        }
                        return w_cal, meta

    
    # 2) Optional final fallback: disable PCA and keep same gating
    if add_no_pca_fallback:
        for tol in tolerances:
            try:
                eb = ebal(
                    PCA=False, 
                    max_iterations=int(max_iterations),
                    constraint_tolerance=float(tol),
                    step_cap=float(step_cap),
                    print_level=print_level,
                    random_state=random_state,
                )
                out = eb.ebalance(if_test, X_soft, X_force)
                if out["converged"]:
                    w_cal = out["w"][:n_cal]
                    _log(f"Converged without PCA at tolerance {tol}", print_level)
                    meta = {
                        "converged": True,
                        "maxdiff": float(out["maxdiff"]),
                        "pca_enabled": False,
                        "pca_var_bound": None,
                        "pca_ratio_bound": None, 
                        "design_cols": int(out["calib_x"].shape[1]),
                        "random_state": random_state,
                        "calib_x": out["calib_x"],
                        "test_x": out["test_x"],
                    }
                    return w_cal, meta
                last_err = RuntimeError(f"not converged with PCA=False; maxdiff={out['maxdiff']:.2e}")
            except Exception as e:
                last_err = e

    raise RuntimeError(f"Entropy balancing failed after retries: {last_err}")