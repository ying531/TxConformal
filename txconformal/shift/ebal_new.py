# Usage (drop-in):
#   from ebal_new import ebal
#   out = ebal(PCA=True, PCA_var_bound=1.0, PCA_ratio_bound=1e-3,
#              max_iterations=500, constraint_tolerance=1e-8, print_level=0
#             ).ebalance(if_test, X_soft, X_force)
#   gratefully inspired by Eddie Yang's implementation
# Returns:
#   {
#     "converged": bool,
#     "maxdiff": float,              # max abs moment gap
#     "w": np.ndarray,               # length n_total (calib weights then test weights)
#     "calib_x": np.ndarray,         # intercept + features for calib
#     "test_x": np.ndarray,          # intercept + features for test
#   }

from typing import Optional, Dict, Any, Tuple
import numpy as np
from numpy.linalg import solve
from sklearn.decomposition import PCA

class ebal:
    def __init__(
        self,
        *,
        # PCA controls
        PCA: bool = True,
        PCA_var_bound: float = 1.0,
        PCA_ratio_bound: float = 1e-3,
        # solver controls
        max_iterations: int = 500,
        constraint_tolerance: float = 1e-8,
        step_cap: float = 1.0,
        # misc
        print_level: int = 0, 
        random_state: Optional[int] = None,
    ) -> None:
        self.PCA = bool(PCA)
        self.PCA_var_bound = float(PCA_var_bound)
        self.PCA_ratio_bound = float(PCA_ratio_bound)
        self.max_iterations = int(max_iterations)
        self.constraint_tolerance = float(constraint_tolerance)
        self.step_cap = float(step_cap)
        self.print_level = int(print_level) 
        self.random_state = random_state

    # ---------- public API (compatible) ----------
    def ebalance(
        self,
        if_test: np.ndarray,
        X_soft: np.ndarray,
        X_force: Optional[np.ndarray] = None,
        base_weight: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        if_test = np.asarray(if_test).astype(int).ravel()
        X_soft = np.asarray(X_soft, dtype=float)
        n_total, d = X_soft.shape
        if if_test.shape[0] != n_total:
            raise ValueError("if_test and X_soft must have the same number of rows")

        # split into calibration and test
        calib_mask = (if_test == 0)
        test_mask  = (if_test == 1)
        if calib_mask.sum() == 0 or test_mask.sum() == 0:
            raise ValueError("Need both calibration and test rows indicated by if_test")

        Xc_soft = X_soft[calib_mask]
        Xt_soft = X_soft[test_mask]

        # optional force block
        Fc = Ft = None
        if X_force is not None:
            X_force = np.asarray(X_force, dtype=float)
            if X_force.shape[0] != n_total:
                raise ValueError("X_force must have the same number of rows as X_soft")
            Fc = X_force[calib_mask]
            Ft = X_force[test_mask]

        # 1) center + optional PCA on the soft block
        Xc, Xt = self._soft_block(Xc_soft, Xt_soft)

        # 2) optionally append "force" columns if they add residual std
        Xc, Xt = self._maybe_append_force_cols(Xc, Xt, Fc, Ft)

        # 3) add intercepts and build targets (test means)
        n_cal, n_tst = Xc.shape[0], Xt.shape[0]
        calib_x = np.concatenate([np.ones((n_cal, 1)), Xc], axis=1)
        test_x  = np.concatenate([np.ones((n_tst, 1)), Xt], axis=1)
        tr_total = test_x.mean(axis=0)

        # 4) base weights on calib
        if base_weight is None:
            base_w = np.ones(n_cal, dtype=float)
        else:
            base_w = np.asarray(base_weight, dtype=float).ravel()
            if base_w.shape[0] != n_cal:
                raise ValueError("base_weight must have length equal to number of calibration rows")

        # 5) solve dual (Newton + backtracking, overflow-safe)
        # w_cal, converged, maxdiff, iters = self._solve_dual_newton(calib_x, target, base_w)
        w_cal, converged, maxdiff, iters, _ = self._eb_solver(tr_total, calib_x, base_w)

        # 6) package full weight vector (test weights set to uniform in test)
        w = np.empty(n_total, dtype=float)
        w[calib_mask] = w_cal
        w[test_mask]  = 1.0 / n_tst

        return {
            "converged": bool(converged),
            "maxdiff": float(maxdiff),
            "w": w,
            "calib_x": calib_x,
            "test_x": test_x,
        }

    # ---------- helpers ----------
    def _soft_block(self, Xc_soft: np.ndarray, Xt_soft: np.ndarray):
        # center using pooled mean (calib + test)
        mean = np.vstack([Xc_soft, Xt_soft]).mean(axis=0, keepdims=True)
        Xc = Xc_soft - mean
        Xt = Xt_soft - mean

        if not self.PCA:
            return Xc, Xt

        # PCA on pooled centered data
        pca = PCA(svd_solver="full", random_state=self.random_state)
        Z_all = pca.fit_transform(np.vstack([Xc, Xt]))
        var = pca.explained_variance_
        ratio = pca.explained_variance_ratio_

        keep = (var >= self.PCA_var_bound) | (ratio >= self.PCA_ratio_bound)
        if not np.any(keep):               # keep at least one component
            keep = np.zeros_like(var, dtype=bool)
            keep[0] = True

        Zc = Z_all[: Xc.shape[0], :][:, keep]
        Zt = Z_all[Xc.shape[0] :, :][:, keep]
        return Zc, Zt

    def _maybe_append_force_cols(
        self,
        Xc_soft: np.ndarray,
        Xt_soft: np.ndarray,
        Fc: Optional[np.ndarray],
        Ft: Optional[np.ndarray],
    ):
        if Fc is None or Ft is None:
            return Xc_soft, Xt_soft
        X_soft = np.vstack([Xc_soft, Xt_soft])
        # center force by pooled mean
        mu = np.vstack([Fc, Ft]).mean(axis=0, keepdims=True)
        X_force = np.vstack([Fc, Ft]) - mu
        
        # PCA on X_force if self.PCA
        if self.PCA and Fc.shape[1] > 1:
            pca = PCA(svd_solver="full", random_state=self.random_state)
            X_force = pca.fit_transform(X_force)
            var = pca.explained_variance_
            ratio = pca.explained_variance_ratio_
            keep = (var >= 0.001) | (ratio >= 0.0001) 
            if len(keep) == 0: # nothing is kept, return original Xc_soft, Xt_soft
                return Xc_soft, Xt_soft
            else:
                X_force = X_force[:, keep]
            
        # determine whether it's necessary to enforce the X_force
        hat_theta = np.linalg.inv(X_soft.T @ X_soft) @ X_soft.T @ X_force
        hat_xforce = X_soft @ hat_theta
        if len(X_force.shape) == 1: # 1d
            if np.std(X_force - hat_xforce) > 0.05 * np.std(X_force):
                X = np.concatenate([X_soft, X_force], axis = 1)
            else:
                return Xc_soft, Xt_soft
        else:
            id_force_preserve = [ii for ii in range(X_force.shape[1]) if np.std(X_force[:,ii] - hat_xforce[:,ii]) > 0.05 * np.std(X_force[:,ii])]
            X_force = X_force[:, id_force_preserve] 
            X = np.concatenate([X_soft, X_force], axis = 1)


        Xc = X[:Xc_soft.shape[0], :]
        Xt = X[Xc_soft.shape[0] :, :]
        return Xc, Xt
     
    
    # ---------------- core solver ----------------

    def _eb_solver(
        self,
        tr_total: np.ndarray,    # (p,)  test mean moments of [1, X]
        calib_x: np.ndarray,     # (n,p) intercept + features (scaled consistently with tr_total)
        base_weight: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, bool, float, int, np.ndarray]:
        """
        Clean-room Newton solver for the EB dual:
          w_i ∝ b_i * exp( (calib_x)_i · λ ),  sum_i w_i = 1,  E_w[calib_x] = tr_total
        Returns calibration weights that sum to 1, convergence info, and λ.
        """
        X = np.asarray(calib_x, dtype=float)
        target = np.asarray(tr_total, dtype=float)
        n, p = X.shape

        if base_weight is None:
            b = np.ones(n, dtype=float)
        else:
            b = np.asarray(base_weight, dtype=float).reshape(-1)
            if b.shape[0] != n:
                raise ValueError("base_weight must have length n_calib")

        tol = self.constraint_tolerance
        max_iter = self.max_iterations
        step_cap = self.step_cap

        lam = np.zeros(p, dtype=float)
        converged = False
        maxgap = np.inf

        eps = 1e-16

        def weights_from(lv):
            z = X @ lv
            z -= np.max(z)                 # overflow-safe
            w = b * np.exp(z)
            s = w.sum() + eps
            return w, s

        for it in range(max_iter):
            w, s = weights_from(lam)
            mom = (w[:, None] * X).sum(axis=0) / s    # mean moments
            grad = mom - target
            maxgap = float(np.max(np.abs(grad)))
            if self.print_level and it % 100 == 0:
                print(f"[EB] iter={it} max_gap={maxgap:.3e}")

            if maxgap < tol:
                converged = True
                break

            # Hessian of mean map: covariance under w
            CW = X * w[:, None]
            Ct_w = CW.sum(axis=0)
            H = (X.T @ CW) / s - np.outer(Ct_w, Ct_w) / (s * s)

            try:
                step_dir = solve(H, grad)
            except np.linalg.LinAlgError:
                H_r = H + 1e-6 * np.eye(p)
                step_dir = solve(H_r, grad)

            # Backtracking line search on L∞ gap
            cur = maxgap
            alpha = step_cap
            improved = False
            for _ in range(100):
                lam_new = lam - alpha * step_dir
                w_new, s_new = weights_from(lam_new)
                mom_new = (w_new[:, None] * X).sum(axis=0) / s_new
                new_gap = float(np.max(np.abs(mom_new - target)))
                if np.isfinite(new_gap) and new_gap < cur:
                    lam = lam_new
                    improved = True
                    break
                alpha *= 0.5
            if not improved:
                lam = lam - 1e-4 * step_dir  # tiny damped step

        w_final, s_final = weights_from(lam)
        w_final = w_final / s_final
        return w_final, converged, maxgap, it + 1, lam
