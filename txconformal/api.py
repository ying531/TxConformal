from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import LinearRegression

from .conformal.pvalues import (
    weighted_pvalues_individual,
    weighted_pvalues_for_selection,
)
from .conformal.scores import make_score
from .features.providers import FeaturesProvider
from .selection.sel import (
    TopKEstimate,
    bh_threshold,
    fp_budget_threshold,
    tp_min_threshold,
    estimate_topk,
)
from .shift.retries import retry_entropy_balancing


# ---------------- results DTO ----------------
@dataclass
class SelectionResult:
    method: str
    idx: np.ndarray
    threshold: float
    p_values: np.ndarray
    p_sel: np.ndarray
    weights_calib: np.ndarray
    weights_test: np.ndarray
    meta: Dict[str, Any]
    fdp_est: float
    fp_count_est: Optional[float] = None
    fdp_est_CI: Optional[Tuple[float, float]] = None


# ---------------- main API ----------------
class TxConformal:
    def __init__(
        self,
        *,
        # cutoff: float = 0.5,
        # alpha: float = 0.10,
        score_name: str = "clip",
        M: float = 100.0,
        # EB settings
        grid = ((0.1, 1e-5), (0.5, 1e-5), (1.0, 1e-5),
                (0.1, 1e-4), (0.5, 1e-4), (1.0, 1e-4),
                (0.1, 1e-3), (0.5, 1e-3), (1.0, 1e-3),
                (0.1, 1e-2), (0.5, 1e-2), (1.0, 1e-2)),
        tolerances = (1e-5, 1e-4, 1e-3, 1e-2),
        max_iterations: int = 500,
        step_cap: float = 1.0,
        # p-value randomization
        randomize_p: bool = False,
        random_state: int = 0, 
    ):
        # self.cutoff = float(cutoff)
        # self.alpha = float(alpha)
        self.score_name = str(score_name)
        self.M = float(M)

        self.grid = tuple(grid)
        self.tolerances = tuple(tolerances)
        self.max_iterations = int(max_iterations)
        self.step_cap = float(step_cap)

        self.randomize_p = bool(randomize_p)
        self.random_state = int(random_state)

        # self.force_mode = force_mode
        # self.force_custom = force_custom

        # Cached state after fit()
        self._p_indiv: Optional[np.ndarray] = None
        self._p_sel: Optional[np.ndarray] = None
        self._weights_calib: Optional[np.ndarray] = None
        self._weights_test: Optional[np.ndarray] = None
        self._meta: Dict[str, Any] = {}
        self._fitted = False
        self._f_calib_raw = None
        self._f_test_raw = None

    # ---------- helpers ----------
    @staticmethod
    def _broadcast(x, n: int) -> np.ndarray:
        x = np.asarray(x, float).ravel()
        if x.size == 1:
            return np.full(n, float(x.item()))
        if x.size != n:
            raise ValueError(f"value must be scalar or length {n}")
        return x

    def _compute_scores(
        self,
        f_calib: np.ndarray,
        y_calib: np.ndarray,
        f_test: np.ndarray,
        *,
        cutoff: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        f_cal = np.asarray(f_calib, float).ravel()
        y_cal = np.asarray(y_calib, float).ravel()
        f_tst = np.asarray(f_test,  float).ravel()
        if f_cal.size != y_cal.size:
            raise ValueError("f_calib and y_calib must have the same length")

        n_cal, n_tst = f_cal.size, f_tst.size
        if cutoff is None:
            raise ValueError("cutoff is required")
        c_cal = self._broadcast(cutoff, n_cal)
        c_tst = self._broadcast(cutoff, n_tst)

        score = make_score(self.score_name, M=self.M)  # uses your scores.py
        V_cal = score(y_cal, f_cal, c_cal)                 # clip score on calib
        V_cal_c = score(c_cal, f_cal, c_cal)           # non-clipped calib score for interpretable p-values
        V_test = score(c_tst, f_tst, c_tst)               # test score for p-values 

        meta = {"score": self.score_name, "M": self.M}
        return V_cal, V_cal_c, V_test, meta

    def _weights_from_provider(
        self, 
        prov: FeaturesProvider, 
        print_level: int = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        
        n_cal = prov.f_calib.shape[0] 
        
        Xc, Xt = prov.get_soft_block() 
        Fc, Ft = prov.get_force_block()
        bc, bt = prov.get_backup_block() 
        w_cal, meta = retry_entropy_balancing(
            phi_calib=Xc, phi_test=Xt,
            force_features_calib=Fc, force_features_test=Ft,
            backup_features_calib=bc,  
            backup_features_test=bt,  
            grid=self.grid, tolerances=self.tolerances,
            max_iterations=self.max_iterations, step_cap=self.step_cap,
            print_level=print_level, random_state=self.random_state,
            add_no_pca_fallback=True,
        )
        return w_cal, {"mode": "entropy_balancing", "using": "pred_bin_no_embeddings", **meta}

    @staticmethod
    def _candidate_weights(
        eb_meta: Dict[str, Any],
        w_cal: np.ndarray,
        n_test: int,
        *,
        mode: str,
        w_test: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Construct candidate-specific test weights on the calibration scale.

        ``mode='uniform'`` preserves the original package behavior.  The
        ``'log_linear'`` mode reproduces the deployment notebook: it regresses
        log calibration weights on the final entropy-balancing design and
        evaluates that regression on the candidate design.  Explicit
        ``w_test`` values take precedence over ``mode``.
        """

        if w_test is not None:
            values = np.asarray(w_test, dtype=float).ravel()
            source = "provided"
        else:
            candidate_mode = str(mode).lower()
            if candidate_mode == "uniform":
                values = np.ones(n_test, dtype=float)
                source = "uniform"
            elif candidate_mode == "log_linear":
                try:
                    calib_x = np.asarray(eb_meta["calib_x"], dtype=float)
                    test_x = np.asarray(eb_meta["test_x"], dtype=float)
                except KeyError as exc:
                    raise ValueError(
                        "candidate_weight_mode='log_linear' requires the final "
                        "calibration and test designs from entropy balancing"
                    ) from exc
                if calib_x.shape[0] != w_cal.size or test_x.shape[0] != n_test:
                    raise ValueError("entropy-balancing designs have incompatible row counts")
                model = LinearRegression().fit(calib_x, np.log(w_cal))
                values = np.exp(model.predict(test_x))
                source = "log_linear"
            else:
                raise ValueError(
                    "candidate_weight_mode must be 'uniform' or 'log_linear'"
                )

        if values.size != n_test:
            raise ValueError(f"w_test must have length {n_test}")
        if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("candidate weights must be positive and finite")

        meta = {
            "mode": source,
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
        }
        return values, meta

    # ---------- public: fit weights and p-values ----------
    def fit(
        self,
        prov: FeaturesProvider,
        y_calib: np.ndarray,
        *,
        cutoff: np.ndarray,
        randomize_p: Optional[bool] = None,
        random_state: Optional[int] = None, 
        weight: Optional[bool] = True,
        candidate_weight_mode: str = "uniform",
        w_test: Optional[np.ndarray] = None,
        print_level: int = -1,
    ):
        """
        Compute weights and p-values. Cache for later selection.

        Returns self for chaining.
        """ 

        # store scores 
        self._f_calib_raw = np.asarray(prov.f_calib, float).copy()
        self._f_test_raw = np.asarray(prov.f_test, float).copy()
        # obtain weights
        if weight:
            w_cal, eb_meta = self._weights_from_provider(prov, print_level=print_level)
            w_cal = w_cal / np.mean(w_cal) 
        else:
            w_cal = np.ones(len(prov.f_calib))
            eb_meta = {}

        w_test_values, candidate_weight_meta = self._candidate_weights(
            eb_meta,
            w_cal,
            len(prov.f_test),
            mode=candidate_weight_mode,
            w_test=w_test,
        )

        # Scores
        V_cal, V_cal_c, V_test, score_meta = self._compute_scores(prov.f_calib, y_calib, prov.f_test, cutoff=cutoff)

        # P-values
        rnd = self.randomize_p if randomize_p is None else bool(randomize_p)
        seed = self.random_state if random_state is None else int(random_state)
        p_indiv = weighted_pvalues_individual(
            V_cal_c, w_cal, V_test, w_test=w_test_values,
            randomize=rnd, random_state=seed,
        )
        p_sel = weighted_pvalues_for_selection(
            V_cal, w_cal, V_test, w_test=w_test_values,
            randomize=rnd, random_state=seed,
        )

        # Cache everything
        self._p_indiv = p_indiv
        self._p_sel = p_sel
        self._weights_calib = w_cal
        self._weights_test = w_test_values
        self._meta = {
            "eb_meta": eb_meta,
            "candidate_weight_meta": candidate_weight_meta,
            "score_meta": score_meta,
            "provider_meta": {
                # "quantiles": quantiles,
                "bins_dim": int(prov.bins_calib.shape[1]) if getattr(prov, "bins_calib", None) is not None else 0,
                "E_dim": None if getattr(prov, "E_calib", None) is None else int(prov.E_calib.shape[1]),
                # "force_mode": self.force_mode,
            },
        }
        self._fitted = True
        return self

    # ---------- public: selection only (uses cached p-values) ----------
    def select(
        self,
        *,
        method: str = "bh",
        alpha: float = 0.1,
        K: Optional[int] = None,
    ) -> SelectionResult:
        """
        Perform selection using cached p-values from fit().

        Parameters
        ----------
        method : str
            One of 'bh', 'fp_budget', 'tp_min', or 'top_k'
        alpha : float, optional
            FDR level (for method='bh')
        K : int, optional
            Number threshold (for method='fp_budget' or 'tp_min')

        Returns
        -------
        SelectionResult
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before select()")

        p_sel = self._p_sel
        p_indiv = self._p_indiv
        w_cal = self._weights_calib
        w_test = self._weights_test

        meth = method.lower()
        A = 0.1 if alpha is None else float(alpha)
        fp_count_est = None

        if meth == "bh":
            thr, idx = bh_threshold(p_sel, alpha=A)
            est = A
        elif meth == "fp_budget":
            if K is None: raise ValueError("K is required for fp_budget")
            thr, idx = fp_budget_threshold(p_sel, K=K)
            est = K / len(idx) if len(idx) > 0 else 0.0
        elif meth == "tp_min":
            if K is None: raise ValueError("K is required for tp_min")
            thr, idx = tp_min_threshold(p_sel, K=K)
            est = (len(idx) - K) / len(idx) if len(idx) > 0 else 0.0
        elif meth == 'top_k':
            if K is None:
                raise ValueError("K is required for top_k")
            topk = estimate_topk(p_sel, K)
            order = np.argsort(p_sel, kind="mergesort")
            idx = order[:topk.k]
            thr = topk.p_boundary
            est = topk.fdp
            fp_count_est = topk.fp_count
        else:
            raise ValueError(f"Unknown method: {method}")
 
        meta = {**self._meta, "alpha": A, "method": meth}
        return SelectionResult(
            method=meth,
            idx=idx,
            threshold=float(thr),
            fdp_est=est,
            fp_count_est=fp_count_est,
            fdp_est_CI=None,
            p_values=p_indiv,
            p_sel=p_sel,
            weights_calib=w_cal,
            weights_test=w_test,
            meta=meta,
        ) 

    # ---------- convenience: fit + select in one call ----------
    def fit_select(
        self,
        prov: FeaturesProvider,
        y_calib: np.ndarray,
        *,
        method: str = "bh",
        alpha: Optional[float] = None,
        K: Optional[int] = None,
        cutoff: Optional[np.ndarray] = None, 
        randomize_p: Optional[bool] = None,
        random_state: Optional[int] = None,
        weight: Optional[bool] = True,
        candidate_weight_mode: str = "uniform",
        w_test: Optional[np.ndarray] = None,
        print_level: int = -1,
    ) -> SelectionResult:
        """Convenience: fit() then select() in one call."""
        self.fit(
            prov, y_calib,
            cutoff=cutoff, 
            randomize_p=randomize_p, random_state=random_state,
            weight=weight, candidate_weight_mode=candidate_weight_mode,
            w_test=w_test, print_level=print_level
        )
        return self.select(method=method, alpha=alpha, K=K)

    # ---------- small convenience ----------
    @staticmethod
    def estimate_fdp_from_p(p_sel: np.ndarray, K: int) -> float:
        """Estimate FDP for top-K from p-values."""
        return estimate_topk(np.asarray(p_sel, float).ravel(), K).fdp

    @staticmethod
    def estimate_topk_from_p(p_sel: np.ndarray, K: int) -> TopKEstimate:
        """Return explicit FP-count and FDP point estimates for top-K."""
        return estimate_topk(np.asarray(p_sel, float).ravel(), K)
