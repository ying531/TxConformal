from typing import Optional, Dict, Any, Tuple, Callable
from dataclasses import dataclass
import numpy as np

from .conformal.pvalues import (
    weighted_pvalues_individual,
    weighted_pvalues_for_selection,
)
from .conformal.scores import make_score
from .features.providers import FeaturesProvider
from .selection.sel import (
    bh_threshold,
    fp_budget_threshold,
    tp_min_threshold,
    estimate_topk_fp,
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
    meta: Dict[str, Any]
    fdp_est: float


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
        self._meta: Dict[str, Any] = {}
        self._fitted = False

    # ---------- helpers ----------
    @staticmethod
    def _broadcast(x, n: int) -> np.ndarray:
        x = np.asarray(x, float).ravel()
        if x.size == 1:
            return np.full(n, float(x))
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
        c_cal = self._broadcast(self.cutoff if cutoff is None else cutoff, n_cal)
        c_tst = self._broadcast(self.cutoff if cutoff is None else cutoff, n_tst)

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
        # has_embeddings = (getattr(prov, "E_calib", None) is not None) or (getattr(prov, "embed_fn", None) is not None)
        # # special case: no embeddings AND user explicitly requests (f, bins)
        # if (not has_embeddings) and (self.force_mode in ("pred_bin", "f+bins")):
            # Fc, Ft = prov.get_block(mode="f+bins")
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
        print_level: int = -1,
    ):
        """
        Compute weights and p-values. Cache for later selection.

        Returns self for chaining.
        """ 
        # obtain weights
        if weight:
            w_cal, eb_meta = self._weights_from_provider(prov, print_level=print_level)
            w_cal = w_cal / np.mean(w_cal) 
        else:
            w_cal = np.ones(len(prov.f_calib))
            eb_meta = {}

        # Scores
        V_cal, V_cal_c, V_test, score_meta = self._compute_scores(prov.f_calib, y_calib, prov.f_test, cutoff=cutoff)

        # P-values
        rnd = self.randomize_p if randomize_p is None else bool(randomize_p)
        seed = self.random_state if random_state is None else int(random_state)
        p_indiv = weighted_pvalues_individual(V_cal_c, w_cal, V_test, w_test=None, randomize=rnd, random_state=seed)
        p_sel = weighted_pvalues_for_selection(V_cal, w_cal, V_test, w_test=None, randomize=rnd, random_state=seed)

        # Cache everything
        self._p_indiv = p_indiv
        self._p_sel = p_sel
        self._weights_calib = w_cal
        self._meta = {
            "eb_meta": eb_meta,
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
            One of 'bh', 'fp_budget', 'tp_min'
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

        A = float(alpha)
        meth = method.lower()

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
            order = np.argsort(p_sel, kind="mergesort")
            k = min(int(K), p_sel.size)
            idx = order[:k]
            thr = float(p_sel[idx[-1]]) if k > 0 else 0.0
            est = estimate_topk_fp(p_sel, int(K)) 
        else:
            raise ValueError(f"Unknown method: {method}")
 
        meta = {**self._meta, "alpha": A, "method": meth}
        return SelectionResult(
            method=meth,
            idx=idx,
            threshold=float(thr),
            fdp_est = est,
            p_values=p_indiv,
            p_sel=p_sel,
            weights_calib=w_cal,
            meta=meta
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
        print_level: int = -1,
    ) -> SelectionResult:
        """Convenience: fit() then select() in one call."""
        self.fit(
            prov, y_calib,
            cutoff=cutoff, 
            randomize_p=randomize_p, random_state=random_state, print_level=print_level
        )
        return self.select(method=method, alpha=alpha, K=K)

    # ---------- small convenience ----------
    @staticmethod
    def estimate_fdp_from_p(p_sel: np.ndarray, K: int) -> float:
        """Estimate FDP for top-K from p-values."""
        return float(estimate_topk_fp(np.asarray(p_sel, float).ravel(), int(K)))
