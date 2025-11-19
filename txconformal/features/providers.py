from typing import Optional, Callable, Tuple, Any
import numpy as np  
from .quantiles import make_bins

class FeaturesProvider:
    """
    Manages features for conformal inference with covariate shift correction.

    Accepts data in three flexible formats:
      1) Precomputed predictions + embeddings: f_calib, f_test, E_calib, E_test
      2) Raw covariates + functions: X_calib, X_test, predict_fn, embed_fn
      3) Mixed: any combination of the above

    Parameters
    ----------
    f_calib, f_test : array-like, optional
        Predicted values (e.g., risk scores, probabilities) for calibration and test sets.
        Shape: (n_calib,) and (n_test,)

    E_calib, E_test : array-like, optional
        Embeddings (e.g., fingerprints, learned descriptors) for shift correction.
        Shape: (n_calib, d_embed) and (n_test, d_embed)

    X_calib, X_test : array-like, optional
        Raw covariates/features (e.g., SMILES strings). Required only if f_*/E_* are not provided
        and you supply predict_fn / embed_fn.

    predict_fn : callable, optional
        Function to compute predictions from X: f = predict_fn(X).
        Used if f_calib/f_test are not provided.

    embed_fn : callable, optional
        Function to compute embeddings from X: E = embed_fn(X).
        Used if E_calib/E_test are not provided.

    Notes
    -----
    After calling `.prepare(quantiles=...)`, these attributes are populated:

      - f_calib, f_test : (n_cal,), predictions
      - E_calib, E_test : (n_cal,d), (n_test,d) or None
      - bins_calib, bins_test : one-hot bins for f
      - edges : bin edges (pooled across calib+test)
      - phi_calib, phi_test : X_soft = [f, bins, (optional) E]

    Use `.force_block(mode=...)` to retrieve X_force=(Fc,Ft).
    """

    # -------------------------- init --------------------------
    def __init__(
        self,
        *,
        f_calib: Optional[np.ndarray] = None,
        f_test: Optional[np.ndarray] = None,
        E_calib: Optional[np.ndarray] = None,
        E_test: Optional[np.ndarray] = None,
        X_calib: Optional[Any] = None,
        X_test: Optional[Any] = None,
        predict_fn: Optional[Callable[[Any], np.ndarray]] = None,
        embed_fn: Optional[Callable[[Any], np.ndarray]] = None,
    ):
        self.f_calib = None if f_calib is None else np.asarray(f_calib, float).ravel()
        self.f_test  = None if f_test  is None else np.asarray(f_test,  float).ravel()

        self.E_calib = None if E_calib is None else np.asarray(E_calib, float)
        self.E_test  = None if E_test  is None else np.asarray(E_test,  float)

        self.X_calib = X_calib
        self.X_test = X_test
        self.predict_fn = predict_fn
        self.embed_fn = embed_fn

        # populated by prepare()
        self.bins_calib: Optional[np.ndarray] = None
        self.bins_test: Optional[np.ndarray] = None
        self.edges: Optional[np.ndarray] = None
        self.phi_calib: Optional[np.ndarray] = None
        self.phi_test: Optional[np.ndarray] = None

        # small cache for force variants built during prepare()
        self._Fc_f: Optional[np.ndarray] = None
        self._Ft_f: Optional[np.ndarray] = None
        self._Fc_fb: Optional[np.ndarray] = None
        self._Ft_fb: Optional[np.ndarray] = None

    # -------------------------- prepare --------------------------
    def prepare(self, *, quantiles: int = 10, filter_unbalanceable: bool = False,
                lower_quantile: float = 0.01, upper_quantile: float = 0.99,
                soft_mode: str = "none", force_mode: str = "none") -> None:
        """
        Compute predictions/embeddings if needed, build bins, and assemble:
        - X_soft = phi = [f, bins, (optional) E]
        - cached force blocks for "f" and "f+bins"

        Parameters
        ----------
        quantiles : int
            Number of bins for predictions
        filter_unbalanceable : bool
            If True, remove embedding dimensions where test mean falls outside
            calibration's [lower_quantile, upper_quantile] range
        lower_quantile, upper_quantile : float
            Quantiles for filtering (default: 1% and 99%)
        """
        # 1) predictions
        if self.f_calib is None or self.f_test is None:
            if self.predict_fn is not None:
                self.f_calib = np.asarray(self.predict_fn(self.X_calib), float).ravel()
                self.f_test  = np.asarray(self.predict_fn(self.X_test),  float).ravel()
            else:
                return 

        # 2) embeddings (optional)
        if self.E_calib is None and self.embed_fn is not None:
            self.E_calib = np.asarray(self.embed_fn(self.X_calib), float)
            self.E_test  = np.asarray(self.embed_fn(self.X_test),  float)

        # 2b) filter unbalanceable embedding dimensions
        if filter_unbalanceable and self.E_calib is not None:
            E_c = self.E_calib
            E_t = self.E_test
            test_mean = E_t.mean(axis=0)
            lower_bound = np.quantile(E_c, lower_quantile, axis=0)
            upper_bound = np.quantile(E_c, upper_quantile, axis=0)
            keep_dims = (test_mean >= lower_bound) & (test_mean <= upper_bound)
            if keep_dims.sum() > 0:
                self.E_calib = E_c[:, keep_dims]
                self.E_test = E_t[:, keep_dims]
            else: 
                self.E_calib = None
                self.E_test = None

        # 3) pooled equal-frequency bins on f
        Zc, Zt, edges = make_bins(self.f_calib, self.f_test, n_bins=quantiles)
        self.bins_calib, self.bins_test, self.edges = Zc, Zt, edges

        # 4) X_soft = [f, bins, (optional) E]
        if soft_mode == 'none':
            blocks_c, blocks_t = self.get_block(mode='f+bins')
            blocks_c =[blocks_c]
            blocks_t =[blocks_t]
            # blocks_t = [self.f_test[:,  None], self.f_test[:,  None]**2, Zt]
            if self.E_calib is not None:
                blocks_c.append(self.E_calib)
                blocks_t.append(self.E_test)
            self.phi_calib = np.concatenate(blocks_c, axis=1)
            self.phi_test  = np.concatenate(blocks_t, axis=1)
        else:
            phi_calib, phi_test = self.get_block(mode=soft_mode)
            if phi_calib is None or phi_test is None:
                raise ValueError(f"soft_mode={soft_mode!r} returned (None, None); cannot build X_soft")
            self.phi_calib = phi_calib
            self.phi_test = phi_test

        # 5) cache common force blocks
        self._F_calib_pred = self.f_calib[:, None]
        self._F_test_pred = self.f_test[:, None] 
        
        if force_mode == 'none': 
            Fc, Ft = self.get_block(mode='f+bins')
            self.force_calib = Fc
            self.force_test = Ft
        else:
            Fc, Ft = self.get_block(mode=force_mode)
            self.force_calib = Fc
            self.force_test = Ft

        # backup default blocks 
        Fc, Ft = self.get_block(mode='f+bins')
        self.backup_calib = [Fc, self._F_calib_pred]
        self.backup_test = [Ft, self._F_test_pred] 

    
    def has_embeddings(self) -> bool:
        """True if embeddings are available or can be computed via embed_fn."""
        return (self.E_calib is not None) or (self.embed_fn is not None)

    def get_soft_block(self):
        """Return X_soft = (phi_calib, phi_test)."""
        if self.phi_calib is None or self.phi_test is None:
            raise ValueError("call prepare(...) before soft_block(...)")
        return self.phi_calib, self.phi_test
    
    def get_force_block(self):
        """Return cached X_force = (Fc, Ft) if set via prepare(); else
        (None, None)."""
        return self.force_calib, self.force_test
    
    def get_backup_block(self):
        """Return cached X_backup = (Fbc, Fbt) if set via prepare(); else
        (None, None)."""
        return self.backup_calib, self.backup_test

    def get_block(
        self,
        mode: str = "auto",
        custom_fn: Optional[Callable[["FeaturesProvider"], Tuple[Optional[np.ndarray], Optional[np.ndarray]]]] = None,
    ):
        """
        Build X=(Fc,Ft).

        Modes
        -----
        - "auto"    : if embeddings exist -> "f"; else -> "f+bins"
        - "f"       : predictions only
        - "f+bins" / "pred_bin" : concatenate [f, one-hot bins]
        - "none"    : (None, None)
        - "custom"  : custom_fn(self) -> (Fc,Ft), both arrays or both None
        """ 

        if mode == "none":
            return None, None

        if mode == "auto":
            mode = "f" if self.has_embeddings() else "f+bins"

        if mode in ("f", "pred", "prediction"):
            return self.f_calib[:, None], self.f_test[:, None]

        if mode in ("f+bins", "pred_bin"):
            # Zc, Zt, edges = make_bins(self.f_calib, self.f_test, n_bins=quantiles)
            # self.bins_calib, self.bins_test, self.edges = Zc, Zt, edges
            F_calib_bins = np.concatenate([self.f_calib[:, None], self.f_calib[:, None]**2, self.bins_calib], axis=1)  # f+bins
            F_test_bins = np.concatenate([self.f_test[:, None], self.f_test[:, None]**2, self.bins_test], axis=1)
            return F_calib_bins, F_test_bins

        if mode == "custom":
            if custom_fn is None or not callable(custom_fn):
                raise ValueError("force_block(mode='custom') requires custom_fn(provider)->(Fc,Ft)")
            Fc, Ft = custom_fn(self)
            Fc = None if Fc is None else np.asarray(Fc, float)
            Ft = None if Ft is None else np.asarray(Ft, float)
            if (Fc is None) != (Ft is None):
                raise ValueError("custom_fn must return both Fc and Ft or both None")
            if Fc is not None and Fc.shape[0] != self.f_calib.shape[0]:
                raise ValueError("Fc has wrong number of rows")
            if Ft is not None and Ft.shape[0] != self.f_test.shape[0]:
                raise ValueError("Ft has wrong number of rows")
            return Fc, Ft

        raise ValueError(f"Unknown force mode: {mode!r}")
    
    def set_soft_block(
        self,
        phi_calib: np.ndarray,
        phi_test: np.ndarray,
    ) -> None:
        """
        Manually set X_soft = (phi_calib, phi_test).

        Parameters
        ----------
        phi_calib : array-like, shape (n_calib, d_phi)
            Calibration design matrix
        phi_test : array-like, shape (n_test, d_phi)
            Test design matrix
        """
        self.phi_calib = np.asarray(phi_calib, float)
        self.phi_test = np.asarray(phi_test, float)
    

    def set_force_block(
        self,
        F_calib: Optional[np.ndarray],
        F_test: Optional[np.ndarray],
    ) -> None:
        """
        Manually set cached X_force = (F_calib, F_test).

        Parameters
        ----------
        Fc : array-like, shape (n_calib, d_Fc) or None
            Calibration force design matrix
        Ft : array-like, shape (n_test, d_Ft) or None
            Test force design matrix
        """
        self.force_calib = None if F_calib is None else np.asarray(F_calib, float)
        self.force_test = None if F_test is None else np.asarray(F_test, float)

    def set_backup_block(
        self,
        Fbc: Optional[list[np.ndarray]],
        Fbt: Optional[list[np.ndarray]],
    ) -> None:
        """
        Manually set cached X_backup = (Fbc, Fbt).

        Parameters
        ----------
        Fbc : list of array-like, shape (n_calib, d_Fbc) or None
            Calibration backup design matrix
        Fbt : list of array-like, shape (n_test, d_Fbt) or None
            Test backup design matrix
        """
        self.backup_calib = None if Fbc is None else [np.asarray(f, float) for f in Fbc]
        self.backup_test = None if Fbt is None else [np.asarray(f, float) for f in Fbt]