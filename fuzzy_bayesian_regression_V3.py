"""
Fuzzy Bayesian Linear Regression (FBLiR) for Time Series Forecasting
Python implementation compatible with FAIM forecasting framework
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm, gamma
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')



# --- Compatibility helpers added to avoid pandas Series.reshape errors ---
import numpy as _np
import pandas as _pd

def _ensure_2d_array(a):
    """Ensure input is a NumPy 2D array suitable for scaler methods.
    Accepts pandas Series/DataFrame or numpy arrays/lists.
    If input is 1D, converts to shape (n,1).
    """
    if isinstance(a, _pd.Series):
        a = a.to_numpy()
    elif isinstance(a, _pd.DataFrame):
        a = a.values
    a = _np.asarray(a)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    return a
# --- end helpers ---

try:
    from bayesian_linear_core import (
        ConjugateBayesianLinearRegression,
        parse_tau_sigma_0_params,
        posterior_linear_samples,
        sanitize_float_matrix,
        sanitize_float_vector,
    )
except ImportError:
    def parse_tau_sigma_0_params(params: dict | None) -> tuple[float, float]:
        params = params or {}
        tau = float(params.get("tau", 1.0))
        if tau <= 0:
            tau = 1.0
        sigma_0_squared = float(params.get("sigma_0_squared", params.get("sigma_0_sq", 1.0)))
        if sigma_0_squared <= 0:
            sigma_0_squared = 1.0
        return tau, sigma_0_squared

    def _as_design_matrix(X):
        if isinstance(X, pd.DataFrame):
            return np.asarray(X, dtype=float)
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X

    def sanitize_float_matrix(X, clip: float = 1e10):
        X = _as_design_matrix(X)
        finite = np.isfinite(X)
        if not finite.all():
            col_fill = np.zeros(X.shape[1], dtype=float)
            for j in range(X.shape[1]):
                col = X[:, j]
                ok = np.isfinite(col)
                col_fill[j] = float(np.median(col[ok])) if ok.any() else 0.0
            X = np.where(finite, X, col_fill)
        if clip is not None and clip > 0:
            X = np.clip(X, -float(clip), float(clip))
        return X

    def sanitize_float_vector(y, clip: float = 1e10):
        y = np.asarray(y, dtype=float).ravel()
        ok = np.isfinite(y)
        if not ok.all():
            fill = float(np.median(y[ok])) if ok.any() else 0.0
            y = np.where(ok, y, fill)
        if clip is not None and clip > 0:
            y = np.clip(y, -float(clip), float(clip))
        return y

    def conjugate_posterior_linear(X, y, tau: float = 1.0, sigma_0_squared: float = 1.0):
        X = sanitize_float_matrix(X)
        y = sanitize_float_vector(y)
        n, p = X.shape
        if n < 2:
            raise ValueError("Need at least 2 samples for Bayesian linear regression.")
        tau = max(float(tau), 1e-8)
        sigma_0_squared = max(float(sigma_0_squared), 1e-12)
        Xd = np.column_stack([np.ones(n, dtype=float), X])
        prior_var = np.concatenate([[sigma_0_squared], np.full(p, tau**2, dtype=float)])
        prior_prec = np.diag(1.0 / prior_var)
        y_hat_init = Xd @ (np.linalg.lstsq(Xd, y, rcond=None)[0])
        residuals = y - y_hat_init
        sigma_squared = float(max(np.var(residuals), 1e-12))
        prec_post = (Xd.T @ Xd) / sigma_squared + prior_prec
        prec_post = prec_post + np.eye(prec_post.shape[0], dtype=float) * 1e-10
        cov_post = np.linalg.inv(prec_post)
        mean_post = cov_post @ (Xd.T @ y / sigma_squared)
        return mean_post, cov_post, sigma_squared

    def posterior_linear_samples(
        X, y, n_samples: int, tau: float = 1.0, sigma_0_squared: float = 1.0, random_state=None
    ):
        mean_post, cov_post, _ = conjugate_posterior_linear(
            X, y, tau=tau, sigma_0_squared=sigma_0_squared
        )
        rng = np.random.default_rng(random_state)
        return rng.multivariate_normal(mean_post, cov_post, size=int(max(1, n_samples)))

    class ConjugateBayesianLinearRegression:
        def __init__(self, tau: float = 1.0, sigma_0_squared: float = 1.0):
            self.tau = float(tau)
            self.sigma_0_squared = float(sigma_0_squared)
            self.coef_ = None
            self.posterior_cov_ = None
            self.sigma_squared_ = None

        @classmethod
        def from_params(cls, params: dict | None):
            tau, sigma_0_squared = parse_tau_sigma_0_params(params)
            return cls(tau=tau, sigma_0_squared=sigma_0_squared)

        def fit(self, X, y):
            self.coef_, self.posterior_cov_, self.sigma_squared_ = conjugate_posterior_linear(
                X, y, tau=self.tau, sigma_0_squared=self.sigma_0_squared
            )
            return self

        def predict(self, X):
            if self.coef_ is None:
                raise ValueError("Model must be fitted before predict.")
            X = sanitize_float_matrix(X)
            n = X.shape[0]
            Xd = np.column_stack([np.ones(n, dtype=float), X])
            return Xd @ self.coef_


class GeneralizedFuzzyNumber:
    """
    Gaussian Fuzzy Number (GFN) implementation
    Represents fuzzy numbers with mean and variance using Gaussian operations
    Based on Abdalla & Buckley 2007 and R code operations
    """
    
    def __init__(self, mean, variance):
        self.mean = float(mean)
        self.variance = float(variance)
    
    def __repr__(self):
        return f"GFN(μ={self.mean:.4f}, σ²={self.variance:.4f})"
    
    def to_array(self):
        """Convert to array format [mean, variance]"""
        return np.array([self.mean, self.variance])
    
    @staticmethod
    def add(A, B):
        """
        Addition of two GFNs
        GFN.add: mean = A[1] + B[1], variance = A[2] + B[2]
        """
        mean = A.mean + B.mean
        variance = A.variance + B.variance
        return GeneralizedFuzzyNumber(mean, variance)
    
    @staticmethod
    def subtract(A, B):
        """
        Subtraction of two GFNs
        GFN.sub: mean = A[1] - B[1], variance = A[2] + B[2]
        """
        mean = A.mean - B.mean
        variance = A.variance + B.variance
        return GeneralizedFuzzyNumber(mean, variance)
    
    @staticmethod
    def multiply(A, B, symmetry_threshold=0.5):
        """
        Multiplication of two GFNs
        GFN.multi: mean = A[1]*B[1], 
                   variance = (B[2]*A[1]^2) + (A[2]*B[1]^2) + (B[2]*A[2])
        """
        mean = A.mean * B.mean
        variance = (B.variance * A.mean**2) + (A.variance * B.mean**2) + (B.variance * A.variance)
        
        # Check delta for symmetry (optional, can be disabled)
        if symmetry_threshold > 0 and mean != 0 and variance > 0:
            delta_A = abs(A.mean / np.sqrt(A.variance)) if A.variance > 0 else np.inf
            delta_B = abs(B.mean / np.sqrt(B.variance)) if B.variance > 0 else np.inf
            
            # If both have small delta, could apply adjustment (from original R code logic)
            # But keeping original GFN.multi formula as primary
        
        return GeneralizedFuzzyNumber(mean, variance)
    
    @staticmethod
    def divide(A, B):
        """
        Division of two GFNs
        GFN.div: mean = A[1]*((1/B[1])+(B[2]/B[1]^3))
                 variance = (A[1]^2*(1/B[1]^4)*B[2])+((1/B[1]^2)*A[2])-(A[2]*(1/B[1]^4)*B[2])
        """
        if abs(B.mean) < 1e-10:
            raise ValueError("Division by zero: B.mean is too close to zero")
        
        mean = A.mean * ((1.0 / B.mean) + (B.variance / (B.mean**3)))
        variance = (A.mean**2 * (1.0 / B.mean**4) * B.variance) + \
                   ((1.0 / B.mean**2) * A.variance) - \
                   (A.variance * (1.0 / B.mean**4) * B.variance)
        
        # Ensure non-negative variance
        variance = max(0.0, variance)
        return GeneralizedFuzzyNumber(mean, variance)
    
    def get_delta(self):
        """Calculate delta = |mean| / sqrt(variance) for symmetry measure"""
        if self.variance <= 0:
            return np.inf if self.mean != 0 else 0.0
        return abs(self.mean / np.sqrt(self.variance))
    
    def defuzzify(self, m=0.1, small_delta_threshold=0.4, max_mean_adjustment=0.75):
        """
        Defuzzification using optimal m parameter and smallDelta threshold
        Based on the R code defuzzification approach
        """
        if self.variance <= 0:
            return self.mean
        
        delta = self.get_delta()
        
        if delta < small_delta_threshold:
            return self.mean + min(float(m) * self.variance, float(max_mean_adjustment))
        return self.mean


class FuzzyBayesianRegression:
    """
    Fuzzy Bayesian Linear Regression using Gaussian Fuzzy Numbers (GFN)
    Based on Abdalla & Buckley 2007 GFN operations
    Uses Bayesian inference with fuzzy arithmetic for uncertainty-aware predictions
    """
    
    def __init__(self, 
                 n_samples=1000,
                 symmetry_threshold=0.5,
                 k=0.5,
                 m=0.1,
                 fuzzify_variance=0.05,
                 uncertainty_weight=0.5,
                 use_quadratic=True,
                 small_delta_threshold=0.4,
                 tau=1.0,
                 sigma_0_squared=1.0):
        """
        Parameters:
        - n_samples: Number of posterior samples for Bayesian inference
        - symmetry_threshold: Threshold for GFN operations (optional, pass 0 to disable)
        - k: (deprecated, kept for compatibility) Defuzzification sensitivity
        - m: Defuzzification magnitude (optimal value typically 0.1-0.3, default 0.1)
        - fuzzify_variance: Variance for fuzzifying features
        - uncertainty_weight: Weight for coefficient uncertainty
        - use_quadratic: Include quadratic terms
        - small_delta_threshold: Threshold for delta = mean/sqrt(variance) (default 0.4)
        - tau: Prior std for slope coefficients beta_j (prior variance tau^2)
        - sigma_0_squared: Prior variance for intercept beta_0
        """
        self.n_samples = n_samples
        self.symmetry_threshold = symmetry_threshold
        self.k = k  # Deprecated but kept for compatibility
        self.m = m  # Defuzzification adjustment magnitude
        self.fuzzify_variance = fuzzify_variance
        self.uncertainty_weight = uncertainty_weight
        self.use_quadratic = use_quadratic
        self.small_delta_threshold = small_delta_threshold
        self.tau = max(float(tau), 1e-8)
        self.sigma_0_squared = max(float(sigma_0_squared), 1e-12)
        
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.coefficients = None
        self.intercept = None
        self.sigma_gfn = None
        self.posterior_mean_coef_ = None
        self.blir_backbone_ = None
        self.is_fitted = False
        
    def _set_identity_y_scaler(self, n_samples):
        """Keep predictions in the same units as IWFR model-scale y (no extra scaling)."""
        self.scaler_y.mean_ = np.array([0.0], dtype=float)
        self.scaler_y.scale_ = np.array([1.0], dtype=float)
        self.scaler_y.var_ = np.array([1.0], dtype=float)
        self.scaler_y.n_features_in_ = 1
        self.scaler_y.n_samples_seen_ = int(n_samples)

    def fit_with_blir_backbone(self, blir_model, X, y, input_prescaled=False):
        """
        IWFR FBLiR: learn from the same ARD fit as BLiR, then wrap coefficients as GFNs.
        Point predictions stay close to BLiR; GFN is a small post-layer adjustment.
        """
        self.blir_backbone_ = blir_model
        self.use_quadratic = False
        X_arr = self._scale_features(X, fit=True, input_prescaled=input_prescaled)
        y_arr = sanitize_float_vector(y)
        self._set_identity_y_scaler(len(y_arr))

        coef = np.asarray(blir_model.coef_, dtype=float).ravel()
        lambdas = np.asarray(
            getattr(blir_model, "lambda_", np.full(coef.shape, 1.0, dtype=float)),
            dtype=float,
        ).ravel()
        if len(lambdas) != len(coef):
            lambdas = np.full(coef.shape, 1.0, dtype=float)
        beta_vars = np.where(lambdas > 0, 1.0 / np.maximum(lambdas, 1e-12), 1e-6)
        mean_var = float(np.mean(beta_vars)) if len(beta_vars) else 1e-6
        beta_vars = beta_vars * self.uncertainty_weight + (1.0 - self.uncertainty_weight) * mean_var

        self.intercept = GeneralizedFuzzyNumber(float(blir_model.intercept_), 1e-12)
        self.coefficients = [
            GeneralizedFuzzyNumber(float(c), max(float(v), 1e-12))
            for c, v in zip(coef, beta_vars)
        ]
        y_hat = np.asarray(blir_model.predict(X_arr), dtype=float).ravel()
        sigma2 = float(max(np.var(y_arr - y_hat), 1e-12))
        self.sigma_gfn = GeneralizedFuzzyNumber(0.0, sigma2)
        self.posterior_mean_coef_ = coef.copy()
        self.is_fitted = True
        return self

    def _bayesian_inference(self, X, y):
        """
        Ridge-style Gaussian posterior on scaled features (zero intercept; y is scaled).
        Legacy fallback when no BLiR backbone is attached.
        """
        n, p = X.shape
        y = np.asarray(y, dtype=float).ravel()
        lambda_prior = 1.0 / max(float(self.tau) ** 2, 1e-8)

        prior_prec = np.eye(p, dtype=float) * lambda_prior
        sigma_squared = float(max(np.var(y), 1e-12))
        prec_post = (X.T @ X) / sigma_squared + prior_prec
        prec_post = prec_post + np.eye(p, dtype=float) * 1e-10
        cov_post = np.linalg.inv(prec_post)
        mean_post = cov_post @ (X.T @ y / sigma_squared)

        rng = np.random.default_rng(42)
        samples = rng.multivariate_normal(mean_post, cov_post, size=int(max(1, self.n_samples)))

        intercept_gfn = GeneralizedFuzzyNumber(0.0, 1e-12)
        self.posterior_mean_coef_ = mean_post.copy()

        beta_gfns = []
        for j in range(p):
            col = j
            beta_mean = float(np.mean(samples[:, col]))
            beta_var = float(np.var(samples[:, col]))
            beta_var = beta_var * self.uncertainty_weight + (
                (1 - self.uncertainty_weight) * float(np.mean(np.var(samples, axis=0)))
            )
            beta_gfns.append(GeneralizedFuzzyNumber(beta_mean, max(beta_var, 1e-12)))

        y_hat = X @ mean_post
        sigma_squared = float(max(np.var(y - y_hat), 1e-12))
        sigma_gfn = GeneralizedFuzzyNumber(0.0, sigma_squared)

        return intercept_gfn, beta_gfns, sigma_gfn
    
    def _add_quadratic_features(self, X):
        """Add quadratic terms to feature matrix"""
        if self.use_quadratic:
            X_quad = X ** 2
            return np.hstack([X, X_quad])
        return X
    
    def _scale_features(self, X, fit=False, input_prescaled=False):
        X_arr = sanitize_float_matrix(X)
        if input_prescaled:
            n_feat = X_arr.shape[1]
            if fit:
                self.scaler_X.mean_ = np.zeros(n_feat, dtype=float)
                self.scaler_X.scale_ = np.ones(n_feat, dtype=float)
                self.scaler_X.var_ = np.ones(n_feat, dtype=float)
                self.scaler_X.n_features_in_ = n_feat
                self.scaler_X.n_samples_seen_ = int(X_arr.shape[0])
            return X_arr
        if fit:
            return self.scaler_X.fit_transform(X_arr)
        return self.scaler_X.transform(X_arr)

    def fit(self, X, y, input_prescaled=False):
        """
        Fit Fuzzy Bayesian regression model
        
        Parameters:
        - X: Feature matrix (n_samples, n_features)
        - y: Target vector (n_samples,)
        - input_prescaled: True when X is already standardized (IWFR app path)
        """
        y_arr = sanitize_float_vector(y)
        X_scaled = self._scale_features(X, fit=True, input_prescaled=input_prescaled)
        y_scaled = self.scaler_y.fit_transform(_ensure_2d_array(y_arr)).ravel()
        
        # Add quadratic features
        X_augmented = self._add_quadratic_features(X_scaled)
        X_augmented = np.clip(X_augmented, -1e8, 1e8)
        
        # Bayesian inference
        self.intercept, self.coefficients, self.sigma_gfn = \
            self._bayesian_inference(X_augmented, y_scaled)
        
        self.is_fitted = True
        return self
    
    def _predict_linear_core(self, X, input_prescaled=False):
        """Stable point predictions from BLiR backbone or posterior mean."""
        X_scaled = self._scale_features(X, fit=False, input_prescaled=input_prescaled)
        if getattr(self, "blir_backbone_", None) is not None:
            return np.asarray(self.blir_backbone_.predict(X_scaled), dtype=float).ravel()
        X_augmented = self._add_quadratic_features(X_scaled)
        X_augmented = np.clip(X_augmented, -1e8, 1e8)
        if self.posterior_mean_coef_ is None:
            raise ValueError("Model missing posterior mean coefficients.")
        y_scaled = np.asarray(X_augmented @ self.posterior_mean_coef_, dtype=float).reshape(-1, 1)
        return self.scaler_y.inverse_transform(y_scaled).ravel()

    def predict_linear_mean(self, X, input_prescaled=False):
        return self._predict_linear_core(X, input_prescaled=input_prescaled)
    
    def _fuzzify_features(self, X, crisp=False):
        """Convert features to GFNs; crisp=True keeps observation means fixed (IWFR BLiR backbone path)."""
        if crisp:
            return [GeneralizedFuzzyNumber(val, 0.0) for val in X]
        return [GeneralizedFuzzyNumber(val, self.fuzzify_variance) for val in X]

    def _fuzzy_predict_on_linear_features(self, X_arr, include_residual_uncertainty=False, crisp_inputs=False):
        """GFN inference on linear features only (same space as BLiR)."""
        predictions = []
        for i in range(X_arr.shape[0]):
            y_gfn = self.intercept
            x_gfns = self._fuzzify_features(X_arr[i, :], crisp=crisp_inputs)
            for beta_gfn, x_gfn in zip(self.coefficients, x_gfns):
                product = GeneralizedFuzzyNumber.multiply(
                    beta_gfn, x_gfn, symmetry_threshold=self.symmetry_threshold
                )
                y_gfn = GeneralizedFuzzyNumber.add(y_gfn, product)
            if include_residual_uncertainty and self.sigma_gfn is not None:
                y_gfn = GeneralizedFuzzyNumber.add(y_gfn, self.sigma_gfn)
            max_adj = 0.0 if crisp_inputs else 0.75
            predictions.append(
                y_gfn.defuzzify(
                    m=self.m,
                    small_delta_threshold=self.small_delta_threshold,
                    max_mean_adjustment=max_adj,
                )
            )
        return np.asarray(predictions, dtype=float)
    
    def predict(self, X, input_prescaled=False, include_residual_uncertainty=False, linear_only=False):
        """
        Make predictions using fuzzy arithmetic
        
        Parameters:
        - X: Feature matrix (n_samples, n_features)
        - input_prescaled: True when X is already standardized (IWFR app path)
        - include_residual_uncertainty: add residual GFN at predict time (off for forecasting)
        - linear_only: skip fuzzy layer (stable recursive multi-step forecasts)
        
        Returns:
        - predictions: numpy array of predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        X_scaled = self._scale_features(X, fit=False, input_prescaled=input_prescaled)

        if getattr(self, "blir_backbone_", None) is not None:
            y_blir = np.asarray(self.blir_backbone_.predict(X_scaled), dtype=float).ravel()
            if linear_only:
                return y_blir
            y_fuzzy = self._fuzzy_predict_on_linear_features(
                X_scaled,
                include_residual_uncertainty=include_residual_uncertainty,
                crisp_inputs=True,
            )
            gfn_weight = min(0.12, max(0.02, float(self.m) * 0.5))
            return (1.0 - gfn_weight) * y_blir + gfn_weight * y_fuzzy
        
        if linear_only:
            return self._predict_linear_core(X, input_prescaled=input_prescaled)
        
        # Legacy standalone FBLiR path (quadratic + internal y scaling)
        X_augmented = self._add_quadratic_features(X_scaled)
        X_augmented = np.clip(X_augmented, -1e8, 1e8)
        
        predictions = []
        
        for i in range(X_augmented.shape[0]):
            # Start with intercept
            y_gfn = self.intercept
            
            # Fuzzify current observation features
            x_gfns = self._fuzzify_features(X_augmented[i, :])
            
            # Fuzzy multiplication and addition
            for j, (beta_gfn, x_gfn) in enumerate(zip(self.coefficients, x_gfns)):
                product = GeneralizedFuzzyNumber.multiply(
                    beta_gfn, x_gfn, 
                    symmetry_threshold=self.symmetry_threshold
                )
                y_gfn = GeneralizedFuzzyNumber.add(y_gfn, product)
            
            if include_residual_uncertainty and self.sigma_gfn is not None:
                y_gfn = GeneralizedFuzzyNumber.add(y_gfn, self.sigma_gfn)
            
            y_pred_scaled = y_gfn.defuzzify(
                m=self.m, 
                small_delta_threshold=self.small_delta_threshold
            )

            if self.posterior_mean_coef_ is not None:
                linear_pred = float(X_augmented[i, :] @ self.posterior_mean_coef_)
                y_pred_scaled = 0.55 * y_pred_scaled + 0.45 * linear_pred
            
            predictions.append(y_pred_scaled)
        
        # Convert to array and rescale
        predictions = np.array(predictions)
        predictions = self.scaler_y.inverse_transform(_ensure_2d_array(predictions)).ravel()
        
        return predictions


class FuzzyBayesianRegressionTuned(FuzzyBayesianRegression):
    """
    Fuzzy Bayesian Regression with automatic hyperparameter tuning
    """
    
    def __init__(self, n_samples=1000, use_quadratic=True, verbose=False):
        super().__init__(n_samples=n_samples, use_quadratic=use_quadratic)
        self.verbose = verbose
        self.best_params = None
        
    def fit(self, X, y, X_val=None, y_val=None, input_prescaled=False):
        """
        Fit with hyperparameter tuning
        
        Parameters:
        - X: Training features
        - y: Training targets
        - X_val: Validation features (optional, uses train if None)
        - y_val: Validation targets (optional, uses train if None)
        """
        if X_val is None or y_val is None:
            # Use training set for validation (simple approach)
            X_val, y_val = X, y
        
        # Hyperparameter grid (simplified from R version)
        param_grid = {
            'symmetry_threshold': [0.1, 0.5, 1.0],
            'k': [-0.5, 0.0, 0.5],
            'm': [-0.5, 0.0, 0.5],
            'fuzzify_variance': [0.01, 0.05, 0.1],
            'uncertainty_weight': [0.25, 0.5, 0.75]
        }
        
        # Grid search
        best_mae = float('inf')
        best_params = None
        
        if self.verbose:
            total_combinations = np.prod([len(v) for v in param_grid.values()])
            print(f"Testing {total_combinations} hyperparameter combinations...")
        
        for sym_thresh in param_grid['symmetry_threshold']:
            for k_val in param_grid['k']:
                for m_val in param_grid['m']:
                    for fuzz_var in param_grid['fuzzify_variance']:
                        for unc_weight in param_grid['uncertainty_weight']:
                            try:
                                # Create model with current params
                                model = FuzzyBayesianRegression(
                                    n_samples=self.n_samples,
                                    symmetry_threshold=sym_thresh,
                                    k=k_val,
                                    m=m_val,
                                    fuzzify_variance=fuzz_var,
                                    uncertainty_weight=unc_weight,
                                    use_quadratic=self.use_quadratic
                                )
                                
                                # Fit and evaluate
                                model.fit(X, y, input_prescaled=input_prescaled)
                                y_pred = model.predict(X_val, input_prescaled=input_prescaled)
                                mae = np.mean(np.abs(y_val - y_pred))
                                
                                if mae < best_mae:
                                    best_mae = mae
                                    best_params = {
                                        'symmetry_threshold': sym_thresh,
                                        'k': k_val,
                                        'm': m_val,
                                        'fuzzify_variance': fuzz_var,
                                        'uncertainty_weight': unc_weight
                                    }
                            except:
                                continue
        
        if best_params is None:
            # Fallback to defaults
            if self.verbose:
                print("Grid search failed, using default parameters")
            best_params = {
                'symmetry_threshold': 0.5,
                'k': 0.5,
                'm': 0.5,
                'fuzzify_variance': 0.05,
                'uncertainty_weight': 0.5
            }
        
        self.best_params = best_params
        
        if self.verbose:
            print(f"\nBest parameters found:")
            for param, value in best_params.items():
                print(f"  {param}: {value}")
            print(f"  Validation MAE: {best_mae:.4f}")
        
        # Fit final model with best parameters
        self.symmetry_threshold = best_params['symmetry_threshold']
        self.k = best_params['k']
        self.m = best_params['m']
        self.fuzzify_variance = best_params['fuzzify_variance']
        self.uncertainty_weight = best_params['uncertainty_weight']
        
        super().fit(X, y, input_prescaled=input_prescaled)
        return self


# Example usage and testing
if __name__ == "__main__":
    # Generate synthetic data for testing
    np.random.seed(42)
    n_samples = 200
    n_features = 5
    
    X = np.random.randn(n_samples, n_features)
    true_coef = np.array([2.0, -1.5, 0.8, -0.3, 1.2])
    noise = np.random.randn(n_samples) * 0.5
    y = X @ true_coef + noise
    
    # Split data
    train_size = int(0.8 * n_samples)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print("=" * 60)
    print("Fuzzy Bayesian Linear Regression Test")
    print("=" * 60)
    
    # Test basic FBLiR
    print("\n1. Basic FBLiR Model:")
    fblr = FuzzyBayesianRegression(n_samples=500, use_quadratic=False)
    fblr.fit(X_train, y_train)
    y_pred = fblr.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    mae = np.mean(np.abs(y_test - y_pred))
    print(f"   MSE: {mse:.4f}")
    print(f"   MAE: {mae:.4f}")
    
    # Test with quadratic features
    print("\n2. FBLiR with Quadratic Features:")
    fblr_quad = FuzzyBayesianRegression(n_samples=500, use_quadratic=True)
    fblr_quad.fit(X_train, y_train)
    y_pred_quad = fblr_quad.predict(X_test)
    
    mse_quad = mean_squared_error(y_test, y_pred_quad)
    mae_quad = np.mean(np.abs(y_test - y_pred_quad))
    print(f"   MSE: {mse_quad:.4f}")
    print(f"   MAE: {mae_quad:.4f}")
    
    # Test with hyperparameter tuning
    print("\n3. FBLiR with Hyperparameter Tuning:")
    fblr_tuned = FuzzyBayesianRegressionTuned(n_samples=500, use_quadratic=True, verbose=True)
    fblr_tuned.fit(X_train, y_train, X_test, y_test)
    y_pred_tuned = fblr_tuned.predict(X_test)
    
    mse_tuned = mean_squared_error(y_test, y_pred_tuned)
    mae_tuned = np.mean(np.abs(y_test - y_pred_tuned))
    print(f"   MSE: {mse_tuned:.4f}")
    print(f"   MAE: {mae_tuned:.4f}")
    
    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)

