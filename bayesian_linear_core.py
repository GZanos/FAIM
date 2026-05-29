"""
Conjugate Gaussian Bayesian linear regression with paper-aligned priors:
  beta_0 ~ N(0, sigma_0^2),  beta_j ~ N(0, tau^2) for j >= 1.
Residual variance sigma^2 is estimated from training residuals (empirical Bayes).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def parse_tau_sigma_0_params(params: dict | None) -> tuple[float, float]:
    """Read tau (prior std for slopes) and sigma_0_squared from model_params dict."""
    params = params or {}
    tau = float(params.get("tau", 1.0))
    if tau <= 0:
        tau = 1.0
    sigma_0_squared = float(params.get("sigma_0_squared", params.get("sigma_0_sq", 1.0)))
    if sigma_0_squared <= 0:
        sigma_0_squared = 1.0
    return tau, sigma_0_squared


def _as_design_matrix(X) -> np.ndarray:
    if isinstance(X, pd.DataFrame):
        return np.asarray(X, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return X


def conjugate_posterior_linear(
    X,
    y,
    tau: float = 1.0,
    sigma_0_squared: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Posterior for beta with design [1, X] and diagonal prior variances
    [sigma_0^2, tau^2, ..., tau^2].
    Returns (posterior_mean, posterior_cov, sigma_squared).
    """
    X = _as_design_matrix(X)
    y = np.asarray(y, dtype=float).ravel()
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
    cov_post = np.linalg.inv(prec_post)
    mean_post = cov_post @ (Xd.T @ y / sigma_squared)
    return mean_post, cov_post, sigma_squared


def posterior_linear_samples(
    X,
    y,
    n_samples: int,
    tau: float = 1.0,
    sigma_0_squared: float = 1.0,
    random_state: int | None = None,
) -> np.ndarray:
    """Draw posterior coefficient vectors (shape n_samples x (p+1))."""
    mean_post, cov_post, _ = conjugate_posterior_linear(X, y, tau=tau, sigma_0_squared=sigma_0_squared)
    rng = np.random.default_rng(random_state)
    return rng.multivariate_normal(mean_post, cov_post, size=int(max(1, n_samples)))


class ConjugateBayesianLinearRegression:
    """IWFR Bayesian Linear Regression (BLiR) with explicit tau and sigma_0^2 priors."""

    def __init__(self, tau: float = 1.0, sigma_0_squared: float = 1.0):
        self.tau = float(tau)
        self.sigma_0_squared = float(sigma_0_squared)
        self.coef_: np.ndarray | None = None
        self.posterior_cov_: np.ndarray | None = None
        self.sigma_squared_: float | None = None

    @classmethod
    def from_params(cls, params: dict | None) -> "ConjugateBayesianLinearRegression":
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
        X = _as_design_matrix(X)
        n = X.shape[0]
        Xd = np.column_stack([np.ones(n, dtype=float), X])
        return Xd @ self.coef_
