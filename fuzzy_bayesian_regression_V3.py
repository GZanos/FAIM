"""
Fuzzy Bayesian Linear Regression (FBLiR) for Time Series Forecasting
Python implementation compatible with FAIM forecasting framework
"""

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
    
    def defuzzify(self, m=0.1, small_delta_threshold=0.4):
        """
        Defuzzification using optimal m parameter and smallDelta threshold
        Based on the R code defuzzification approach
        
        Parameters:
        - m: adjustment magnitude (optimal value typically around 0.1-0.3)
        - small_delta_threshold: threshold for delta = mean/sqrt(variance) (default 0.4)
        
        Returns:
        - predicted value using: mean + m * variance (if delta < threshold)
        - or just mean (if delta >= threshold)
        """
        if self.variance <= 0:
            return self.mean
        
        delta = self.get_delta()
        
        if delta < small_delta_threshold:
            # Use adjustment: mean + m * variance
            return self.mean + m * self.variance
        else:
            # Use just mean
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
        self.is_fitted = False
        
    def _bayesian_inference(self, X, y):
        """
        Conjugate Gaussian inference: beta_0 ~ N(0, sigma_0^2), beta_j ~ N(0, tau^2).
        """
        from bayesian_linear_core import posterior_linear_samples

        n, p = X.shape
        samples = posterior_linear_samples(
            X,
            y,
            n_samples=self.n_samples,
            tau=self.tau,
            sigma_0_squared=self.sigma_0_squared,
        )
        # samples columns: [beta_0, beta_1, ..., beta_p] for design [1, X]
        intercept_mean = float(np.mean(samples[:, 0]))
        intercept_var = float(np.var(samples[:, 0]))
        intercept_gfn = GeneralizedFuzzyNumber(intercept_mean, max(intercept_var, 1e-12))

        beta_gfns = []
        for j in range(p):
            col = j + 1
            beta_mean = float(np.mean(samples[:, col]))
            beta_var = float(np.var(samples[:, col]))
            beta_var = beta_var * self.uncertainty_weight + (
                (1 - self.uncertainty_weight) * float(np.mean(np.var(samples, axis=0)))
            )
            beta_gfns.append(GeneralizedFuzzyNumber(beta_mean, max(beta_var, 1e-12)))

        mean_coef = np.mean(samples, axis=0)
        Xd = np.column_stack([np.ones(n, dtype=float), X])
        y_hat = Xd @ mean_coef
        sigma_squared = float(max(np.var(y - y_hat), 1e-12))
        sigma_gfn = GeneralizedFuzzyNumber(0.0, sigma_squared)

        return intercept_gfn, beta_gfns, sigma_gfn
    
    def _add_quadratic_features(self, X):
        """Add quadratic terms to feature matrix"""
        if self.use_quadratic:
            X_quad = X ** 2
            return np.hstack([X, X_quad])
        return X
    
    def fit(self, X, y):
        """
        Fit Fuzzy Bayesian regression model
        
        Parameters:
        - X: Feature matrix (n_samples, n_features)
        - y: Target vector (n_samples,)
        """
        from bayesian_linear_core import sanitize_float_matrix, sanitize_float_vector

        X_arr = sanitize_float_matrix(X)
        y_arr = sanitize_float_vector(y)
        X_scaled = self.scaler_X.fit_transform(X_arr)
        y_scaled = self.scaler_y.fit_transform(_ensure_2d_array(y_arr)).ravel()
        
        # Add quadratic features
        X_augmented = self._add_quadratic_features(X_scaled)
        X_augmented = np.clip(X_augmented, -1e8, 1e8)
        
        # Bayesian inference
        self.intercept, self.coefficients, self.sigma_gfn = \
            self._bayesian_inference(X_augmented, y_scaled)
        
        self.is_fitted = True
        return self
    
    def _fuzzify_features(self, X):
        """Convert features to GFNs"""
        return [GeneralizedFuzzyNumber(val, self.fuzzify_variance) 
                for val in X]
    
    def predict(self, X):
        """
        Make predictions using fuzzy arithmetic
        
        Parameters:
        - X: Feature matrix (n_samples, n_features)
        
        Returns:
        - predictions: numpy array of predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        from bayesian_linear_core import sanitize_float_matrix

        X_arr = sanitize_float_matrix(X)
        X_scaled = self.scaler_X.transform(X_arr)
        
        # Add quadratic features
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
            
            # Add residual uncertainty
            y_gfn = GeneralizedFuzzyNumber.add(y_gfn, self.sigma_gfn)
            
            # Defuzzify using Gaussian Fuzzy Number defuzzification
            # Uses m parameter and small_delta_threshold based on R code
            y_pred_scaled = y_gfn.defuzzify(
                m=self.m, 
                small_delta_threshold=self.small_delta_threshold
            )
            
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
        
    def fit(self, X, y, X_val=None, y_val=None):
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
                                model.fit(X, y)
                                y_pred = model.predict(X_val)
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
        
        super().fit(X, y)
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

