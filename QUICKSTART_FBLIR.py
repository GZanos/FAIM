"""
QUICK START: Minimal FBLiR Integration Example for FAIM
========================================================

This is the SIMPLEST way to add FBLiR to your FAIM app.
Copy this code into your wildfire_forecast_app_V1_5_1.py
"""

# ===========================================================================
# STEP 1: Add at the top of your file (with other imports)
# ===========================================================================
try:
    from fuzzy_bayesian_regression import FuzzyBayesianRegressionTuned
    FBLIR_AVAILABLE = True
    print("✓ FBLiR loaded successfully")
except ImportError:
    FBLIR_AVAILABLE = False
    print("✗ FBLiR not available")


# ===========================================================================
# STEP 2: Find your forecasting model training section and add FBLiR
# ===========================================================================

# Example: In your existing train_forecast_models function:

def train_forecast_models_EXAMPLE(df, target_col, forecast_days=30):
    """
    Your existing function - just add FBLiR to the models dict
    """
    
    # Your existing data preparation code...
    X_train, X_val = ...  # Your train/val split
    y_train, y_val = ...
    
    # Your existing models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    }
    
    # ADD THIS: FBLiR model
    if FBLIR_AVAILABLE:
        models['FBLiR'] = FuzzyBayesianRegressionTuned(
            n_samples=500,           # Number of Bayesian samples
            use_quadratic=True,      # Include squared terms
            verbose=False            # Set True to see tuning progress
        )
    
    # Your existing training loop
    results = {}
    for name, model in models.items():
        try:
            # MODIFY THIS: Handle FBLiR's special fit method
            if name == 'FBLiR':
                # FBLiR needs validation data for hyperparameter tuning
                model.fit(X_train, y_train, X_val, y_val)
                y_pred = model.predict(X_val)
            else:
                # Standard sklearn models
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
            
            # Your existing metrics calculation
            mse = mean_squared_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            results[name] = {'mse': mse, 'r2': r2, 'model': model}
            
        except Exception as e:
            st.warning(f"{name} failed: {str(e)}")
    
    return results


# ===========================================================================
# STEP 3: Add FBLiR to your UI
# ===========================================================================

"""
In your sidebar forecasting mode, add code like this:

if data_mode == "Forecasting":
    st.sidebar.subheader("🤖 Forecast Methods")
    
    # Your existing method options
    method_options = [
        "Linear Regression",
        "Random Forest",
        "Gradient Boosting",
        "Prophet"
    ]
    
    # ADD THIS: Add FBLiR if available
    if FBLIR_AVAILABLE:
        method_options.append("FBLiR")
    
    # Your existing multiselect
    selected_methods = st.sidebar.multiselect(
        "Select Methods",
        options=method_options,
        default=["Random Forest"]
    )
    
    # ADD THIS: Optional FBLiR settings
    if FBLIR_AVAILABLE and "FBLiR" in selected_methods:
        with st.sidebar.expander("⚙️ FBLiR Settings"):
            st.info("FBLiR: Fuzzy Bayesian Linear Regression")
            st.markdown('''
            - Provides uncertainty estimates
            - Good for noisy data
            - Automatic hyperparameter tuning
            ''')
"""


# ===========================================================================
# STEP 4: Display FBLiR results
# ===========================================================================

"""
In your results display section, add code like this:

if forecast_result and 'models' in forecast_result:
    st.subheader("📊 Model Performance")
    
    for name, metrics in forecast_result['models'].items():
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write(f"**{name}**")
        with col2:
            st.metric("R²", f"{metrics['r2']:.3f}")
        with col3:
            st.metric("RMSE", f"{np.sqrt(metrics['mse']):.2f}")
        
        # ADD THIS: Special display for FBLiR
        if name == 'FBLiR':
            # Show uncertainty level
            if metrics['r2'] > 0.85:
                st.success("🟢 Low uncertainty - High confidence predictions")
            elif metrics['r2'] > 0.7:
                st.info("🟡 Medium uncertainty - Moderate confidence")
            else:
                st.warning("🔴 High uncertainty - Use with caution")
"""


# ===========================================================================
# COMPLETE MINIMAL EXAMPLE
# ===========================================================================

"""
Here's a complete standalone example you can test:
"""

def minimal_fblir_example():
    import numpy as np
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.ensemble import RandomForestRegressor
    
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(200, 5)
    y = X @ np.array([2, -1, 0.5, -0.3, 1]) + np.random.randn(200) * 0.3
    
    # Split
    X_train, X_test = X[:150], X[150:]
    y_train, y_test = y[:150], y[150:]
    X_val, y_val = X[120:150], y[120:150]  # Last 30 samples of train for validation
    
    # Compare models
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    }
    
    if FBLIR_AVAILABLE:
        from fuzzy_bayesian_regression import FuzzyBayesianRegressionTuned
        models['FBLiR'] = FuzzyBayesianRegressionTuned(
            n_samples=300,  # Fewer samples for speed
            use_quadratic=True,
            verbose=False
        )
    
    # Train and compare
    print("\n" + "="*60)
    print("Model Comparison")
    print("="*60)
    
    for name, model in models.items():
        if name == 'FBLiR':
            model.fit(X_train, y_train, X_val, y_val)
        else:
            model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"\n{name}:")
        print(f"  R²:   {r2:.4f}")
        print(f"  RMSE: {rmse:.4f}")
    
    print("\n" + "="*60)


# Run the example
if __name__ == "__main__":
    if FBLIR_AVAILABLE:
        print("\n🚀 Running minimal FBLiR example...")
        minimal_fblir_example()
        print("\n✅ Example complete! FBLiR is working.")
        print("\nNow you can integrate it into your FAIM app!")
    else:
        print("\n❌ FBLiR not available.")
        print("\nTo fix this:")
        print("1. Make sure fuzzy_bayesian_regression.py is in the same directory")
        print("2. Run: python fuzzy_bayesian_regression.py (to test)")
        print("3. Try importing: python -c 'from fuzzy_bayesian_regression import FuzzyBayesianRegressionTuned'")


# ===========================================================================
# CHECKLIST
# ===========================================================================
"""
✓ INTEGRATION CHECKLIST:

□ Step 1: Place fuzzy_bayesian_regression.py in your app directory
□ Step 2: Add import at top of app file
□ Step 3: Add FBLiR to models dictionary
□ Step 4: Handle FBLiR's special fit method (needs val data)
□ Step 5: Add "FBLiR" to method selection UI
□ Step 6: Test with sample data
□ Step 7: Add to your actual forecasting code
□ Step 8: Display results with uncertainty indicator

DONE! FBLiR is now integrated with your FAIM app.

KEY POINTS TO REMEMBER:
1. FBLiR.fit() needs BOTH training AND validation data
2. FBLiR.predict() works like any sklearn model
3. Higher R² = lower uncertainty
4. Use 500-1000 samples for production
5. Enable quadratic for non-linear patterns
"""
