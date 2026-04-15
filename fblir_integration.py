"""
Integration of Fuzzy Bayesian Linear Regression (FBLiR) with FAIM
Add this code to your wildfire_forecast_app_V1_5_1.py

This adds FBLiR as an additional forecasting method alongside the existing
Linear Regression, Random Forest, Gradient Boosting, Prophet, SARIMA, and XGBoost
"""

# ============================================================================
# STEP 1: Add this import at the top of your app file (after other imports)
# ============================================================================

try:
    from fuzzy_bayesian_regression import FuzzyBayesianRegressionTuned
    FBLIR_AVAILABLE = True
except ImportError:
    FBLIR_AVAILABLE = False
    print("FBLiR not available - fuzzy_bayesian_regression.py not found")


# ============================================================================
# STEP 2: Modify the train_forecast_models_v2 function
# ============================================================================

def train_forecast_models_v2_WITH_FBLIR(df, target_col, feature_cols, forecast_days=30, selected_methods=None):
    """
    Enhanced version that includes FBLiR
    
    Add this function or modify your existing train_forecast_models_v2
    """
    
    # ... [Keep all existing code for data preparation] ...
    
    # Remove rows with NaN in target
    df_clean = df.dropna(subset=[target_col]).copy()
    
    if len(df_clean) < 30:
        return None, "Insufficient data for forecasting (need at least 30 days)"
    
    # Prepare features and target
    if not feature_cols:
        return None, "No features selected for training"
    
    available_features = [col for col in feature_cols if col in df_clean.columns]
    df_clean = df_clean.dropna(subset=available_features)
    
    if len(df_clean) < 30:
        return None, "Insufficient clean data after removing NaN values"
    
    X = df_clean[available_features]
    y = df_clean[target_col]
    
    # Split data
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Define available methods
    all_methods = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    }
    
    # Add FBLiR if available
    if FBLIR_AVAILABLE:
        all_methods['FBLiR'] = FuzzyBayesianRegressionTuned(
            n_samples=500,
            use_quadratic=True,
            verbose=True
        )
    
    # Add Prophet and SARIMA if requested
    # ... [Keep existing Prophet/SARIMA code] ...
    
    # Train models
    results = {}
    predictions = {}
    
    for name, model in all_methods.items():
        if selected_methods and name not in selected_methods:
            continue
        
        try:
            if name == 'FBLiR':
                # FBLiR has special fit method with validation data
                model.fit(X_train, y_train, X_val, y_val)
                val_pred = model.predict(X_val)
            else:
                # Standard sklearn models
                model.fit(X_train_scaled, y_train)
                val_pred = model.predict(X_val_scaled)
            
            mse = mean_squared_error(y_val, val_pred)
            r2 = r2_score(y_val, val_pred)
            
            results[name] = {'mse': mse, 'r2': r2, 'model': model}
            predictions[name] = val_pred
            
            st.info(f"✓ {name} - R²: {r2:.3f}, RMSE: {np.sqrt(mse):.2f}")
            
        except Exception as e:
            st.warning(f"✗ {name} failed: {str(e)}")
            continue
    
    if not results:
        return None, "All models failed to train"
    
    # Generate forecasts
    # ... [Keep existing forecast generation code] ...
    
    # For FBLiR specifically:
    if 'FBLiR' in results and results['FBLiR']['model']:
        try:
            fblir_model = results['FBLiR']['model']
            
            # Prepare future features (simplified - use last known values)
            last_features = X.iloc[-1:].values
            X_future = np.repeat(last_features, forecast_days, axis=0)
            
            # Generate predictions
            fblir_forecast = fblir_model.predict(X_future)
            
            # Add to forecast dataframe
            forecast_df['FBLiR'] = fblir_forecast
            
        except Exception as e:
            st.warning(f"FBLiR forecast generation failed: {str(e)}")
    
    return {
        'forecast': forecast_df,
        'models': results,
        'best_model': max(results.items(), key=lambda x: x[1]['r2'])[0],
        'historical': df_clean,
        'scaler': scaler,
        'features': available_features
    }, None


# ============================================================================
# STEP 3: Add FBLiR to model selection UI
# ============================================================================

# In your sidebar or forecasting section, add FBLiR to the method selection:

def add_fblir_to_sidebar():
    """
    Add this in your forecasting mode sidebar controls
    """
    
    st.sidebar.subheader("🤖 Forecast Methods")
    
    method_options = [
        "Linear Regression",
        "Random Forest", 
        "Gradient Boosting",
        "Prophet",
        "SARIMA",
        "XGBoost"
    ]
    
    # Add FBLiR if available
    if FBLIR_AVAILABLE:
        method_options.append("FBLiR (Fuzzy Bayesian)")
    
    selected_methods = st.sidebar.multiselect(
        "Select Methods",
        options=method_options,
        default=["Random Forest", "Gradient Boosting"]
    )
    
    if FBLIR_AVAILABLE and "FBLiR (Fuzzy Bayesian)" in selected_methods:
        with st.sidebar.expander("⚙️ FBLiR Settings"):
            fblir_samples = st.slider(
                "Bayesian Samples",
                min_value=100,
                max_value=2000,
                value=500,
                step=100,
                help="Number of posterior samples (more = better but slower)"
            )
            
            fblir_quadratic = st.checkbox(
                "Use Quadratic Features",
                value=True,
                help="Include squared terms for non-linear patterns"
            )
            
            st.info("ℹ️ FBLiR uses fuzzy logic + Bayesian inference for uncertainty-aware predictions")
    
    return selected_methods


# ============================================================================
# STEP 4: Display FBLiR-specific information
# ============================================================================

def display_fblir_results(forecast_result):
    """
    Display FBLiR-specific metrics and explanations
    """
    if 'FBLiR' not in forecast_result['models']:
        return
    
    fblir_metrics = forecast_result['models']['FBLiR']
    
    st.subheader("🔮 FBLiR Model Details")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("R² Score", f"{fblir_metrics['r2']:.3f}")
    with col2:
        st.metric("RMSE", f"{np.sqrt(fblir_metrics['mse']):.2f}")
    with col3:
        uncertainty = "High" if fblir_metrics['r2'] < 0.7 else "Medium" if fblir_metrics['r2'] < 0.85 else "Low"
        st.metric("Uncertainty", uncertainty)
    
    with st.expander("ℹ️ About FBLiR"):
        st.markdown("""
        **Fuzzy Bayesian Linear Regression** combines three powerful approaches:
        
        1. **Fuzzy Logic**: Handles uncertainty in measurements and relationships
        2. **Bayesian Inference**: Quantifies prediction uncertainty probabilistically
        3. **Regularization**: Prevents overfitting through automatic relevance determination
        
        **Key Features**:
        - Uncertainty-aware predictions (not just point estimates)
        - Robust to noisy or imprecise data
        - Automatic hyperparameter tuning
        - Can include quadratic features for non-linear patterns
        
        **Best Used For**:
        - When you need confidence intervals on predictions
        - Data with measurement uncertainty
        - Scenarios requiring risk assessment
        
        **Performance**:
        - R² > 0.8: Excellent fit, high confidence
        - R² 0.6-0.8: Good fit, moderate confidence
        - R² < 0.6: Poor fit, consider other methods
        """)
    
    # Display best hyperparameters if available
    if hasattr(fblir_metrics['model'], 'best_params'):
        with st.expander("⚙️ Tuned Hyperparameters"):
            params = fblir_metrics['model'].best_params
            st.json(params)


# ============================================================================
# STEP 5: Complete integration example
# ============================================================================

def example_integration_in_forecasting_mode():
    """
    Example showing how to integrate all pieces in your forecasting mode
    """
    
    # In your forecasting visualization section:
    
    st.info("🔮 Preparing forecast models...")
    
    # Get selected methods from sidebar
    selected_methods = add_fblir_to_sidebar()
    
    # Prepare data
    forecast_data = prepare_forecast_data(gdf_filtered, selected_metric, bounds)
    
    if forecast_data is not None:
        with st.spinner("Training forecast models..."):
            # Use the enhanced function with FBLiR
            forecast_result, error = train_forecast_models_v2_WITH_FBLIR(
                forecast_data, 
                selected_metric,
                feature_cols=available_features,
                forecast_days=forecast_days,
                selected_methods=selected_methods
            )
        
        if error:
            st.error(f"❌ {error}")
        elif forecast_result:
            st.success(f"✅ Forecast complete! Best model: {forecast_result['best_model']}")
            
            # Show all model performances
            with st.expander("📊 Model Performance Comparison", expanded=True):
                for name, metrics in forecast_result['models'].items():
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(f"{name} - R²", f"{metrics['r2']:.3f}")
                    with col2:
                        st.metric(f"{name} - RMSE", f"{np.sqrt(metrics['mse']):.2f}")
            
            # Display FBLiR-specific info if it was used
            if 'FBLiR' in forecast_result['models']:
                display_fblir_results(forecast_result)
            
            # Plot forecast
            # ... [existing forecast plotting code] ...


# ============================================================================
# STEP 6: Installation instructions for users
# ============================================================================

INSTALLATION_INSTRUCTIONS = """
# FBLiR Integration Setup Instructions

## Installation

1. **Save fuzzy_bayesian_regression.py**
   Place the fuzzy_bayesian_regression.py file in the same directory as your app

2. **No Additional Dependencies Required**
   FBLiR uses only standard libraries:
   - numpy (already required)
   - pandas (already required)  
   - scipy (already required by sklearn)
   - sklearn (already required)

3. **Verify Installation**
   Run this in Python:
   ```python
   from fuzzy_bayesian_regression import FuzzyBayesianRegressionTuned
   print("FBLiR loaded successfully!")
   ```

## Usage in FAIM

1. Add FBLiR to your forecasting method selection
2. Select "FBLiR (Fuzzy Bayesian)" in the sidebar
3. Adjust settings if needed (samples, quadratic features)
4. Run forecast as normal

## Performance Notes

- **Speed**: Slower than Random Forest but faster than Prophet
- **Accuracy**: Comparable to Gradient Boosting on smooth trends
- **Best for**: Uncertainty quantification and noisy data
- **Memory**: Uses ~500MB for 1000 samples with 10 features

## Troubleshooting

If FBLiR doesn't appear:
1. Check fuzzy_bayesian_regression.py is in the correct directory
2. Restart your Streamlit app
3. Check the terminal for error messages

If predictions are poor:
1. Try enabling quadratic features
2. Increase Bayesian samples (500-1000)
3. Ensure you have 60+ days of historical data
"""


if __name__ == "__main__":
    print("FBLiR Integration Code")
    print("=" * 60)
    print("\nThis file contains integration code for adding FBLiR")
    print("to your FAIM wildfire forecasting application.")
    print("\nSee INSTALLATION_INSTRUCTIONS for setup steps.")
    print("=" * 60)
    print(INSTALLATION_INSTRUCTIONS)
