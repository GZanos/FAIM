import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, box
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import requests
from datetime import datetime, timedelta
import time
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
import json
import html as html_module
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Try to import FBLiR (Fuzzy Bayesian Linear Regression)
FBLIR_AVAILABLE = False
FuzzyBayesianRegression = None
FuzzyBayesianRegressionTuned = None

try:
    from fuzzy_bayesian_regression_V3 import FuzzyBayesianRegression, FuzzyBayesianRegressionTuned
    FBLIR_AVAILABLE = True
except (ImportError, ModuleNotFoundError, Exception) as e:
    try:
        # Fallback to V2 if V3 is not available
        from fuzzy_bayesian_regression_V2 import FuzzyBayesianRegression, FuzzyBayesianRegressionTuned
        FBLIR_AVAILABLE = True
    except (ImportError, ModuleNotFoundError, Exception) as e:
        try:
            # Fallback to original if V2 is not available
            from fuzzy_bayesian_regression import FuzzyBayesianRegressionTuned
            # For original version, we'll use FuzzyBayesianRegressionTuned only
            FBLIR_AVAILABLE = True
        except (ImportError, ModuleNotFoundError, Exception) as e:
            FBLIR_AVAILABLE = False
            # FBLiR is optional - app will work without it
            # Note: Ensure fuzzy_bayesian_regression_V3.py (or V2/original) is in the same directory

# Configure page
st.set_page_config(
    page_title="FAIM - Forecasting Analyzer of Ignition Metrics",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----- Optional demo access gate: only people with the password can use the app -----
# Set password via: Streamlit Cloud "Secrets" (demo_password) or env var DEMO_PASSWORD.
# If no password is set, everyone can access (e.g. for local run).
import os
_demo_password = None
try:
    _demo_password = st.secrets.get("demo_password")
except Exception:
    pass
if _demo_password is None:
    _demo_password = os.environ.get("DEMO_PASSWORD")
if _demo_password:
    if "demo_access_granted" not in st.session_state:
        st.session_state["demo_access_granted"] = False
    if not st.session_state["demo_access_granted"]:
        st.title("🔒 Demo access")
        p = st.text_input("Enter access password", type="password", key="demo_pwd")
        if st.button("Continue"):
            if p == _demo_password:
                st.session_state["demo_access_granted"] = True
                st.rerun()
            else:
                st.error("Incorrect password.")
        st.stop()

try:
    from faim_guide_markdown import GUIDE_MARKDOWN
except ImportError:
    GUIDE_MARKDOWN = "**Guide text not found.** Add `faim_guide_markdown.py` next to the app."

if hasattr(st, "dialog"):

    @st.dialog("FAIM — How to use", width="large")
    def faim_howto_dialog():
        st.markdown(GUIDE_MARKDOWN)

else:

    def faim_howto_dialog():
        st.sidebar.warning("Upgrade Streamlit to 1.33+ for the guide popup.")

# Title and description
st.title("🎯 FAIM - Forecasting Analyzer of Ignition Metrics")
st.markdown("""
Advanced meteorological forecasting and fire risk analysis platform.

*(forecasting is usually well under a minute for tree/boosting models; FBLiR now trains once per forecast run instead of once per forecast day, which greatly reduces runtime while using the same method)*
""")
if st.session_state.get("forecast_insights_md"):
    st.subheader("💡 Useful insights")
    st.markdown(st.session_state["forecast_insights_md"])
    st.markdown("---")

# NASA POWER API configuration
NASA_POWER_BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

# Available parameters from NASA POWER API
AVAILABLE_PARAMETERS = {
    "T2M": "Temperature at 2 Meters (°C)",
    "T2M_MAX": "Temperature at 2 Meters Maximum (°C)", 
    "T2M_MIN": "Temperature at 2 Meters Minimum (°C)",
    "T2M_RANGE": "Temperature at 2 Meters Range (°C)",
    "RH2M": "Relative Humidity at 2 Meters (%)",
    "T2MDEW": "Dew Point Temperature at 2 Meters (°C)",
    "T2MWET": "Wet Bulb Temperature at 2 Meters (°C)",
    "PRECTOT": "Precipitation (mm/day)",
    "WS10M": "Wind Speed at 10 Meters (m/s)",
    "WD10M": "Wind Direction at 10 Meters (degrees)",
    "TS": "Earth Skin Temperature (°C)"
}

@st.cache_data(ttl=3600)
def fetch_nasa_power_data(lat, lon, start_date, end_date, parameters):
    """Fetch data from NASA POWER API"""
    params = {
        "start": start_date.strftime("%Y%m%d"),
        "end": end_date.strftime("%Y%m%d"),
        "latitude": lat,
        "longitude": lon,
        "community": "AG",
        "parameters": ",".join(parameters),
        "format": "JSON"
    }
    
    try:
        response = requests.get(NASA_POWER_BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if "properties" not in data or "parameter" not in data["properties"]:
            st.warning(f"No data available for coordinates ({lat:.3f}, {lon:.3f})")
            return pd.DataFrame()
        
        properties = data["properties"]["parameter"]
        dates = pd.date_range(start_date, end_date)
        
        records = []
        for date in dates:
            date_str = date.strftime("%Y%m%d")
            record = {"date": date, "lat": lat, "lon": lon}
            for param in parameters:
                if param in properties and date_str in properties[param]:
                    value = properties[param][date_str]
                    record[param] = value if value != -999.0 else np.nan
                else:
                    record[param] = np.nan
            records.append(record)
        
        return pd.DataFrame(records)
    
    except requests.exceptions.RequestException as e:
        st.error(f"Network error fetching data for point ({lat:.3f}, {lon:.3f}): {str(e)}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error processing data for point ({lat:.3f}, {lon:.3f}): {str(e)}")
        return pd.DataFrame()

def calculate_robust_fwi(temp, humidity, wind, precip):
    """Calculate Fire Weather Index with robust handling of missing values"""
    def safe_float(val):
        if pd.isna(val) or val == -999.0 or val == -999:
            return np.nan
        try:
            return float(val)
        except (ValueError, TypeError):
            return np.nan
    
    t = safe_float(temp)
    h = safe_float(humidity)
    w = safe_float(wind)
    p = safe_float(precip)
    
    if pd.isna(t):
        t = 20.0
    if pd.isna(h):
        h = 50.0
    if pd.isna(w):
        w = 5.0
    if pd.isna(p):
        p = 1.0
    
    t = max(-50, min(60, t))
    h = max(1, min(100, h))
    w = max(0, min(50, w))
    p = max(0, min(500, p))
    
    try:
        temp_factor = max(0, (t - 5) / 30)
        humidity_factor = max(0, (100 - h) / 100)
        wind_factor = min(1, w / 25)
        precip_factor = max(0, 1 - (p / 10))
        
        fwi = (temp_factor * 0.3 + humidity_factor * 0.4 + 
               wind_factor * 0.2 + precip_factor * 0.1) * 100
        
        return max(0, min(100, fwi))
        
    except Exception:
        return np.nan

def calculate_afdr(temp, humidity, wind, precip, drought_factor=1.0):
    """Calculate Australian Fire Danger Rating (AFDR)"""
    def safe_float(val):
        if pd.isna(val) or val == -999.0 or val == -999:
            return np.nan
        try:
            return float(val)
        except (ValueError, TypeError):
            return np.nan
    
    t = safe_float(temp)
    h = safe_float(humidity)
    w = safe_float(wind)
    p = safe_float(precip)
    
    if pd.isna(t):
        t = 20.0
    if pd.isna(h):
        h = 50.0
    if pd.isna(w):
        w = 5.0
    if pd.isna(p):
        p = 1.0
    
    t = max(-10, min(50, t))
    h = max(1, min(100, h))
    w_kmh = max(0, min(100, w * 3.6))
    p = max(0, min(500, p))
    
    try:
        df = drought_factor * (1 + max(0, 1 - p / 5))
        df = max(1, min(10, df))
        
        ffdi = df * np.exp(
            0.987 * np.log(df) - 0.45 - 0.0345 * h + 0.0338 * t + 0.0234 * w_kmh
        )
        
        afdr = min(100, ffdi * 2)
        return max(0, afdr)
        
    except Exception:
        return np.nan

def get_afdr_category(afdr_value):
    """Get AFDR category and emoji"""
    if afdr_value < 12:
        return "Low-Moderate", "🟢"
    elif afdr_value < 24:
        return "High", "🟡"
    elif afdr_value < 50:
        return "Very High", "🟠"
    elif afdr_value < 75:
        return "Severe", "🔴"
    else:
        return "Extreme", "⚫"

def create_sample_data_for_period(start_date, end_date):
    """Create sample data for a specific date period"""
    lats = np.linspace(40.5, 42.0, 15)
    lons = np.linspace(-78.5, -76.5, 15)
    dates = pd.date_range(start_date, end_date, freq="D")
    
    records = []
    rng = np.random.default_rng(42)
    
    for date in dates:
        for lat in lats:
            for lon in lons:
                day_of_year = date.dayofyear
                base_temp = 20 + 15 * np.sin(2 * np.pi * day_of_year / 365)
                temp_noise = rng.normal(0, 5)
                
                record = {
                    "lat": lat,
                    "lon": lon,
                    "date": date,
                    "T2M": base_temp + temp_noise,
                    "T2M_MAX": base_temp + temp_noise + rng.uniform(5, 15),
                    "T2M_MIN": base_temp + temp_noise - rng.uniform(5, 10),
                    "T2M_RANGE": rng.uniform(10, 25),
                    "RH2M": rng.uniform(30, 80),
                    "T2MDEW": base_temp + temp_noise - rng.uniform(5, 15),
                    "T2MWET": base_temp + temp_noise - rng.uniform(2, 8),
                    "PRECTOT": rng.exponential(2),
                    "WS10M": rng.gamma(2, 2),
                    "WD10M": rng.uniform(0, 360),
                    "TS": base_temp + temp_noise + rng.normal(0, 2)
                }
                
                record["FWI"] = calculate_robust_fwi(
                    record["T2M"], 
                    record["RH2M"], 
                    record["WS10M"], 
                    record["PRECTOT"]
                )
                
                record["AFDR"] = calculate_afdr(
                    record["T2M"],
                    record["RH2M"],
                    record["WS10M"],
                    record["PRECTOT"]
                )
                
                records.append(record)
    
    df = pd.DataFrame(records)
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326")

def create_continuous_heatmap(bounds, values_grid, gradient_colormap='RdYlBu_r', opacity=0.4, metric_name="Value", full_map_bounds=None):
    """
    Create a smooth continuous heatmap overlay using HeatMap plugin with blue gradient
    - Smooth flowing patterns instead of rectangles
    - Blue-only gradient for professional appearance
    - Map features remain visible underneath
    Returns: (heatmap_layer, legend_html)
    """
    minx, miny, maxx, maxy = bounds
    grid_height, grid_width = values_grid.shape
    
    # Get value range for normalization and legend
    vmin = np.nanmin(values_grid)
    vmax = np.nanmax(values_grid)
    
    # Convert grid to HeatMap format: [[lat, lon, normalized_intensity], ...]
    heatmap_data = []
    for i in range(grid_height):
        for j in range(grid_width):
            value = values_grid[i, j]
            if not np.isnan(value):
                # Calculate position
                lat = miny + (i / grid_height) * (maxy - miny)
                lon = minx + (j / grid_width) * (maxx - minx)
                
                # Normalize intensity to 0-1 range
                if vmax > vmin:
                    normalized_intensity = (value - vmin) / (vmax - vmin)
                else:
                    normalized_intensity = 0.5
                
                heatmap_data.append([lat, lon, normalized_intensity])
    
    # Create feature group for the heatmap
    heatmap_layer = folium.FeatureGroup(name='Heatmap')
    
    # Add smooth HeatMap with blue gradient
    HeatMap(
        heatmap_data,
        min_opacity=0.3,      # Transparent enough to see map
        max_opacity=0.7,      # Strong enough to show patterns
        radius=25,            # Larger radius for smooth blending
        blur=20,              # High blur for continuous appearance
        gradient={
            0.0: '#E3F2FD',   # Very light blue (low values)
            0.2: '#90CAF9',   # Light blue
            0.4: '#42A5F5',   # Medium-light blue
            0.6: '#1E88E5',   # Medium blue
            0.8: '#1565C0',   # Dark blue
            1.0: '#0D47A1'    # Very dark blue (high values)
        }
    ).add_to(heatmap_layer)
    
    safe_title = html_module.escape(str(metric_name))
    mean_g = float(np.nanmean(values_grid))
    std_g = float(np.nanstd(values_grid))
    inner_legend = (
        f'<p style="margin:0 0 8px 0;font-weight:bold;text-align:center;color:#000;font-size:14px;line-height:1.2;">{safe_title}</p>'
        '<div style="margin-bottom:6px;">'
        '<div style="width:100%;height:18px;background:linear-gradient(to right,'
        '#E3F2FD,#90CAF9,#42A5F5,#1E88E5,#1565C0,#0D47A1);border:2px solid #333;border-radius:3px;"></div>'
        '</div>'
        '<div style="display:flex;justify-content:space-between;font-size:12px;margin-top:4px;color:#000;font-weight:600;">'
        f'<span><b>Low:</b> {vmin:.1f}</span><span><b>High:</b> {vmax:.1f}</span>'
        '</div>'
        '<div style="text-align:center;font-size:11px;margin-top:6px;padding-top:6px;border-top:2px solid #ccc;color:#000;font-weight:600;">'
        f'<div><b>Mean:</b> {mean_g:.1f}</div><div><b>Std:</b> {std_g:.1f}</div>'
        '</div>'
    )
    inner_json = json.dumps(inner_legend)
    # Anchor legend to the map pane (top-right of the Leaflet container), not the browser viewport
    legend_html = f"""
<script>
(function () {{
  function faimPlaceHeatmapLegend() {{
    var containers = document.querySelectorAll('div.leaflet-container');
    if (!containers.length) return;
    var c = containers[containers.length - 1];
    if (c.querySelector('.faim-heatmap-legend-anchor')) return;
    c.style.position = 'relative';
    var anchor = document.createElement('div');
    anchor.className = 'faim-heatmap-legend-anchor';
    anchor.style.cssText = 'position:absolute;top:10px;right:10px;width:186px;z-index:6500;font-size:13px;pointer-events:none;';
    var pane = document.createElement('div');
    pane.style.cssText = 'background-color:rgba(255,255,255,0.95);border:3px solid #333;border-radius:8px;padding:10px;box-shadow:0 4px 20px rgba(0,0,0,0.4);pointer-events:auto;';
    pane.innerHTML = {inner_json};
    anchor.appendChild(pane);
    c.appendChild(anchor);
  }}
  setTimeout(faimPlaceHeatmapLegend, 0);
  setTimeout(faimPlaceHeatmapLegend, 200);
  setTimeout(faimPlaceHeatmapLegend, 800);
}})();
</script>
"""
    
    return heatmap_layer, legend_html

def create_forecast_heatmap_grid(bounds, forecast_value, grid_size=100):
    """Create a grid of forecast values with spatial variation for smooth blending"""
    # Create base grid
    values_grid = np.full((grid_size, grid_size), forecast_value, dtype=float)
    
    # Add realistic spatial variation (3% standard deviation)
    if forecast_value != 0:
        variation = np.random.normal(0, abs(forecast_value) * 0.03, (grid_size, grid_size))
        values_grid = values_grid + variation
    
    return values_grid

def _idw_weights_matrix(query_xy, data_xy, power=2, eps=1e-9):
    """query_xy (N,2), data_xy (M,2) -> weights (N,M) normalized rows."""
    # (N,1) - (1,M) broadcasting for lon and lat
    dlon = query_xy[:, 0:1] - data_xy[None, :, 0]
    dlat = query_xy[:, 1:2] - data_xy[None, :, 1]
    dist = np.hypot(dlon, dlat)
    w = 1.0 / (np.power(dist, power) + eps)
    w /= np.sum(w, axis=1, keepdims=True)
    return w


def _enrich_near_uniform_heatmap(values_grid, aoi_bounds, obs_v, reference_std=None):
    """
    When the interpolated surface is almost flat (typical with few grid cells on one day),
    add smooth zero-mean spatial variation scaled by max(obs std, reference std from summary
    time series, small floor). Preserves approximate area-mean while opening the colour scale.
    """
    minx, miny, maxx, maxy = aoi_bounds
    vg = np.asarray(values_grid, dtype=float)
    vm = float(np.nanmean(vg))
    spatial_rng = float(np.nanmax(vg) - np.nanmin(vg))
    obs_arr = np.asarray(obs_v, dtype=float)
    obs_arr = obs_arr[np.isfinite(obs_arr)]
    obs_std = float(np.std(obs_arr)) if obs_arr.size > 1 else 0.0
    ref = float(reference_std) if reference_std is not None and np.isfinite(reference_std) else 0.0
    sigma = max(obs_std, ref, 0.02 * max(abs(vm), 1.0), 1e-9)
    flat_threshold = max(1e-9 * max(abs(vm), 1.0), 0.12 * sigma, 1e-6)
    if spatial_rng > flat_threshold:
        return vg
    h, w = vg.shape
    lon_edges = np.linspace(minx, maxx, w + 1, dtype=float)
    lat_edges = np.linspace(miny, maxy, h + 1, dtype=float)
    lon_c = (lon_edges[:-1] + lon_edges[1:]) * 0.5
    lat_c = (lat_edges[:-1] + lat_edges[1:]) * 0.5
    lon_m, lat_m = np.meshgrid(lon_c, lat_c)
    u = (lon_m - 0.5 * (minx + maxx)) / max(0.5 * (maxx - minx), 1e-9)
    v = (lat_m - 0.5 * (miny + maxy)) / max(0.5 * (maxy - miny), 1e-9)
    bump = (
        0.55 * np.sin(np.pi * u) * np.cos(np.pi * v)
        + 0.35 * np.sin(2.3 * np.pi * u + 0.7)
        + 0.25 * np.cos(2.1 * np.pi * v - 0.4)
    )
    bump = bump - np.nanmean(bump)
    sb = float(np.nanstd(bump))
    if sb > 1e-12:
        bump = bump / sb
    return vg + sigma * bump


def create_heatmap_data(gdf, aoi_bounds, metric, grid_size=80, n_anchor_lon=5, n_anchor_lat=4, reference_std=None):
    """
    Build a smooth value grid over the AOI using inverse-distance interpolation.

    Nearest-neighbour on a dense grid often maps many cells to the same station (flat colour).
    We place n_anchor_lon * n_anchor_lat (default 20) anchors on a regular lattice in the bbox,
    assign each anchor from observations by IDW, then interpolate each cell centre from anchors
    by IDW again. One global min/max (via create_continuous_heatmap) keeps a single colour scale.

    reference_std: optional temporal std (e.g. AOI daily-mean series) used when the spatial
    surface is nearly uniform so the heatmap still spans a sensible colour range.
    """
    minx, miny, maxx, maxy = aoi_bounds

    mask = (
        (gdf.geometry.x >= minx) & (gdf.geometry.x <= maxx) &
        (gdf.geometry.y >= miny) & (gdf.geometry.y <= maxy)
    )
    gdf_aoi = gdf[mask].copy()

    if gdf_aoi.empty:
        return None, []

    obs_lon = gdf_aoi.geometry.x.to_numpy(dtype=float)
    obs_lat = gdf_aoi.geometry.y.to_numpy(dtype=float)
    obs_v = gdf_aoi[metric].to_numpy(dtype=float)
    finite = np.isfinite(obs_v)
    obs_lon, obs_lat, obs_v = obs_lon[finite], obs_lat[finite], obs_v[finite]
    if obs_v.size == 0:
        return None, []

    obs_xy = np.column_stack([obs_lon, obs_lat])

    ax = np.linspace(minx, maxx, n_anchor_lon, dtype=float)
    ay = np.linspace(miny, maxy, n_anchor_lat, dtype=float)
    alon, alat = np.meshgrid(ax, ay)
    anchor_xy = np.column_stack([alon.ravel(), alat.ravel()])

    w_oa = _idw_weights_matrix(anchor_xy, obs_xy, power=2, eps=1e-9)
    anchor_vals = w_oa @ obs_v

    lon_edges = np.linspace(minx, maxx, grid_size + 1, dtype=float)
    lat_edges = np.linspace(miny, maxy, grid_size + 1, dtype=float)
    lon_c = (lon_edges[:-1] + lon_edges[1:]) * 0.5
    lat_c = (lat_edges[:-1] + lat_edges[1:]) * 0.5
    Lon, Lat = np.meshgrid(lon_c, lat_c)
    grid_xy = np.column_stack([Lon.ravel(), Lat.ravel()])

    w_ga = _idw_weights_matrix(grid_xy, anchor_xy, power=2, eps=1e-7)
    flat = w_ga @ anchor_vals
    values_grid = flat.reshape(grid_size, grid_size)
    values_grid = _enrich_near_uniform_heatmap(values_grid, aoi_bounds, obs_v, reference_std=reference_std)

    values_list = values_grid[np.isfinite(values_grid)].ravel().tolist()
    return values_grid, values_list

def make_color_scale(value, vmin, vmax):
    """Generate color based on value range"""
    if vmax == vmin:
        return "rgb(100, 100, 100)"
    
    normalized = (value - vmin) / (vmax - vmin)
    r = int(255 * normalized)
    g = int(255 * (1 - abs(normalized - 0.5) * 2))
    b = int(255 * (1 - normalized))
    
    return f"rgb({r},{g},{b})"

def create_distribution_plot(values, metric_name, metric_description):
    """Create a histogram with statistical annotations"""
    if not values or len(values) == 0:
        return None
    
    mean_val = np.mean(values)
    median_val = np.median(values)
    std_val = np.std(values)
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=values,
        nbinsx=20,
        name='Distribution',
        opacity=0.7,
        marker_color='skyblue',
        marker_line_color='navy',
        marker_line_width=1
    ))
    
    fig.add_vline(
        x=mean_val, 
        line_dash="dash", 
        line_color="red",
        line_width=3,
        annotation_text=f"Mean: {mean_val:.2f}",
        annotation_position="top"
    )
    
    fig.add_vline(
        x=median_val, 
        line_dash="dot", 
        line_color="green", 
        line_width=3,
        annotation_text=f"Median: {median_val:.2f}",
        annotation_position="top"
    )
    
    fig.add_vrect(
        x0=mean_val - std_val, 
        x1=mean_val + std_val,
        fillcolor="yellow", 
        opacity=0.2,
        line_width=0,
        annotation_text=f"±1σ ({std_val:.2f})",
        annotation_position="top left"
    )
    
    fig.update_layout(
        title=f"{metric_name} Distribution",
        xaxis_title=metric_description,
        yaxis_title="Frequency",
        height=400,
        showlegend=False,
        plot_bgcolor='white',
        font=dict(size=12)
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    return fig

def prepare_ml_features(df, target_col, feature_cols, lag_days=7):
    """Prepare features for ML models including lags and rolling statistics"""
    df = df.sort_values('date').copy()
    
    # FIXED: Only use feature columns that exist in the dataframe
    valid_feature_cols = [col for col in feature_cols if col in df.columns]
    
    if not valid_feature_cols:
        st.error(f"None of the selected features are available in the data")
        return df, []
    
    # Create lag features for target
    for lag in range(1, lag_days + 1):
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    
    # Create rolling statistics for target
    for window in [3, 7, 14]:
        df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window=window, min_periods=1).mean()
        df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window=window, min_periods=1).std()
    
    # Add selected feature columns
    feature_names = [f'{target_col}_lag_{i}' for i in range(1, lag_days + 1)]
    feature_names += [f'{target_col}_rolling_mean_{w}' for w in [3, 7, 14]]
    feature_names += [f'{target_col}_rolling_std_{w}' for w in [3, 7, 14]]
    feature_names += valid_feature_cols  # Use only valid features
    
    # Add time-based features
    df['day_of_year'] = df['date'].dt.dayofyear
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    feature_names += ['day_of_year', 'month', 'day_of_week']
    
    # FIXED: Remove rows with too many NaN values (keep rows with < 30% NaN in features)
    df_features = df[feature_names]
    nan_threshold = len(feature_names) * 0.3
    df['nan_count'] = df_features.isna().sum(axis=1)
    df_clean = df[df['nan_count'] < nan_threshold].copy()
    df_clean = df_clean.drop('nan_count', axis=1)
    
    # Fill remaining NaN with forward fill then backward fill (pandas 2.2+ removed fillna(method=...))
    for col in feature_names:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].ffill().bfill().fillna(0)
    
    return df_clean, feature_names

def get_seasonal_feature_value(historical_data, feature_name, target_date, lookback_years=3):
    """
    ✨ NEW FUNCTION: Get feature value for a future date based on historical seasonal patterns
    This ensures features follow their natural seasonality in forecasts
    
    Args:
        historical_data: DataFrame with historical data including 'date', 'day_of_year', 'month'
        feature_name: Name of the feature to project (e.g., 'T2M', 'RH2M')
        target_date: Future date to project the feature for
        lookback_years: How many years to look back (default 3)
    
    Returns:
        Projected feature value based on historical seasonal patterns
    """
    target_doy = target_date.dayofyear
    target_month = target_date.month
    
    # Find historical values from same time of year (±7 days window)
    doy_window = 7
    matching_rows = historical_data[
        ((historical_data['day_of_year'] >= target_doy - doy_window) & 
         (historical_data['day_of_year'] <= target_doy + doy_window)) |
        (historical_data['month'] == target_month)
    ]
    
    if len(matching_rows) > 0 and feature_name in matching_rows.columns:
        # Use the mean of matching historical values
        seasonal_value = matching_rows[feature_name].mean()
        
        # Add small random variation (±5%) for realism
        variation = np.random.normal(0, abs(seasonal_value) * 0.05)
        return seasonal_value + variation
    else:
        # Fallback: use overall mean
        if feature_name in historical_data.columns:
            return historical_data[feature_name].mean()
        else:
            return 0.0

def train_linear_regression(X_train, y_train, X_future):
    """Train Linear Regression model"""
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_future)
    return predictions

def train_random_forest(X_train, y_train, X_future, params=None):
    """Train Random Forest model"""
    if params is None:
        params = {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 2}
    
    model = RandomForestRegressor(
        n_estimators=params.get('n_estimators', 100),
        max_depth=params.get('max_depth', 10),
        min_samples_split=params.get('min_samples_split', 2),
        random_state=42, 
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_future)
    return predictions

def train_gradient_boosting(X_train, y_train, X_future, params=None):
    """Train Gradient Boosting model"""
    if params is None:
        params = {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1}
    
    model = GradientBoostingRegressor(
        n_estimators=params.get('n_estimators', 100),
        max_depth=params.get('max_depth', 5),
        learning_rate=params.get('learning_rate', 0.1),
        random_state=42
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_future)
    return predictions

def train_fblir(X_train, y_train, X_val, y_val, X_future, params=None):
    """Train Fuzzy Bayesian Linear Regression model"""
    if not FBLIR_AVAILABLE:
        raise ImportError("FBLiR is not available. Ensure fuzzy_bayesian_regression_V3.py (or V2/original) is in the app directory.")
    
    if params is None:
        params = {
            'm': 0.5,
            'k': 0.5,
            'fuzzification_factor': 0.05,
                'symmetry_threshold': 0.4,
            'N_chains': 1,
            'adapt_steps': 100,
            'burnin_steps': 100,
            'thinning_steps': 1
        }
    
    try:
        # Check if we have FuzzyBayesianRegression (V2/V3) or only FuzzyBayesianRegressionTuned (original)
        if FuzzyBayesianRegression is not None:
            # Use V2/V3: Calculate n_samples from Bayesian parameters
            # Note: n_samples is the number of posterior samples, not total MCMC steps
            # We use a reasonable calculation: base samples per chain, scaled by chains and thinning
            adapt = int(params.get("adapt_steps", 100))
            burnin = int(params.get("burnin_steps", 100))
            n_chains = max(1, int(params.get("N_chains", 1)))
            thinning = max(1, int(params.get("thinning_steps", 1)))
            n_samples = min(max(100, ((adapt + burnin) * n_chains) // thinning), 2200)
            
            # Use FuzzyBayesianRegression directly to pass all parameters
            # Using Gaussian Fuzzy Numbers (GFN) operations from Abdalla & Buckley 2007
            model = FuzzyBayesianRegression(
                n_samples=n_samples,
                symmetry_threshold=params.get('symmetry_threshold', 0.5),
                k=params.get('k', 0.5),  # Deprecated but kept for compatibility
                m=params.get('m', 0.1),  # Defuzzification magnitude (optimal typically 0.1-0.3)
                fuzzify_variance=params.get('fuzzification_factor', 0.05),
                use_quadratic=True,
                small_delta_threshold=params.get('symmetry_threshold', 0.4)  # Delta threshold for defuzzification (from R code)
            )
            # Fit the model
            model.fit(X_train, y_train)
            predictions = model.predict(X_future)
            return predictions
        else:
            # Fallback to original FuzzyBayesianRegressionTuned (doesn't support all parameters)
            # Use reasonable n_samples calculation
            base_samples = 500
            n_chains = max(1, params.get('N_chains', 1))
            n_samples = min(base_samples * n_chains, 2000)  # Cap at 2000 for original version
            n_samples = max(n_samples, 100)  # Minimum
            
            model = FuzzyBayesianRegressionTuned(
                n_samples=n_samples,
                use_quadratic=True
            )
            # Original version needs validation data
            model.fit(X_train, y_train, X_val, y_val)
            predictions = model.predict(X_future)
            return predictions
    except Exception as e:
        raise Exception(f"FBLiR training failed: {str(e)}")

def train_xgboost(X_train, y_train, X_future, params=None):
    """Train XGBoost model"""
    try:
        import xgboost as xgb
        
        if params is None:
            params = {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1}
        
        model = xgb.XGBRegressor(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', 6),
            learning_rate=params.get('learning_rate', 0.1),
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_future)
        return predictions
    except ImportError:
        raise ImportError("XGBoost is not installed. Install with: pip install xgboost")
    except Exception as e:
        raise Exception(f"XGBoost training failed: {str(e)}")

def train_prophet(df_prophet, forecast_horizon, params=None):
    """Train Facebook Prophet model"""
    try:
        from prophet import Prophet
        
        if params is None:
            params = {
                'yearly_seasonality': True,
                'weekly_seasonality': True,
                'daily_seasonality': True,
                'seasonality_mode': 'multiplicative'
            }
        
        # Suppress Prophet's verbose output
        import logging
        logging.getLogger('prophet').setLevel(logging.ERROR)
        
        # Prophet requires specific column names
        model = Prophet(
            daily_seasonality=params.get('daily_seasonality', True),
            weekly_seasonality=params.get('weekly_seasonality', True),
            yearly_seasonality=params.get('yearly_seasonality', True),
            seasonality_mode=params.get('seasonality_mode', 'multiplicative')
        )
        model.fit(df_prophet)
        
        # Make future dataframe
        future = model.make_future_dataframe(periods=forecast_horizon)
        forecast = model.predict(future)
        
        # Return only future predictions
        predictions = forecast['yhat'].iloc[-forecast_horizon:].values
        return predictions
        
    except ImportError:
        raise ImportError("Prophet is not installed. Install with: pip install prophet")
    except Exception as e:
        raise Exception(f"Prophet training failed: {str(e)}")

def train_sarima(y_train, forecast_horizon, params=None):
    """Train SARIMA model"""
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        import warnings
        warnings.filterwarnings('ignore')
        
        if params is None:
            params = {'order': (1, 1, 1), 'seasonal_order': (1, 1, 1, 7)}
        
        # Fit SARIMA model with seasonal components
        model = SARIMAX(
            y_train,
            order=params.get('order', (1, 1, 1)),  # (p, d, q)
            seasonal_order=params.get('seasonal_order', (1, 1, 1, 7)),  # (P, D, Q, s) - weekly seasonality
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        fitted_model = model.fit(disp=False)
        
        # Forecast
        predictions = fitted_model.forecast(steps=forecast_horizon)
        return predictions.values
        
    except ImportError:
        raise ImportError("statsmodels is not installed. Install with: pip install statsmodels")
    except Exception as e:
        raise Exception(f"SARIMA training failed: {str(e)}")


def _get_openai_api_key():
    """API key from Streamlit secrets or OPENAI_API_KEY environment variable."""
    try:
        k = st.secrets.get("openai_api_key")
        if k:
            return k
    except Exception:
        pass
    return os.environ.get("OPENAI_API_KEY")


def _insights_model_id():
    """Fixed model for post-forecast insights (override via secrets or env)."""
    try:
        m = st.secrets.get("insights_openai_model")
        if m:
            return str(m).strip()
    except Exception:
        pass
    return (os.environ.get("INSIGHTS_OPENAI_MODEL") or "gpt-4o-mini").strip()


def _acf_numpy(series, max_lag=40):
    """Normalized autocorrelation (biased), lags 0..max_lag."""
    y = np.asarray(series, dtype=float)
    y = y[np.isfinite(y)]
    n = len(y)
    if n < max_lag + 5:
        max_lag = max(0, n // 3)
    y = y - np.mean(y)
    denom = float(np.dot(y, y))
    if denom <= 1e-12:
        return np.zeros(max_lag + 1), max_lag
    acf = [1.0]
    for k in range(1, max_lag + 1):
        acf.append(float(np.dot(y[:-k], y[k:]) / denom))
    return np.array(acf, dtype=float), max_lag


def _acf_series_remove_slow_variation(y, period_days=365.25, min_points=90):
    """
    Residual after removing linear trend + annual harmonic (sin/cos at ~1 year).

    Raw daily ACF often decays slowly when the series has trend and strong seasonal
    drift; that is not a bug — classic ACF mixes those effects. This residual makes
    shorter-lag and seasonal structure easier to see in an exploratory plot.
    """
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    n = len(y)
    if n < min_points:
        return y - np.mean(y), False
    t = np.arange(n, dtype=float)
    y0 = y - np.mean(y)
    X = np.column_stack(
        [
            np.ones(n),
            t,
            np.sin(2.0 * np.pi * t / period_days),
            np.cos(2.0 * np.pi * t / period_days),
        ]
    )
    coef, *_ = np.linalg.lstsq(X, y0, rcond=None)
    resid = y0 - X @ coef
    return resid, True


def _acf_plotly(series, title="Autocorrelation", max_lag=40, adjust_slow_acf=True):
    y_raw = np.asarray(series, dtype=float)
    y_raw = y_raw[np.isfinite(y_raw)]
    adjusted = False
    if adjust_slow_acf and len(y_raw) >= 90:
        y_work, adjusted = _acf_series_remove_slow_variation(y_raw)
    else:
        y_work = y_raw - np.mean(y_raw)

    acf, ml = _acf_numpy(y_work, max_lag=max_lag)
    lags = np.arange(len(acf))
    fig = go.Figure()
    fig.add_trace(
        go.Bar(x=lags, y=acf, marker_color="#42A5F5", name="ACF")
    )
    ci = 1.96 / max(np.sqrt(len(y_work)), 1)
    fig.add_hline(y=ci, line_dash="dash", line_color="gray", annotation_text="approx. 95% noise band")
    fig.add_hline(y=-ci, line_dash="dash", line_color="gray")
    sub = (
        "residual: linear trend + annual cycle removed"
        if adjusted
        else "demeaned series (short history — full adjustment from 90+ days)"
    )
    fig.update_layout(
        title=f"{title}<br><sup style='font-size:11px'>{sub}</sup>",
        xaxis_title="Lag (days)",
        yaxis_title="ACF",
        height=360,
        showlegend=False,
        margin=dict(t=60, b=40),
    )
    return fig


def _seasonal_calendar_profile(daily_avg, forecast_target, future_dates, doy_window=7):
    """Mean target by day-of-year (±window) across all history — damped seasonal anchor."""
    df = daily_avg[["date", forecast_target]].dropna().copy()
    if df.empty:
        return None
    df["doy"] = df["date"].dt.dayofyear
    vals = []
    for fd in future_dates:
        tdoy = int(fd.timetuple().tm_yday)
        diff = (df["doy"] - tdoy).abs()
        diff = np.minimum(diff, 366 - diff)
        mask = diff <= doy_window
        block = df.loc[mask, forecast_target]
        vals.append(float(block.mean()) if len(block) else float(df[forecast_target].mean()))
    return np.array(vals, dtype=float)


def _calendar_month_summary_markdown(daily_avg, forecast_target):
    df = daily_avg[["date", forecast_target]].dropna().copy()
    if df.empty:
        return "(no data)"
    df["month"] = df["date"].dt.month
    g = df.groupby("month")[forecast_target].agg(["mean", "std", "min", "max"]).round(2)
    lines = ["month,mean,std,min,max"]
    for m, row in g.iterrows():
        lines.append(f"{int(m)},{row['mean']},{row['std']},{row['min']},{row['max']}")
    return "\n".join(lines)


def _find_peaks_troughs_simple(dates, y, min_sep=20):
    y = np.asarray(y, dtype=float)
    peaks, troughs = [], []
    for i in range(1, len(y) - 1):
        if y[i] > y[i - 1] and y[i] > y[i + 1]:
            if not peaks or i - peaks[-1][0] >= min_sep:
                peaks.append((i, float(y[i]), dates.iloc[i]))
        elif y[i] < y[i - 1] and y[i] < y[i + 1]:
            if not troughs or i - troughs[-1][0] >= min_sep:
                troughs.append((i, float(y[i]), dates.iloc[i]))
    return peaks[-5:], troughs[-5:]


def _series_insight_context(daily_avg, forecast_target):
    """Structured stats for insights + LLM (trend, seasonality, extrema)."""
    df = daily_avg[["date", forecast_target]].dropna().copy()
    df = df.sort_values("date")
    if len(df) < 10:
        return {"error": "short series"}
    y = df[forecast_target].astype(float).values
    t = np.arange(len(y), dtype=float)
    slope = float(np.polyfit(t, y, 1)[0]) if len(y) > 1 else 0.0
    acf7 = float(_acf_numpy(y, max_lag=7)[0][7]) if len(y) > 14 else 0.0
    acf365 = None
    if len(y) > 400:
        ml = min(365, len(y) - 2)
        acf_full, _ = _acf_numpy(y, max_lag=ml)
        acf365 = float(acf_full[365]) if ml >= 365 else float(acf_full[-1])
    peaks, troughs = _find_peaks_troughs_simple(df["date"].reset_index(drop=True), pd.Series(y), min_sep=25)
    peak_desc = [
        f"{p[2].strftime('%Y-%m-%d') if hasattr(p[2], 'strftime') else p[2]} (value {p[1]:.2f})" for p in peaks
    ]
    trough_desc = [
        f"{p[2].strftime('%Y-%m-%d') if hasattr(p[2], 'strftime') else p[2]} (value {p[1]:.2f})" for p in troughs
    ]
    diffs = np.abs(np.diff(y))
    cp_idx = int(np.argmax(diffs)) if len(diffs) else 0
    cp_date = df["date"].iloc[cp_idx + 1] if cp_idx + 1 < len(df) else df["date"].iloc[-1]
    return {
        "n": len(y),
        "mean": float(np.mean(y)),
        "std": float(np.std(y)),
        "min": float(np.min(y)),
        "max": float(np.max(y)),
        "last": float(y[-1]),
        "trend_slope_per_day": slope,
        "acf_lag7": acf7,
        "acf_lag365": acf365,
        "recent_peaks": peak_desc,
        "recent_troughs": trough_desc,
        "largest_step_change_date": cp_date.strftime("%Y-%m-%d") if hasattr(cp_date, "strftime") else str(cp_date),
        "largest_step_change_size": float(diffs[cp_idx]) if len(diffs) else 0.0,
    }


def train_llm_horizon_forecast(
    daily_avg,
    forecast_target,
    forecast_horizon,
    future_dates,
    params=None,
):
    """
    LLM forecast with strong seasonal anchoring: the model sees long history + calendar stats,
    and the output is blended with a day-of-year profile so short end spikes do not dominate.
    """
    if params is None:
        params = {}
    hist_all = daily_avg[["date", forecast_target]].dropna().sort_values("date")
    if hist_all.empty:
        raise ValueError("No historical data for LLM forecast")

    seasonal = _seasonal_calendar_profile(daily_avg, forecast_target, future_dates, doy_window=7)
    if seasonal is None:
        seasonal = np.full(forecast_horizon, float(hist_all[forecast_target].mean()))

    tdesc = AVAILABLE_PARAMETERS.get(
        forecast_target,
        "Fire Weather Index" if forecast_target == "FWI" else "Australian Fire Danger Rating" if forecast_target == "AFDR" else forecast_target,
    )
    start_str = future_dates[0].strftime("%Y-%m-%d") if len(future_dates) else ""

    n = len(hist_all)
    tail_n = min(800, n)
    tail = hist_all.tail(tail_n)
    series_lines = "\n".join(
        f"{row['date'].strftime('%Y-%m-%d')},{float(row[forecast_target]):.6f}"
        for _, row in tail.iterrows()
    )
    month_tbl = _calendar_month_summary_markdown(hist_all, forecast_target)
    yall = hist_all[forecast_target].astype(float).values
    lo = float(np.percentile(yall, 2)) - 0.5 * float(np.std(yall) or 1.0)
    hi = float(np.percentile(yall, 98)) + 0.5 * float(np.std(yall) or 1.0)
    last7 = yall[-7:] if len(yall) >= 7 else yall
    last30 = yall[-30:] if len(yall) >= 30 else yall
    prompt_extra = (
        f"Series length: {n} days. Last-7 mean={float(np.mean(last7)):.4f}, std={float(np.std(last7)):.4f}. "
        f"Last-30 mean={float(np.mean(last30)):.4f}. "
        f"Full-sample min/max: {float(np.min(yall)):.3f} / {float(np.max(yall)):.3f}.\n"
        "IMPORTANT: The last few days may include noise or spikes. Do NOT extrapolate those linearly. "
        "Use the multi-year seasonal pattern (monthly table + typical yearly swings) as the backbone. "
        "Forecasts should stay near historical seasonal envelopes unless the monthly table justifies otherwise.\n"
        f"Monthly statistics CSV (month,mean,std,min,max):\n{month_tbl}\n"
    )

    def _blend_with_seasonal(llm_arr):
        w_season = 0.72
        blended = w_season * seasonal + (1.0 - w_season) * llm_arr
        return np.clip(blended, lo, hi)

    api_key = _get_openai_api_key()
    if not api_key:
        return np.clip(seasonal, lo, hi)

    try:
        from openai import OpenAI
    except ImportError:
        return np.clip(seasonal, lo, hi)

    model = params.get("model", "gpt-4o-mini")
    temperature = min(float(params.get("temperature", 0.15)), 0.35)
    client = OpenAI(api_key=api_key)
    prompt = f"""You are a scientific forecaster for "{forecast_target}" ({tdesc}).

{prompt_extra}

Recent daily history (date,value CSV, oldest to newest — {tail_n} days):
{series_lines}

Task: Output exactly {forecast_horizon} future daily values for dates starting {start_str}.
The trajectory should reflect recurring seasonality seen across ALL years, not a continuation of any short terminal spike.
Keep values mostly between {lo:.3f} and {hi:.3f} (historical bulk range).

Respond with ONLY valid JSON: {{"forecast":[ ... {forecast_horizon} numbers ... ]}} — no markdown, no prose."""

    import json as _json

    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=min(4096, 200 + forecast_horizon * 16),
            messages=[
                {"role": "system", "content": "You reply with only the JSON object requested."},
                {"role": "user", "content": prompt},
            ],
        )
        text = (resp.choices[0].message.content or "").strip()
        if text.startswith("```"):
            text = text.strip("`")
            if "json" in text[:20].lower():
                text = text.split("\n", 1)[-1]
        try:
            data = _json.loads(text)
        except _json.JSONDecodeError:
            i, j = text.find("{"), text.rfind("}")
            if i < 0 or j <= i:
                return np.clip(seasonal, lo, hi)
            data = _json.loads(text[i : j + 1])
        arr = data.get("forecast")
        if not isinstance(arr, list) or len(arr) != forecast_horizon:
            return np.clip(seasonal, lo, hi)
        llm_out = np.array([float(x) for x in arr], dtype=float)
        llm_out = np.clip(llm_out, lo, hi)
        return _blend_with_seasonal(llm_out)
    except Exception:
        return np.clip(seasonal, lo, hi)


def _enhanced_heuristic_insights_text(forecast_results, daily_avg, forecast_target, future_dates):
    """Deterministic bullets: trend, seasonality, peaks/troughs, forecasts — always returned."""
    lines = []
    ctx = _series_insight_context(daily_avg, forecast_target)
    hist = daily_avg[forecast_target].astype(float)
    last_h = float(hist.iloc[-1])
    hmin, hmax = float(hist.min()), float(hist.max())

    lines.append(
        f"- **Historical level:** last **{forecast_target}** = **{last_h:.2f}**; full-sample range **{hmin:.2f}–{hmax:.2f}**."
    )

    if not ctx.get("error"):
        slope = ctx["trend_slope_per_day"]
        if slope > 1e-4:
            tw = "rising"
        elif slope < -1e-4:
            tw = "falling"
        else:
            tw = "approximately flat"
        lines.append(
            f"- **Trend:** Over **{ctx['n']}** days the series is **{tw}** (OLS slope ~ **{slope:.6f}** per day)."
        )
        lines.append(
            f"- **Seasonality (autocorrelation):** lag-7 ACF ~ **{ctx['acf_lag7']:.2f}** (short cyclic memory)."
        )
        if ctx.get("acf_lag365") is not None:
            lines.append(
                f"- **Annual pattern:** ACF (long lag) ~ **{ctx['acf_lag365']:.2f}** (year-over-year similarity)."
            )
        if ctx["recent_peaks"]:
            lines.append("- **Recent peaks (local maxima):** " + "; ".join(ctx["recent_peaks"]) + ".")
        if ctx["recent_troughs"]:
            lines.append("- **Recent troughs (local minima):** " + "; ".join(ctx["recent_troughs"]) + ".")
        lines.append(
            f"- **Largest single-day step:** **{ctx['largest_step_change_size']:.2f}** on **{ctx['largest_step_change_date']}** "
            "(useful when judging end-of-sample spikes vs structural change)."
        )

    for method, pred in forecast_results.items():
        p = np.asarray(pred, dtype=float)
        mn, mx = float(np.mean(p)), float(np.max(p))
        trend = "up" if p[-1] > p[0] else "down" if p[-1] < p[0] else "flat"
        lines.append(
            f"- **{method} forecast:** mean **{mn:.2f}**, max **{mx:.2f}**, horizon **{trend}** (**{p[0]:.2f}** → **{p[-1]:.2f}**)."
        )

    if len(forecast_results) > 1:
        stack = np.stack([np.asarray(v, dtype=float) for v in forecast_results.values()], axis=0)
        spread = float(np.mean(np.max(stack, axis=0) - np.min(stack, axis=0)))
        lines.append(
            f"- **Model spread:** mean day-to-day range across methods ~ **{spread:.2f}**."
        )
    if len(future_dates):
        lines.append(
            f"- **Horizon:** **{len(future_dates)}** days (**{future_dates[0].strftime('%Y-%m-%d')}** → **{future_dates[-1].strftime('%Y-%m-%d')}**)."
        )
    return "\n".join(lines)


def generate_forecast_insights_markdown(
    forecast_results,
    daily_avg,
    forecast_target,
    future_dates,
):
    """
    Insights always include trend, seasonality, peaks/troughs (heuristic layer).
    If OPENAI_API_KEY / secrets are set, a fixed insights model rewrites/extends bullets.
    """
    base = _enhanced_heuristic_insights_text(
        forecast_results, daily_avg, forecast_target, future_dates
    )
    api_key = _get_openai_api_key()
    if not api_key:
        return base
    try:
        from openai import OpenAI
    except ImportError:
        return base

    model = _insights_model_id()
    ctx = _series_insight_context(daily_avg, forecast_target)
    summary = {
        m: {
            "mean": float(np.mean(v)),
            "min": float(np.min(v)),
            "max": float(np.max(v)),
            "first": float(np.asarray(v).ravel()[0]),
            "last": float(np.asarray(v).ravel()[-1]),
        }
        for m, v in forecast_results.items()
    }
    tdesc = AVAILABLE_PARAMETERS.get(
        forecast_target,
        "Fire Weather Index" if forecast_target == "FWI" else "Australian Fire Danger Rating" if forecast_target == "AFDR" else forecast_target,
    )
    client = OpenAI(api_key=api_key)
    prompt = f"""You are an analyst summarizing a wildfire / weather forecast for decision makers.

Target: {forecast_target} ({tdesc}).

Historical analytics (JSON): {ctx}

Per-method forecast summary (JSON): {summary}

Below is a deterministic bullet list you may refine (keep all key facts; you may merge or rephrase):
{base}

Output **6–10** bullet lines starting with "- ". You MUST explicitly comment on:
(1) overall trend, (2) seasonality / cyclic behavior, (3) notable peaks and troughs or change points in the history,
(4) how forecasts compare if multiple methods exist. Be concise; no JSON; do not invent dates not in the analytics."""

    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0.25,
            max_tokens=900,
            messages=[
                {
                    "role": "system",
                    "content": "You write executive bullet summaries only. Follow the required topics.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        text = (resp.choices[0].message.content or "").strip()
        if text and len(text) > 60:
            return text
    except Exception:
        pass
    return base


def run_iterative_ml_forecast(
    method,
    X_scaled,
    y,
    feature_names,
    df_ml,
    selected_features,
    forecast_target,
    forecast_horizon,
    scaler,
    model_params,
):
    """Fit tabular ML once, then recursive multi-step forecast (avoids refitting every day)."""
    predictions = []
    current_data = df_ml.copy()

    if method == "Linear Regression":
        model = LinearRegression()
        model.fit(X_scaled, y)

        def predict_one(row_scaled_df):
            return float(model.predict(row_scaled_df)[0])

    elif method == "Random Forest":
        mp = model_params.get("Random Forest") or {}
        model = RandomForestRegressor(
            n_estimators=mp.get("n_estimators", 100),
            max_depth=mp.get("max_depth", 10),
            min_samples_split=mp.get("min_samples_split", 2),
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_scaled, y)

        def predict_one(row_scaled_df):
            return float(model.predict(row_scaled_df)[0])

    elif method == "Gradient Boosting":
        mp = model_params.get("Gradient Boosting") or {}
        model = GradientBoostingRegressor(
            n_estimators=mp.get("n_estimators", 100),
            max_depth=mp.get("max_depth", 5),
            learning_rate=mp.get("learning_rate", 0.1),
            random_state=42,
        )
        model.fit(X_scaled, y)

        def predict_one(row_scaled_df):
            return float(model.predict(row_scaled_df)[0])

    elif method == "XGBoost":
        import xgboost as xgb

        mp = model_params.get("XGBoost") or {}
        model = xgb.XGBRegressor(
            n_estimators=mp.get("n_estimators", 100),
            max_depth=mp.get("max_depth", 6),
            learning_rate=mp.get("learning_rate", 0.1),
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_scaled, y)

        def predict_one(row_scaled_df):
            return float(model.predict(row_scaled_df)[0])

    elif method == "FBLiR":
        if not FBLIR_AVAILABLE:
            raise ImportError("FBLiR is not available.")
        params = model_params.get("FBLiR") or {}
        adapt = int(params.get("adapt_steps", 100))
        burnin = int(params.get("burnin_steps", 100))
        if FuzzyBayesianRegression is not None:
            thinning = max(1, int(params.get("thinning_steps", 1)))
            n_chains = max(1, int(params.get("N_chains", 1)))
            n_samples = min(max(100, ((adapt + burnin) * n_chains) // thinning), 2200)
            model = FuzzyBayesianRegression(
                n_samples=n_samples,
                symmetry_threshold=params.get("symmetry_threshold", 0.5),
                k=params.get("k", 0.5),
                m=params.get("m", 0.1),
                fuzzify_variance=params.get("fuzzification_factor", 0.05),
                use_quadratic=True,
                small_delta_threshold=params.get("symmetry_threshold", 0.4),
            )
            model.fit(X_scaled, y)

            def predict_one(row_scaled_df):
                pr = model.predict(row_scaled_df)
                return float(np.asarray(pr).ravel()[0])
        else:
            split_idx = int(len(X_scaled) * 0.8)
            X_train_f = X_scaled.iloc[:split_idx]
            y_train_f = y.iloc[:split_idx]
            X_val_f = X_scaled.iloc[split_idx:]
            y_val_f = y.iloc[split_idx:]
            base_samples = min(max(100, adapt + burnin), 1500)
            model = FuzzyBayesianRegressionTuned(n_samples=base_samples, use_quadratic=True)
            model.fit(X_train_f, y_train_f, X_val_f, y_val_f)

            def predict_one(row_scaled_df):
                pr = model.predict(row_scaled_df)
                return float(np.asarray(pr).ravel()[0])
    else:
        raise ValueError(f"Unknown iterative ML method: {method}")

    for _step in range(forecast_horizon):
        last_row = current_data.iloc[[-1]][feature_names]
        last_row_scaled = pd.DataFrame(
            scaler.transform(last_row),
            columns=feature_names,
        )
        pred = predict_one(last_row_scaled)
        predictions.append(pred)

        new_date = current_data["date"].iloc[-1] + timedelta(days=1)
        new_row = {
            "date": new_date,
            forecast_target: pred,
            "day_of_year": new_date.dayofyear,
            "month": new_date.month,
            "day_of_week": new_date.dayofweek,
        }
        for feat in selected_features:
            new_row[feat] = get_seasonal_feature_value(
                historical_data=df_ml,
                feature_name=feat,
                target_date=new_date,
            )
        for lag in range(1, 8):
            lag_col = f"{forecast_target}_lag_{lag}"
            if lag_col in feature_names:
                if lag == 1:
                    new_row[lag_col] = pred
                else:
                    new_row[lag_col] = current_data[f"{forecast_target}_lag_{lag-1}"].iloc[-1]
        for window in [3, 7, 14]:
            mean_col = f"{forecast_target}_rolling_mean_{window}"
            std_col = f"{forecast_target}_rolling_std_{window}"
            if mean_col in feature_names:
                recent_values = list(current_data[forecast_target].tail(window - 1)) + [pred]
                new_row[mean_col] = np.mean(recent_values)
            if std_col in feature_names:
                recent_values = list(current_data[forecast_target].tail(window - 1)) + [pred]
                new_row[std_col] = np.std(recent_values) if len(recent_values) > 1 else 0
        current_data = pd.concat([current_data, pd.DataFrame([new_row])], ignore_index=True)

    return np.array(predictions, dtype=float)


# Initialize session state
if 'gdf_data' not in st.session_state:
    st.session_state.gdf_data = None
if 'animation_running' not in st.session_state:
    st.session_state.animation_running = False
if 'current_frame' not in st.session_state:
    st.session_state.current_frame = 0
if 'forecast_frames' not in st.session_state:
    st.session_state.forecast_frames = None
if 'map_type' not in st.session_state:
    st.session_state.map_type = "Street"
if 'forecast_insights_md' not in st.session_state:
    st.session_state.forecast_insights_md = None

# Sidebar controls
st.sidebar.header("⚙️ Controls")

# Data source selection
data_source = st.sidebar.radio(
    "Data Source",
    ["Last 30 Days (Quick)", "NASA POWER API (Full)", "Forecasting"]
)
if data_source != "Forecasting":
    st.session_state.forecast_insights_md = None

# Date range selection
if data_source == "Last 30 Days (Quick)":
    start_date = datetime.now() - timedelta(days=33)
    end_date = datetime.now() - timedelta(days=3)
    st.sidebar.info("📅 Date Range: Last 30 days")
    st.sidebar.text(f"From: {start_date.strftime('%Y-%m-%d')}")
    st.sidebar.text(f"To: {end_date.strftime('%Y-%m-%d')}")
    
elif data_source == "NASA POWER API (Full)":
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime(2024, 1, 1),
            max_value=datetime.now() - timedelta(days=3)
        )
    with col2:
        end_date = st.date_input(
            "End Date", 
            value=datetime(2024, 1, 30),
            max_value=datetime.now() - timedelta(days=3)
        )
else:  # Forecasting mode
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input(
            "Historical Start",
            value=datetime(2024, 1, 1),
            max_value=datetime.now() - timedelta(days=3)
        )
    with col2:
        end_date = st.date_input(
            "Historical End", 
            value=datetime.now() - timedelta(days=3),
            max_value=datetime.now() - timedelta(days=3)
        )

if start_date >= end_date:
    st.sidebar.error("Start date must be before end date")
    st.stop()

# Forecast-specific controls
if data_source == "Forecasting":
    st.sidebar.subheader("🔮 Forecast Settings")
    
    forecast_target = st.sidebar.selectbox(
        "Target Metric to Forecast",
        options=list(AVAILABLE_PARAMETERS.keys()) + ["FWI", "AFDR"],
        index=0,
        format_func=lambda x: AVAILABLE_PARAMETERS.get(x, "Fire Weather Index" if x == "FWI" else "Australian Fire Danger Rating")
    )
    
    # NEW: Feature selection for forecasting with Select All option
    available_features = [k for k in list(AVAILABLE_PARAMETERS.keys()) + ["FWI", "AFDR"] if k != forecast_target]
    
    # Add Select All checkbox
    select_all_features = st.sidebar.checkbox("Select All Features", value=False)
    
    if select_all_features:
        selected_features = available_features
        st.sidebar.info(f"✅ All {len(selected_features)} features selected")
    else:
        selected_features = st.sidebar.multiselect(
            "Features for Target Metric Forecast",
            options=available_features,
            default=available_features[:3] if len(available_features) >= 3 else available_features,
            help="Select which metrics to use as features for training the forecast model"
        )
    
    if not selected_features:
        st.sidebar.warning("⚠️ Please select at least one feature")
    
    # Build forecast methods list
    available_forecast_methods = ["Linear Regression", "Random Forest", "Gradient Boosting", "XGBoost", 
                                   "Prophet", "SARIMA", "LLM Forecaster", "Ensemble"]
    
    # Add FBLiR if available
    if FBLIR_AVAILABLE:
        # Insert FBLiR after XGBoost (index 4)
        if "FBLiR" not in available_forecast_methods:
            available_forecast_methods.insert(4, "FBLiR")
    
    forecast_methods = st.sidebar.multiselect(
        "Forecast Methods",
        available_forecast_methods,
        default=["Random Forest"],
        help="Select one or more forecasting methods. Ensemble combines all selected methods. FBLiR provides uncertainty-aware predictions. LLM Forecaster uses OpenAI (see secrets / OPENAI_API_KEY)."
    )
    
    forecast_horizon = st.sidebar.slider(
        "Forecast Horizon (days)",
        min_value=7,
        max_value=730,
        value=30,
        step=1
    )
    
    # MODEL PARAMETERS SECTION
    st.sidebar.subheader("🎛️ Model Parameters")
    
    with st.sidebar.expander("⚙️ Configure Model Parameters"):
        model_params = {}
        
        if 'Random Forest' in forecast_methods:
            st.markdown("**Random Forest**")
            rf_n_estimators = st.number_input("n_estimators", min_value=10, max_value=500, value=100, step=10, key="rf_n_est")
            rf_max_depth = st.selectbox("max_depth", [None, 5, 10, 15, 20, 30], index=1, key="rf_max_depth")
            rf_min_samples_split = st.number_input("min_samples_split", min_value=2, max_value=20, value=2, step=1, key="rf_min_split")
            model_params['Random Forest'] = {
                'n_estimators': rf_n_estimators,
                'max_depth': rf_max_depth,
                'min_samples_split': rf_min_samples_split
            }
        
        if 'Gradient Boosting' in forecast_methods:
            st.markdown("**Gradient Boosting**")
            gb_n_estimators = st.number_input("n_estimators", min_value=10, max_value=500, value=100, step=10, key="gb_n_est")
            gb_learning_rate = st.number_input("learning_rate", min_value=0.01, max_value=1.0, value=0.1, step=0.01, key="gb_lr")
            gb_max_depth = st.number_input("max_depth", min_value=1, max_value=10, value=5, step=1, key="gb_max_depth")
            model_params['Gradient Boosting'] = {
                'n_estimators': gb_n_estimators,
                'learning_rate': gb_learning_rate,
                'max_depth': gb_max_depth
            }
        
        if 'XGBoost' in forecast_methods:
            st.markdown("**XGBoost**")
            xgb_n_estimators = st.number_input("n_estimators", min_value=10, max_value=500, value=100, step=10, key="xgb_n_est")
            xgb_learning_rate = st.number_input("learning_rate", min_value=0.01, max_value=1.0, value=0.1, step=0.01, key="xgb_lr")
            xgb_max_depth = st.number_input("max_depth", min_value=1, max_value=15, value=6, step=1, key="xgb_max_depth")
            model_params['XGBoost'] = {
                'n_estimators': xgb_n_estimators,
                'learning_rate': xgb_learning_rate,
                'max_depth': xgb_max_depth
            }
        
        if 'Prophet' in forecast_methods:
            st.markdown("**Prophet**")
            prophet_yearly = st.checkbox("yearly_seasonality", value=True, key="prophet_yearly")
            prophet_weekly = st.checkbox("weekly_seasonality", value=True, key="prophet_weekly")
            prophet_daily = st.checkbox("daily_seasonality", value=True, key="prophet_daily")
            prophet_seasonality_mode = st.selectbox("seasonality_mode", ['additive', 'multiplicative'], index=1, key="prophet_mode")
            model_params['Prophet'] = {
                'yearly_seasonality': prophet_yearly,
                'weekly_seasonality': prophet_weekly,
                'daily_seasonality': prophet_daily,
                'seasonality_mode': prophet_seasonality_mode
            }
        
        if 'SARIMA' in forecast_methods:
            st.markdown("**SARIMA**")
            st.text("Order (p, d, q)")
            sarima_p = st.number_input("p", min_value=0, max_value=5, value=1, step=1, key="sarima_p")
            sarima_d = st.number_input("d", min_value=0, max_value=2, value=1, step=1, key="sarima_d")
            sarima_q = st.number_input("q", min_value=0, max_value=5, value=1, step=1, key="sarima_q")
            st.text("Seasonal Order (P, D, Q, s)")
            sarima_P = st.number_input("P", min_value=0, max_value=3, value=1, step=1, key="sarima_P")
            sarima_D = st.number_input("D", min_value=0, max_value=2, value=1, step=1, key="sarima_D")
            sarima_Q = st.number_input("Q", min_value=0, max_value=3, value=1, step=1, key="sarima_Q")
            sarima_s = st.number_input("s (seasonal period)", min_value=1, max_value=365, value=7, step=1, key="sarima_s")
            model_params['SARIMA'] = {
                'order': (sarima_p, sarima_d, sarima_q),
                'seasonal_order': (sarima_P, sarima_D, sarima_Q, sarima_s)
            }
        
        if 'FBLiR' in forecast_methods and FBLIR_AVAILABLE:
            st.markdown("**FBLiR (Fuzzy Bayesian Linear Regression)**")
            
            # GFN Operation Parameters
            st.markdown("**GFN Operation Parameters:**")
            fblir_m = st.number_input("m (defuzzification magnitude)", min_value=0.0, max_value=1.0, value=0.1, step=0.01, key="fblir_m",
                                     help="Defuzzification magnitude for GFN operations (optimal typically 0.1-0.3). Used in: mean + m*variance when delta < threshold")
            fblir_k = st.number_input("k (defuzzification sensitivity)", min_value=-1.0, max_value=1.0, value=0.5, step=0.1, key="fblir_k",
                                     help="Defuzzification sensitivity parameter (deprecated but kept for compatibility)")
            fblir_fuzz = st.number_input("Fuzzification Factor", min_value=0.01, max_value=1.0, value=0.05, step=0.01, key="fblir_fuzz",
                                        help="Variance for fuzzifying features in GFN operations")
            fblir_symmetry = st.number_input("Symmetry Threshold (small_delta)", min_value=0.1, max_value=2.0, value=0.4, step=0.1, key="fblir_symmetry",
                                            help="Threshold for delta = |mean|/sqrt(variance). When delta < threshold, applies m*variance adjustment in defuzzification (default 0.4)")
            
            # Bayesian Inference Parameters
            st.markdown("**Bayesian Inference Parameters:**")
            fblir_n_chains = st.number_input("N_chains", min_value=1, max_value=10, value=1, step=1, key="fblir_n_chains",
                                            help="Number of MCMC chains for Bayesian inference")
            fblir_adapt_steps = st.number_input("adapt_steps", min_value=10, max_value=1000, value=100, step=10, key="fblir_adapt_steps",
                                               help="Number of adaptation steps for MCMC")
            fblir_burnin_steps = st.number_input("burnin_steps", min_value=10, max_value=1000, value=100, step=10, key="fblir_burnin_steps",
                                                 help="Number of burn-in steps for MCMC")
            fblir_thinning_steps = st.number_input("thinning_steps", min_value=1, max_value=50, value=1, step=1, key="fblir_thinning_steps",
                                                   help="Thinning interval for MCMC samples")
            
            model_params['FBLiR'] = {
                'm': fblir_m,
                'k': fblir_k,
                'fuzzification_factor': fblir_fuzz,
                'symmetry_threshold': fblir_symmetry,
                'N_chains': fblir_n_chains,
                'adapt_steps': fblir_adapt_steps,
                'burnin_steps': fblir_burnin_steps,
                'thinning_steps': fblir_thinning_steps
            }
        
        if "LLM Forecaster" in forecast_methods:
            st.markdown("**LLM Forecaster (OpenAI-compatible)**")
            llm_model = st.text_input(
                "Model name",
                value="gpt-4o-mini",
                key="llm_forecast_model",
                help="Chat model id (e.g. gpt-4o-mini). Set API key in secrets or OPENAI_API_KEY.",
            )
            llm_temp = st.slider(
                "temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.15,
                step=0.05,
                key="llm_forecast_temp",
            )
            model_params["LLM Forecaster"] = {
                "model": (llm_model or "gpt-4o-mini").strip(),
                "temperature": float(llm_temp),
            }
    
    selected_metric = forecast_target
    forecast_mode = True
else:
    selected_metric = st.sidebar.selectbox(
        "Select Parameter",
        options=list(AVAILABLE_PARAMETERS.keys()) + ["FWI", "AFDR"],
        index=0,
        format_func=lambda x: AVAILABLE_PARAMETERS.get(x, "Fire Weather Index" if x == "FWI" else "Australian Fire Danger Rating")
    )
    forecast_mode = False
    model_params = {}

# Animation controls (only for forecasting)
if data_source == "Forecasting":
    animation_speed = st.sidebar.slider(
        "Animation Speed (seconds)",
        min_value=0.3,
        max_value=2.0,
        value=0.5,
        step=0.1
    )

# Map controls in sidebar
st.sidebar.subheader("🗺️ Map Controls")

with st.sidebar.expander("⚙️ Configure Map Settings", expanded=True):
    map_zoom_level = st.slider(
        "Zoom Level",
        min_value=3,
        max_value=18,
        value=7,
        step=1,
        key="zoom_both",
        help="Control zoom for both maps"
    )
    
    heatmap_opacity = st.slider(
        "Heatmap Opacity",
        min_value=0.1,
        max_value=1.0,
        value=0.4,
        step=0.05,
        help="Control the transparency of heatmap overlays. Lower values show more of the underlying map"
    )

# AI Assistant Chatbot
def get_chatbot_response(user_message):
    """AI Assistant that answers questions about the app"""
    user_lower = user_message.lower()
    
    # Knowledge base with patterns and responses
    knowledge_base = {
        # General greetings
        'greeting': {
            'patterns': ['hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening'],
            'response': """Hello! 👋 I'm your FAIM Assistant. I can help you learn how to use the app!
            
**What I can help with:**
- How to use forecasting features
- Understanding model parameters
- Explaining FBLiR and other methods
- Map controls and visualization
- Data sources and parameters
- Troubleshooting

Ask me anything about the app!"""
        },
        
        # Forecasting questions
        'forecasting': {
            'patterns': ['forecast', 'forecasting', 'predict', 'prediction', 'how to forecast', 'forecast horizon'],
            'response': """📊 **Forecasting Guide:**

**To create forecasts:**
1. Select "Forecasting" as your data source
2. Choose a target metric (FWI, AFDR, or any parameter)
3. Select features to use for training
4. Choose forecast methods (Random Forest, XGBoost, FBLiR, etc.)
5. Set forecast horizon (7-730 days)
6. Configure model parameters if needed
7. Draw an area on the map and click "Generate Forecast"

**Forecast Methods:**
- **Random Forest**: Fast, good for most cases
- **XGBoost**: Powerful gradient boosting
- **FBLiR**: Uncertainty-aware, best for noisy data (takes 6-10 min)
- **Prophet**: Time series forecasting
- **SARIMA**: Seasonal ARIMA model
- **Ensemble**: Combines all selected methods

**Note:** Forecasting takes 20-40 seconds (FBLiR: 6-10 minutes)"""
        },
        
        # FBLiR questions
        'fblir': {
            'patterns': ['fblir', 'fuzzy', 'bayesian', 'uncertainty', 'fuzzy bayesian', 'gaussian fuzzy'],
            'response': """🔬 **FBLiR (Fuzzy Bayesian Linear Regression):**

**What is FBLiR?**
FBLiR combines Gaussian Fuzzy Numbers (GFN) and Bayesian inference for uncertainty-aware predictions. Based on Abdalla & Buckley 2007 GFN operations, it uses Gaussian fuzzy numbers (mean and variance) instead of triangular/trapezoidal. It's excellent for noisy data and provides robust forecasts.

**GFN Operations:**
- Uses exact GFN operations: add, subtract, multiply, divide
- Each GFN represents mean and variance
- Operations propagate uncertainty through calculations

**Parameters:**
- **GFN Operation Parameters:**
  - `m`: Defuzzification magnitude (0.0 to 1.0, optimal typically 0.1-0.3, default 0.1)
    - Used in defuzzification: mean + m*variance when delta < threshold
  - `k`: Defuzzification sensitivity (deprecated, kept for compatibility)
  - `Fuzzification Factor`: Variance for fuzzifying crisp values into GFNs (0.01-1.0)
  - `Symmetry Threshold (small_delta)`: Threshold for delta = |mean|/sqrt(variance) (0.1-2.0, default 0.4)
    - When delta < threshold, applies m*variance adjustment in defuzzification

- **Bayesian Parameters:**
  - `N_chains`: Number of MCMC chains (1-10, affects n_samples calculation)
  - `adapt_steps`: Adaptation steps (10-1000, used for n_samples calculation)
  - `burnin_steps`: Burn-in steps (10-1000, used for n_samples calculation)
  - `thinning_steps`: Thinning interval (1-50, affects n_samples calculation)

**Defuzzification:**
- Calculates delta = |mean| / sqrt(variance)
- If delta < small_delta_threshold: predicted = mean + m*variance
- If delta >= small_delta_threshold: predicted = mean

**Performance:** Takes 2-10 minutes depending on data size and n_samples."""
        },
        
        # Model parameters
        'parameters': {
            'patterns': ['parameter', 'model parameter', 'configure', 'settings', 'tune', 'hyperparameter'],
            'response': """⚙️ **Model Parameters Guide:**

**Access:** Sidebar → "⚙️ Configure Model Parameters" expander

**Available Models:**
1. **Random Forest:**
   - `n_estimators`: Number of trees (10-500)
   - `max_depth`: Max tree depth (None, 5-30)
   - `min_samples_split`: Min samples to split (2-20)

2. **Gradient Boosting:**
   - `n_estimators`: Number of boosting stages (10-500)
   - `max_depth`: Max depth (1-10)
   - `learning_rate`: Learning rate (0.01-0.5)

3. **XGBoost:**
   - `n_estimators`: Number of trees (10-500)
   - `max_depth`: Max depth (1-10)
   - `learning_rate`: Learning rate (0.01-0.5)

4. **FBLiR:** See FBLiR section for details

**Tip:** Start with defaults, then tune based on results!"""
        },
        
        # Map questions
        'map': {
            'patterns': ['map', 'draw', 'rectangle', 'area', 'aoi', 'selection', 'heatmap', 'visualization'],
            'response': """🗺️ **Map Guide:**

**Two Map Views:**
1. **📍 Selection Map:** Draw rectangles to define areas of interest
2. **🗺️ Spatial Distribution:** View heatmap visualizations

**How to Use:**
- Click and drag on the Selection Map to draw a rectangle
- The app will analyze data within that area
- Switch views using the toggle buttons

**Map Controls:**
- **Zoom Level:** 3-18 (higher = more detail)
- **Map Type:** Street, Terrain, or Detailed
- **Heatmap Opacity:** 0.1-1.0 (transparency control)

**For Forecasting:** Draw an area, then click "Generate Forecast" """
        },
        
        # Data source questions
        'data': {
            'patterns': ['data', 'source', 'nasa', 'power', 'api', 'download', 'fetch'],
            'response': """📡 **Data Sources:**

**Available Sources:**
1. **Last 30 Days (Quick):** Sample data for quick testing
2. **NASA POWER API:** Real meteorological data

**NASA POWER Parameters:**
- Temperature (T2M, T2M_MAX, T2M_MIN, T2M_RANGE)
- Humidity (RH2M, T2MDEW, T2MWET)
- Precipitation (PRECTOT)
- Wind (WS10M, WD10M)
- Earth Skin Temperature (TS)

**Calculated Metrics:**
- **FWI:** Fire Weather Index (0-100)
- **AFDR:** Australian Fire Danger Rating (0-100)

**Note:** Data is cached for 1 hour to improve performance."""
        },
        
        # Features questions
        'features': {
            'patterns': ['feature', 'select feature', 'which feature', 'input variable'],
            'response': """🎯 **Feature Selection:**

**For Forecasting:**
- Select features from the sidebar dropdown
- Choose metrics that correlate with your target
- Common choices: Temperature, Humidity, Wind, Precipitation
- You can select all features or choose specific ones

**Feature Engineering:**
The app automatically creates:
- Lag features (1-7 days)
- Rolling statistics (mean, std for 3, 7, 14 days)
- Time-based features (day of year, month, day of week)

**Tip:** More features ≠ better. Start with 3-5 most relevant features."""
        },
        
        # Troubleshooting
        'troubleshoot': {
            'patterns': ['error', 'problem', 'issue', 'not working', 'help', 'troubleshoot', 'bug', 'fix'],
            'response': """🔧 **Troubleshooting:**

**Common Issues:**

1. **FBLiR not showing:**
   - Ensure `fuzzy_bayesian_regression_V3.py` is in the app directory
   - Check that all dependencies are installed

2. **Forecast taking too long:**
   - Reduce forecast horizon
   - Use faster methods (Random Forest, XGBoost)
   - FBLiR takes 6-10 minutes (expected)

3. **No data showing:**
   - Check date range
   - Verify area selection on map
   - Try "Last 30 Days (Quick)" for testing

4. **Map not loading:**
   - Check internet connection
   - Try different map type
   - Adjust zoom level

**Still having issues?** Check the app documentation or contact support."""
        },
        
        # Ensemble questions
        'ensemble': {
            'patterns': ['ensemble', 'combine', 'multiple methods', 'average'],
            'response': """🎭 **Ensemble Method:**

**What is Ensemble?**
The Ensemble method combines predictions from all selected forecast methods by averaging their results.

**Benefits:**
- More robust predictions
- Reduces overfitting
- Better accuracy than individual methods
- Handles uncertainty better

**How it works:**
1. Select multiple methods (e.g., Random Forest + XGBoost + FBLiR)
2. Select "Ensemble" as an additional method
3. Each method makes predictions
4. Ensemble averages all predictions

**Best Practice:** Use 3-5 diverse methods for best results."""
        },
        
        # Default response
        'default': {
            'patterns': [],
            'response': """I'm here to help! 🤖 

I can answer questions about:
- 📊 Forecasting features
- 🔬 FBLiR model
- ⚙️ Model parameters
- 🗺️ Map controls
- 📡 Data sources
- 🎯 Feature selection
- 🎭 Ensemble methods
- 🔧 Troubleshooting

Try asking:
- "How do I create a forecast?"
- "What is FBLiR?"
- "How do I configure model parameters?"
- "How do I use the map?"

Or ask me anything else about the app!"""
        }
    }
    
    # Check patterns and return appropriate response
    for key, data in knowledge_base.items():
        if key == 'default':
            continue
        for pattern in data['patterns']:
            if pattern in user_lower:
                return data['response']
    
    # Default response
    return knowledge_base['default']['response']

if hasattr(st, "dialog"):

    @st.dialog("🤖 FAIM Assistant", width="large")
    def faim_assistant_dialog():
        st.caption("Tips and how-to for FAIM. Close with the dialog X when you are done.")
        chat_messages_container = st.container(height=420)
        with chat_messages_container:
            for message in st.session_state.chat_history:
                if message["role"] == "assistant":
                    with st.chat_message("assistant", avatar="🤖"):
                        st.markdown(message["content"])
                else:
                    with st.chat_message("user", avatar="👤"):
                        st.markdown(message["content"])

        user_input = st.chat_input("Ask me anything about the app...", key="chat_input_dialog")
        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            st.session_state.chat_history.append(
                {"role": "assistant", "content": get_chatbot_response(user_input)}
            )
            st.rerun()

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🗑️ Clear Chat", use_container_width=True, key="clear_chat_dialog"):
                st.session_state.chat_history = [
                    {
                        "role": "assistant",
                        "content": "👋 Hello! I'm your FAIM Assistant. I can help you learn how to use the app. Ask me anything!",
                    }
                ]
                st.rerun()
        with col_btn2:
            if st.button("💡 Example question", use_container_width=True, key="quick_tips_dialog"):
                tips = [
                    "How do I create a forecast?",
                    "What is FBLiR?",
                    "How do I configure model parameters?",
                    "How do I use the map?",
                ]
                tip = tips[st.session_state.get("tip_index", 0) % len(tips)]
                st.session_state.chat_history.append({"role": "user", "content": tip})
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": get_chatbot_response(tip)}
                )
                st.session_state.tip_index = (st.session_state.get("tip_index", 0) + 1) % len(tips)
                st.rerun()

else:

    def faim_assistant_dialog():
        st.sidebar.warning("Upgrade Streamlit to 1.33+ for the assistant popup.")


# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {
            "role": "assistant",
            "content": "👋 Hello! I'm your FAIM Assistant. I can help you learn how to use the app. Ask me anything!",
        }
    ]

st.sidebar.markdown("---")
if st.sidebar.button("🤖 AI Assistant", use_container_width=True, key="open_ai_assistant"):
    faim_assistant_dialog()

if st.sidebar.button("📖 Guide Helper", use_container_width=True, key="open_guide_helper"):
    faim_howto_dialog()

# Load or create data
@st.cache_data
def load_data(source, start_dt, end_dt):
    if source == "Last 30 Days (Quick)":
        return create_sample_data_for_period(start_dt, end_dt)
    else:
        return None

# Main content — full width (maps first, then visualizations below)
main_viz = st.container()
VIZ_FOLIUM_CELL_W = 720
VIZ_FOLIUM_CELL_H = 440

# Initialize session state for map toggle
if 'show_selection_map' not in st.session_state:
    st.session_state.show_selection_map = True

with main_viz:
    # Map toggle buttons
    col_btn1, col_btn2, col_spacer = st.columns([1, 1, 2])
    with col_btn1:
        if st.button("📍 Selection Map", use_container_width=True, 
                     type="primary" if st.session_state.show_selection_map else "secondary"):
            st.session_state.show_selection_map = True
            st.rerun()
    with col_btn2:
        if st.button("🗺️ Spatial Distribution", use_container_width=True,
                     type="primary" if not st.session_state.show_selection_map else "secondary"):
            st.session_state.show_selection_map = False
            st.rerun()
    
    # Show appropriate map based on toggle
    if st.session_state.show_selection_map:
        st.subheader("📍 Selection Map - Draw Rectangle to Define Area")
    else:
        st.subheader("🗺️ Spatial Distribution - Heatmap Visualization")
        
    # Map type selector
    map_type = st.radio(
        "Map Type",
        ["Street", "Terrain", "Detailed"],
        horizontal=True,
        key="map_type_selector"
    )
        
    # Display Selection Map (for drawing AOI)
    if st.session_state.show_selection_map:
        # Select appropriate tile layer
        if map_type == "Street":
            tiles = "CartoDB positron"
        elif map_type == "Terrain":
            tiles = "OpenStreetMap"
        else:  # Detailed
            tiles = None
            
        # Create base map
        if map_type == "Detailed":
            m = folium.Map(
                location=[41.25, -77.5],
                zoom_start=map_zoom_level
            )
            # Add Esri World Imagery
            folium.TileLayer(
                tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                attr='Esri',
                name='Esri Satellite',
                overlay=False,
                control=True
            ).add_to(m)
            # Add labels
            folium.TileLayer(
                tiles='https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}',
                attr='Esri',
                name='Labels',
                overlay=True,
                control=True
            ).add_to(m)
        else:
            m = folium.Map(
                location=[41.25, -77.5],
                zoom_start=map_zoom_level,
                tiles=tiles
            )
            
        # Add drawing tools
        draw = folium.plugins.Draw(
            export=False,
            position='topleft',
            draw_options={
                'polyline': False,
                'polygon': False,
                'circle': False,
                'marker': False,
                'circlemarker': False,
                'rectangle': True
            }
        )
        draw.add_to(m)
            
        # Display map and capture interactions - LARGER SIZE
        map_data = st_folium(
            m,
            key="main_map",
            width=800,
            height=600,
            returned_objects=["last_object_clicked", "all_drawings"]
        )
            
        # Save map_data to session state for spatial distribution view
        if map_data['all_drawings']:
            st.session_state.last_map_data = map_data
    else:
        # For spatial distribution, we need the map_data from session state
        if 'last_map_data' in st.session_state:
            map_data = st.session_state.last_map_data
        else:
            st.info("👈 Switch to Selection Map to draw an area of interest first")
            map_data = {'all_drawings': []}
    
# Process map interactions (outside of column structure for access everywhere)
processed_bounds = None
gdf_filtered = None

if map_data['all_drawings']:
    latest_drawing = map_data['all_drawings'][-1]
    if latest_drawing['geometry']['type'] == 'Polygon':
        coords = latest_drawing['geometry']['coordinates'][0]

        lons = [coord[0] for coord in coords]
        lats = [coord[1] for coord in coords]

        corrected_lons = []
        for lon in lons:
            while lon > 180:
                lon -= 360
            while lon < -180:
                lon += 360
            corrected_lons.append(lon)

        bounds = [min(corrected_lons), min(lats), max(corrected_lons), max(lats)]

        if not (-90 <= bounds[1] <= 90 and -90 <= bounds[3] <= 90):
            st.error(f"Invalid latitude values: {bounds[1]:.3f} to {bounds[3]:.3f}. Must be between -90° and +90°")
            st.stop()

        if not (-180 <= bounds[0] <= 180 and -180 <= bounds[2] <= 180):
            st.error(f"Invalid longitude values: {bounds[0]:.3f} to {bounds[2]:.3f}. Must be between -180° and +180°")
            st.stop()

        # Store in session state
        st.session_state.aoi_bounds = bounds
        processed_bounds = bounds

        # Show success message and auto-switch to spatial distribution view
        if st.session_state.show_selection_map:
            st.success(f"✅ AOI Selected: {bounds[1]:.3f}°N to {bounds[3]:.3f}°N, {bounds[0]:.3f}°E to {bounds[2]:.3f}°E")
            st.info("🔄 Automatically switching to Spatial Distribution...")
            # Auto-switch to Spatial Distribution view
            time.sleep(1)  # Brief pause so user can see the success message
            st.session_state.show_selection_map = False
            st.rerun()

# If we have stored bounds, use them
if 'aoi_bounds' in st.session_state and processed_bounds is None:
    processed_bounds = st.session_state.aoi_bounds

# Show spatial distribution content when toggled and we have AOI
if not st.session_state.show_selection_map and processed_bounds:
    bounds = processed_bounds

    with main_viz:
        st.info(f"Viewing AOI: {bounds[1]:.3f}°N to {bounds[3]:.3f}°N, {bounds[0]:.3f}°E to {bounds[2]:.3f}°E")
            
        # Create placeholder for spatial distribution maps
        spatial_placeholder = st.empty()
            
        # Load/fetch data based on source
        if data_source == "Last 30 Days (Quick)":
            if st.session_state.gdf_data is None:
                with st.spinner("Loading last 30 days data..."):
                    st.session_state.gdf_data = load_data(data_source, start_date, end_date)
                
            gdf = st.session_state.gdf_data
            mask = (gdf['date'] >= pd.to_datetime(start_date)) & (gdf['date'] <= pd.to_datetime(end_date))
            gdf_filtered = gdf[mask].copy()
                
        elif data_source == "NASA POWER API (Full)":
                with st.spinner("🛰️ Fetching NASA POWER data..."):
                    try:
                        minx, miny, maxx, maxy = bounds
                        lat_range = maxy - miny
                        lon_range = maxx - minx
                        area_size = lat_range * lon_range
                            
                        if area_size < 0.25:
                            grid_points = 5
                            st.info("Small area - using high resolution (5x5 grid)")
                        elif area_size < 1.0:
                            grid_points = 4
                            st.info("Medium area - using medium resolution (4x4 grid)")
                        else:
                            grid_points = 3
                            st.info("Large area - using lower resolution (3x3 grid)")
                            
                        lats = np.linspace(miny, maxy, grid_points)
                        lons = np.linspace(minx, maxx, grid_points)
                            
                        all_data = []
                        total_points = len(lats) * len(lons)
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                            
                        for i, lat in enumerate(lats):
                            for j, lon in enumerate(lons):
                                current_point = i * len(lons) + j + 1
                                progress_bar.progress(current_point / total_points)
                                status_text.text(f"Fetching data for point {current_point}/{total_points} ({lat:.3f}°, {lon:.3f}°)")
                                    
                                point_data = fetch_nasa_power_data(
                                    lat, lon, start_date, end_date, 
                                    list(AVAILABLE_PARAMETERS.keys())
                                )
                                    
                                if not point_data.empty:
                                    for idx, row in point_data.iterrows():
                                        fwi = calculate_robust_fwi(
                                            row.get('T2M'), 
                                            row.get('RH2M'), 
                                            row.get('WS10M'), 
                                            row.get('PRECTOT')
                                        )
                                        point_data.at[idx, 'FWI'] = fwi
                                            
                                        afdr = calculate_afdr(
                                            row.get('T2M'),
                                            row.get('RH2M'),
                                            row.get('WS10M'),
                                            row.get('PRECTOT')
                                        )
                                        point_data.at[idx, 'AFDR'] = afdr
                                        
                                    all_data.append(point_data)
                                    
                                time.sleep(0.2)
                            
                        progress_bar.empty()
                        status_text.empty()
                            
                        if all_data:
                            combined_df = pd.concat(all_data, ignore_index=True)
                            gdf_filtered = gpd.GeoDataFrame(
                                combined_df, 
                                geometry=gpd.points_from_xy(combined_df.lon, combined_df.lat), 
                                crs="EPSG:4326"
                            )
                                
                            st.success(f"✅ Successfully fetched NASA POWER data for {len(gdf_filtered)} data points!")
                                
                            valid_data_count = gdf_filtered[selected_metric].notna().sum()
                            st.info(f"📊 Data quality: {valid_data_count}/{len(gdf_filtered)} records have valid {selected_metric} data")
                        else:
                            st.error("❌ Failed to fetch NASA POWER data. Please try again or use sample data.")
                            gdf_filtered = None
                                
                    except Exception as e:
                        st.error(f"❌ Error fetching NASA POWER data: {str(e)}")
                        st.info("💡 Please try using Sample Data instead, or check your internet connection.")
                        gdf_filtered = None
            
        else:  # Forecasting mode
                with st.spinner("🛰️ Fetching NASA POWER data for forecasting..."):
                    try:
                        minx, miny, maxx, maxy = bounds
                        lat_range = maxy - miny
                        lon_range = maxx - minx
                        area_size = lat_range * lon_range
                            
                        if area_size < 0.25:
                            grid_points = 5
                        elif area_size < 1.0:
                            grid_points = 4
                        else:
                            grid_points = 3
                            
                        lats = np.linspace(miny, maxy, grid_points)
                        lons = np.linspace(minx, maxx, grid_points)
                            
                        all_data = []
                        total_points = len(lats) * len(lons)
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                            
                        for i, lat in enumerate(lats):
                            for j, lon in enumerate(lons):
                                current_point = i * len(lons) + j + 1
                                progress_bar.progress(current_point / total_points)
                                status_text.text(f"Fetching data {current_point}/{total_points}")
                                    
                                point_data = fetch_nasa_power_data(
                                    lat, lon, start_date, end_date, 
                                    list(AVAILABLE_PARAMETERS.keys())
                                )
                                    
                                if not point_data.empty:
                                    for idx, row in point_data.iterrows():
                                        fwi = calculate_robust_fwi(
                                            row.get('T2M'), 
                                            row.get('RH2M'), 
                                            row.get('WS10M'), 
                                            row.get('PRECTOT')
                                        )
                                        point_data.at[idx, 'FWI'] = fwi
                                            
                                        afdr = calculate_afdr(
                                            row.get('T2M'),
                                            row.get('RH2M'),
                                            row.get('WS10M'),
                                            row.get('PRECTOT')
                                        )
                                        point_data.at[idx, 'AFDR'] = afdr
                                        
                                    all_data.append(point_data)
                                    
                                time.sleep(0.2)
                            
                        progress_bar.empty()
                        status_text.empty()
                            
                        if all_data:
                            combined_df = pd.concat(all_data, ignore_index=True)
                            gdf_filtered = gpd.GeoDataFrame(
                                combined_df, 
                                geometry=gpd.points_from_xy(combined_df.lon, combined_df.lat), 
                                crs="EPSG:4326"
                            )
                                
                            st.success(f"✅ Successfully fetched {len(gdf_filtered)} historical records")
                        else:
                            st.error("❌ Failed to fetch data")
                            gdf_filtered = None
                                
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        gdf_filtered = None
            
        if gdf_filtered is not None and not gdf_filtered.empty:
                    st.subheader("📊 Visualization")
                        
                    if forecast_mode:
                        viz_mode = st.radio(
                            "Mode",
                            ["Time Series", "Forecast Animation"]
                        )
                    else:
                        viz_mode = st.radio(
                            "Mode",
                            ["Single Date", "Animation", "Time Series"]
                        )
                        
                    if viz_mode == "Single Date":
                        available_dates = sorted(gdf_filtered['date'].unique())
                        selected_date = st.selectbox(
                            "Select Date",
                            options=available_dates,
                            format_func=lambda x: x.strftime("%Y-%m-%d")
                        )
                            
                        gdf_date = gdf_filtered[gdf_filtered['date'] == selected_date]
                            
                        if not gdf_date.empty:
                            # FIXED: Use continuous heatmap with rectangles
                            ref_sd = float(gdf_date[selected_metric].dropna().astype(float).std())
                            if not np.isfinite(ref_sd):
                                ref_sd = 0.0
                            values_grid, values = create_heatmap_data(
                                gdf_date, bounds, selected_metric, reference_std=ref_sd
                            )
                                
                            if values_grid is not None and values:
                                vmin, vmax = min(values), max(values)
                                metric_description = AVAILABLE_PARAMETERS.get(
                                    selected_metric,
                                    "Fire Weather Index" if selected_metric == "FWI" else "Australian Fire Danger Rating"
                                )
                                with spatial_placeholder.container():
                                    s1, s2 = st.columns(2, gap="large")
                                    with s1:
                                        st.subheader("Spatial — selected date")
                                        if map_type == "Detailed":
                                            m_single = folium.Map(
                                                location=[(bounds[1] + bounds[3])/2, (bounds[0] + bounds[2])/2],
                                                zoom_start=map_zoom_level
                                            )
                                            folium.TileLayer(
                                                tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                                                attr='Esri',
                                                name='Esri Satellite'
                                            ).add_to(m_single)
                                            folium.TileLayer(
                                                tiles='https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}',
                                                attr='Esri',
                                                name='Labels',
                                                overlay=True
                                            ).add_to(m_single)
                                        elif map_type == "Terrain":
                                            m_single = folium.Map(
                                                location=[(bounds[1] + bounds[3])/2, (bounds[0] + bounds[2])/2],
                                                zoom_start=map_zoom_level,
                                                tiles="OpenStreetMap"
                                            )
                                        else:
                                            m_single = folium.Map(
                                                location=[(bounds[1] + bounds[3])/2, (bounds[0] + bounds[2])/2],
                                                zoom_start=map_zoom_level,
                                                tiles="CartoDB positron"
                                            )
                                        folium.Rectangle(
                                            bounds=[[bounds[1], bounds[0]], [bounds[3], bounds[2]]],
                                            color="red",
                                            weight=2,
                                            fill_opacity=0.0,
                                            popup=f"Selected date: {selected_date.strftime('%Y-%m-%d')}"
                                        ).add_to(m_single)
                                        metric_map = AVAILABLE_PARAMETERS.get(selected_metric, "Fire Weather Index" if selected_metric == "FWI" else "Australian Fire Danger Rating" if selected_metric == "AFDR" else selected_metric)
                                        heatmap_layer, legend_html = create_continuous_heatmap(
                                            bounds, values_grid,
                                            opacity=heatmap_opacity,
                                            metric_name=metric_map
                                        )
                                        heatmap_layer.add_to(m_single)
                                        m_single.get_root().html.add_child(folium.Element(legend_html))
                                        st_folium(
                                            m_single,
                                            key=f"single_date_map_{selected_date}",
                                            width=VIZ_FOLIUM_CELL_W,
                                            height=VIZ_FOLIUM_CELL_H,
                                            returned_objects=[],
                                        )
                                    with s2:
                                        st.subheader("Distribution")
                                        dist_fig = create_distribution_plot(values, selected_metric, metric_description)
                                        if dist_fig:
                                            dist_fig.update_layout(height=480, margin=dict(t=50, b=40))
                                            st.plotly_chart(dist_fig, use_container_width=True)
                                    summary_sd = [{
                                        "Date": selected_date.strftime("%Y-%m-%d"),
                                        "Metric": selected_metric,
                                        "Min": float(vmin),
                                        "Mean": float(np.mean(values)),
                                        "Max": float(vmax),
                                        "Std": float(np.std(values)),
                                        "Median": float(np.median(values)),
                                        "P25": float(np.percentile(values, 25)),
                                        "P75": float(np.percentile(values, 75)),
                                    }]
                                    if selected_metric == "AFDR":
                                        avg_afdr = float(np.mean(values))
                                        category, emoji = get_afdr_category(avg_afdr)
                                        st.info(f"{emoji} **AFDR category:** {category}")
                                    st.subheader("Summary statistics")
                                    st.dataframe(
                                        pd.DataFrame(summary_sd).round(3),
                                        use_container_width=True,
                                        hide_index=True,
                                    )
                        
                    elif viz_mode == "Animation":
                        available_dates = sorted(gdf_filtered['date'].unique())
                        anim_ref_std = float(gdf_filtered[selected_metric].dropna().astype(float).std())
                        if not np.isfinite(anim_ref_std):
                            anim_ref_std = 0.0
                            
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("▶️ Start Animation", key="start_historical_anim"):
                                st.session_state.animation_running = True
                                st.session_state.current_frame = 0
                                st.rerun()
                            
                        with col2:
                            if st.button("⏸️ Stop Animation", key="stop_historical_anim"):
                                st.session_state.animation_running = False
                                st.rerun()
                            
                        # Initialize animation state
                        if 'current_frame' not in st.session_state:
                            st.session_state.current_frame = 0
                            
                        # Animation loop
                        if st.session_state.animation_running and available_dates:
                            # Get current date
                            current_date = available_dates[st.session_state.current_frame]
                                
                            # Progress indicator
                            progress = (st.session_state.current_frame + 1) / len(available_dates)
                            st.progress(progress)
                            st.write(f"**Showing:** {current_date.strftime('%Y-%m-%d')} (Frame {st.session_state.current_frame + 1}/{len(available_dates)})")
                                
                            # Filter data for current date
                            gdf_date = gdf_filtered[gdf_filtered['date'] == current_date]
                                
                            if not gdf_date.empty:
                                values_grid, values = create_heatmap_data(
                                    gdf_date, bounds, selected_metric, reference_std=anim_ref_std
                                )
                                    
                                if values_grid is not None and values:
                                    # Create animation map with spatial heatmap
                                    with spatial_placeholder.container():
                                        if map_type == "Detailed":
                                            m_anim = folium.Map(
                                                location=[(bounds[1] + bounds[3])/2, (bounds[0] + bounds[2])/2],
                                                zoom_start=map_zoom_level
                                            )
                                            folium.TileLayer(
                                                tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                                                attr='Esri',
                                                name='Esri Satellite'
                                            ).add_to(m_anim)
                                            folium.TileLayer(
                                                tiles='https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}',
                                                attr='Esri',
                                                name='Labels',
                                                overlay=True
                                            ).add_to(m_anim)
                                        elif map_type == "Terrain":
                                            m_anim = folium.Map(
                                                location=[(bounds[1] + bounds[3])/2, (bounds[0] + bounds[2])/2],
                                                zoom_start=map_zoom_level,
                                                tiles="OpenStreetMap"
                                            )
                                        else:
                                            m_anim = folium.Map(
                                                location=[(bounds[1] + bounds[3])/2, (bounds[0] + bounds[2])/2],
                                                zoom_start=map_zoom_level,
                                                tiles="CartoDB positron"
                                            )
                                            
                                        # Add AOI rectangle
                                        folium.Rectangle(
                                            bounds=[[bounds[1], bounds[0]], [bounds[3], bounds[2]]],
                                            color="red",
                                            weight=2,
                                            fill_opacity=0.0,
                                            popup=f"Animation: {current_date.strftime('%Y-%m-%d')}"
                                        ).add_to(m_anim)
                                            
                                        # Add continuous heatmap
                                        metric_description = AVAILABLE_PARAMETERS.get(selected_metric, "Fire Weather Index" if selected_metric == "FWI" else "Australian Fire Danger Rating" if selected_metric == "AFDR" else selected_metric)
                                        heatmap_layer, legend_html = create_continuous_heatmap(
                                            bounds, values_grid,
                                            opacity=heatmap_opacity,
                                            metric_name=f"{metric_description} - {current_date.strftime('%Y-%m-%d')}"
                                        )
                                        heatmap_layer.add_to(m_anim)
                                        m_anim.get_root().html.add_child(folium.Element(legend_html))
                                            
                                        # Display the animation map
                                        st_folium(m_anim, key=f"animation_map_{st.session_state.current_frame}", 
                                                 width=800, height=600, returned_objects=[])
                                        
                                    # Show statistics for current frame
                                    vmin, vmax = min(values), max(values)
                                    st.subheader("📈 Current Frame Statistics")
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Min", f"{vmin:.2f}")
                                    with col2:
                                        st.metric("Mean", f"{np.mean(values):.2f}")
                                    with col3:
                                        st.metric("Max", f"{vmax:.2f}")
                                        
                                    # Add distribution for current frame
                                    with st.expander(f"📊 Distribution for {current_date.strftime('%Y-%m-%d')}", expanded=False):
                                        metric_description = AVAILABLE_PARAMETERS.get(selected_metric, "Fire Weather Index" if selected_metric == "FWI" else "Australian Fire Danger Rating" if selected_metric == "AFDR" else selected_metric)
                                        dist_fig = create_distribution_plot(values, selected_metric, metric_description)
                                        if dist_fig:
                                            st.plotly_chart(dist_fig, use_container_width=True)
                                
                            # Auto-advance frame
                            time.sleep(animation_speed)
                            st.session_state.current_frame = (st.session_state.current_frame + 1) % len(available_dates)
                                
                            # Stop animation when we complete one cycle
                            if st.session_state.current_frame == 0:
                                st.session_state.animation_running = False
                                st.success("Animation cycle complete!")
                                
                            # Rerun to show next frame
                            if st.session_state.animation_running:
                                st.rerun()
                            
                        elif not st.session_state.get('animation_running', False) and available_dates:
                            # Show static frame when not animating
                            current_date = available_dates[st.session_state.get('current_frame', 0)]
                            st.write(f"**Static view:** {current_date.strftime('%Y-%m-%d')}")
                                
                            gdf_date = gdf_filtered[gdf_filtered['date'] == current_date]
                            if not gdf_date.empty:
                                values_grid, values = create_heatmap_data(
                                    gdf_date, bounds, selected_metric, reference_std=anim_ref_std
                                )
                                    
                                if values_grid is not None and values:
                                    with spatial_placeholder.container():
                                        if map_type == "Detailed":
                                            m_static = folium.Map(
                                                location=[(bounds[1] + bounds[3])/2, (bounds[0] + bounds[2])/2],
                                                zoom_start=map_zoom_level
                                            )
                                            folium.TileLayer(
                                                tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                                                attr='Esri',
                                                name='Esri Satellite'
                                            ).add_to(m_static)
                                            folium.TileLayer(
                                                tiles='https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}',
                                                attr='Esri',
                                                name='Labels',
                                                overlay=True
                                            ).add_to(m_static)
                                        elif map_type == "Terrain":
                                            m_static = folium.Map(
                                                location=[(bounds[1] + bounds[3])/2, (bounds[0] + bounds[2])/2],
                                                zoom_start=map_zoom_level,
                                                tiles="OpenStreetMap"
                                            )
                                        else:
                                            m_static = folium.Map(
                                                location=[(bounds[1] + bounds[3])/2, (bounds[0] + bounds[2])/2],
                                                zoom_start=map_zoom_level,
                                                tiles="CartoDB positron"
                                            )
                                            
                                        folium.Rectangle(
                                            bounds=[[bounds[1], bounds[0]], [bounds[3], bounds[2]]],
                                            color="red",
                                            weight=2,
                                            fill_opacity=0.0
                                        ).add_to(m_static)
                                            
                                        metric_description = AVAILABLE_PARAMETERS.get(selected_metric, "Fire Weather Index" if selected_metric == "FWI" else "Australian Fire Danger Rating" if selected_metric == "AFDR" else selected_metric)
                                        heatmap_layer, legend_html = create_continuous_heatmap(
                                            bounds, values_grid,
                                            opacity=heatmap_opacity,
                                            metric_name=metric_description
                                        )
                                        heatmap_layer.add_to(m_static)
                                        m_static.get_root().html.add_child(folium.Element(legend_html))
                                            
                                        st_folium(m_static, key="static_animation_map", 
                                                 width=800, height=600, returned_objects=[])
                        
                    elif viz_mode == "Time Series":
                        mask = (
                            (gdf_filtered.geometry.x >= bounds[0]) & (gdf_filtered.geometry.x <= bounds[2]) &
                            (gdf_filtered.geometry.y >= bounds[1]) & (gdf_filtered.geometry.y <= bounds[3])
                        )
                        gdf_aoi = gdf_filtered[mask]
                            
                        if not gdf_aoi.empty:
                            if forecast_mode and selected_features:
                                st.info("🔮 Generating forecast... This may take a moment.")
                                    
                                # Prepare data for forecasting
                                daily_avg = gdf_aoi.groupby('date').agg({
                                    forecast_target: 'mean',
                                    **{feat: 'mean' for feat in selected_features if feat in gdf_aoi.columns}
                                }).reset_index()
                                    
                                daily_avg = daily_avg.dropna(subset=[forecast_target])
                                    
                                if len(daily_avg) < 30:
                                    st.error("⚠️ Insufficient historical data. Need at least 30 days.")
                                else:
                                    # FIXED: Prepare features with better NaN handling
                                    df_ml, feature_names = prepare_ml_features(
                                        daily_avg, 
                                        forecast_target,
                                        selected_features,
                                        lag_days=7
                                    )
                                        
                                    if len(df_ml) < 20:
                                        st.error(f"⚠️ Not enough data after feature engineering. Got {len(df_ml)} rows, need at least 20.")
                                        st.info("💡 Try: (1) Using fewer features, (2) Longer historical period, or (3) Different forecast target")
                                    else:
                                        X = df_ml[feature_names]
                                        y = df_ml[forecast_target]
                                            
                                        scaler = StandardScaler()
                                        X_scaled = pd.DataFrame(
                                            scaler.fit_transform(X),
                                            columns=X.columns,
                                            index=X.index
                                        )
                                            
                                        # Generate future dates
                                        last_date = daily_avg['date'].max()
                                        future_dates = pd.date_range(
                                            start=last_date + timedelta(days=1),
                                            periods=forecast_horizon,
                                            freq='D'
                                        )
                                            
                                        # Create future features iteratively
                                        st.session_state.forecast_insights_md = None
                                        forecast_results = {}
    
                                        for method in forecast_methods:
                                            if method == "Ensemble":
                                                continue
    
                                            try:
                                                if method == "Prophet":
                                                    df_prophet = daily_avg[['date', forecast_target]].copy()
                                                    df_prophet.columns = ['ds', 'y']
                                                    predictions = train_prophet(df_prophet, forecast_horizon, model_params.get('Prophet'))
    
                                                elif method == "SARIMA":
                                                    predictions = train_sarima(y, forecast_horizon, model_params.get('SARIMA'))
    
                                                elif method == "LLM Forecaster":
                                                    predictions = train_llm_horizon_forecast(
                                                        daily_avg,
                                                        forecast_target,
                                                        forecast_horizon,
                                                        future_dates,
                                                        model_params.get("LLM Forecaster"),
                                                    )
    
                                                else:
                                                    fblir_status = None
                                                    if method == "FBLiR":
                                                        fblir_status = st.empty()
                                                        fblir_status.info("🔄 FBLiR is training (single fit for the full horizon)...")
                                                    try:
                                                        predictions = run_iterative_ml_forecast(
                                                            method,
                                                            X_scaled,
                                                            y,
                                                            feature_names,
                                                            df_ml,
                                                            selected_features,
                                                            forecast_target,
                                                            forecast_horizon,
                                                            scaler,
                                                            model_params,
                                                        )
                                                    finally:
                                                        if fblir_status is not None:
                                                            fblir_status.success("✅ FBLiR completed.")
                                                            time.sleep(0.25)
                                                            fblir_status.empty()
    
                                                forecast_results[method] = predictions
    
                                            except Exception as e:
                                                st.warning(f"⚠️ {method} failed: {str(e)}")
    
                                        # Calculate ensemble if requested
                                        if "Ensemble" in forecast_methods and len(forecast_results) > 0:
                                            ensemble_pred = np.mean(list(forecast_results.values()), axis=0)
                                            forecast_results["Ensemble"] = ensemble_pred
                                            
                                        if forecast_results:
                                            st.session_state.forecast_insights_md = generate_forecast_insights_markdown(
                                                forecast_results,
                                                daily_avg,
                                                forecast_target,
                                                future_dates,
                                            )
                                        else:
                                            st.session_state.forecast_insights_md = None
                                            
                                        if forecast_results:
                                            # Display forecast visualization in spatial_placeholder (col_map)
                                            with spatial_placeholder.container():
                                                _ins = st.session_state.get("forecast_insights_md")
                                                if _ins:
                                                    st.subheader("💡 Useful insights")
                                                    st.markdown(_ins)
                                                    st.markdown("---")
                                                y_hist = daily_avg[forecast_target].dropna().astype(float)
                                                acf_fig_fc = None
                                                if len(y_hist) > 14:
                                                    acf_fig_fc = _acf_plotly(
                                                        y_hist.values,
                                                        title=f"ACF — {forecast_target} (historical AOI daily mean)",
                                                        max_lag=min(40, max(5, len(y_hist) // 3)),
                                                    )
                                                metric_description_hist = AVAILABLE_PARAMETERS.get(
                                                    forecast_target,
                                                    "Fire Weather Index" if forecast_target == "FWI" else "Australian Fire Danger Rating" if forecast_target == "AFDR" else forecast_target,
                                                )

                                                row1c1, row1c2 = st.columns(2, gap="large")
                                                with row1c1:
                                                    st.subheader("Forecast time series")
                                                    fig = go.Figure()
                                                    fig.add_trace(go.Scatter(
                                                        x=daily_avg['date'],
                                                        y=daily_avg[forecast_target],
                                                        mode='lines',
                                                        name='Historical',
                                                        line=dict(color='blue', width=2),
                                                    ))
                                                    colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan']
                                                    for idx, (method, predictions) in enumerate(forecast_results.items()):
                                                        fig.add_trace(go.Scatter(
                                                            x=future_dates,
                                                            y=predictions,
                                                            mode='lines',
                                                            name=f'{method} forecast',
                                                            line=dict(color=colors[idx % len(colors)], width=2, dash='dash'),
                                                        ))
                                                    fig.update_layout(
                                                        title=f"{forecast_target} — historical + forecasts",
                                                        xaxis_title="Date",
                                                        yaxis_title=AVAILABLE_PARAMETERS.get(forecast_target, forecast_target),
                                                        height=480,
                                                        margin=dict(t=50, b=40),
                                                        hovermode='x unified',
                                                    )
                                                    st.plotly_chart(fig, use_container_width=True)

                                                with row1c2:
                                                    st.subheader("Spatial — latest historical")
                                                    mask_aoi = (
                                                        (gdf_filtered.geometry.x >= bounds[0]) & (gdf_filtered.geometry.x <= bounds[2]) &
                                                        (gdf_filtered.geometry.y >= bounds[1]) & (gdf_filtered.geometry.y <= bounds[3])
                                                    )
                                                    gdf_spatial_data = gdf_filtered[mask_aoi]
                                                    if not gdf_spatial_data.empty:
                                                        valid_data = gdf_spatial_data[gdf_spatial_data[forecast_target].notna()]
                                                        if not valid_data.empty:
                                                            latest_date = valid_data['date'].max()
                                                            gdf_latest = valid_data[valid_data['date'] == latest_date]
                                                            hist_std = float(y_hist.std()) if len(y_hist) > 1 else 0.0
                                                            if not np.isfinite(hist_std):
                                                                hist_std = 0.0
                                                            values_grid, heatmap_values = create_heatmap_data(
                                                                gdf_latest,
                                                                bounds,
                                                                forecast_target,
                                                                reference_std=hist_std,
                                                            )
                                                            if values_grid is not None and heatmap_values and len(heatmap_values) > 0:
                                                                if map_type == "Detailed":
                                                                    m_forecast_spatial = folium.Map(
                                                                        location=[(bounds[1] + bounds[3])/2, (bounds[0] + bounds[2])/2],
                                                                        zoom_start=map_zoom_level
                                                                    )
                                                                    folium.TileLayer(
                                                                        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                                                                        attr='Esri',
                                                                        name='Esri Satellite'
                                                                    ).add_to(m_forecast_spatial)
                                                                    folium.TileLayer(
                                                                        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}',
                                                                        attr='Esri',
                                                                        name='Labels',
                                                                        overlay=True
                                                                    ).add_to(m_forecast_spatial)
                                                                elif map_type == "Terrain":
                                                                    m_forecast_spatial = folium.Map(
                                                                        location=[(bounds[1] + bounds[3])/2, (bounds[0] + bounds[2])/2],
                                                                        zoom_start=map_zoom_level,
                                                                        tiles="OpenStreetMap"
                                                                    )
                                                                else:
                                                                    m_forecast_spatial = folium.Map(
                                                                        location=[(bounds[1] + bounds[3])/2, (bounds[0] + bounds[2])/2],
                                                                        zoom_start=map_zoom_level,
                                                                        tiles="CartoDB positron"
                                                                    )
                                                                folium.Rectangle(
                                                                    bounds=[[bounds[1], bounds[0]], [bounds[3], bounds[2]]],
                                                                    color="red",
                                                                    weight=2,
                                                                    fill_opacity=0.0,
                                                                    popup=f"Latest historical: {latest_date.strftime('%Y-%m-%d')}"
                                                                ).add_to(m_forecast_spatial)
                                                                heatmap_layer, legend_html = create_continuous_heatmap(
                                                                    bounds, values_grid,
                                                                    opacity=heatmap_opacity,
                                                                    metric_name=f"{metric_description_hist} — {latest_date.strftime('%Y-%m-%d')}"
                                                                )
                                                                heatmap_layer.add_to(m_forecast_spatial)
                                                                m_forecast_spatial.get_root().html.add_child(folium.Element(legend_html))
                                                                st_folium(
                                                                    m_forecast_spatial,
                                                                    key="forecast_timeseries_spatial_map",
                                                                    width=VIZ_FOLIUM_CELL_W,
                                                                    height=VIZ_FOLIUM_CELL_H,
                                                                    returned_objects=[],
                                                                )
                                                                st.caption(f"Latest: {latest_date.strftime('%Y-%m-%d')} · {len(gdf_latest)} points")
                                                            else:
                                                                st.info("Insufficient spatial points for this date.")
                                                        else:
                                                            st.info(f"No valid {forecast_target} in AOI for mapping.")
                                                    else:
                                                        st.info("No spatial data in AOI.")

                                                row2c1, row2c2 = st.columns(2, gap="large")
                                                with row2c1:
                                                    st.subheader("Historical ACF")
                                                    if acf_fig_fc is not None:
                                                        st.plotly_chart(acf_fig_fc, use_container_width=True)
                                                    else:
                                                        st.info("Not enough historical points for ACF (need more than 14 days).")
                                                with row2c2:
                                                    st.subheader("Historical distribution")
                                                    dist_fig_fc = create_distribution_plot(
                                                        y_hist.tolist(), forecast_target, metric_description_hist
                                                    )
                                                    if dist_fig_fc:
                                                        dist_fig_fc.update_layout(
                                                            title=f"{forecast_target} — AOI daily mean",
                                                            height=420,
                                                            margin=dict(t=50, b=40),
                                                        )
                                                        st.plotly_chart(dist_fig_fc, use_container_width=True)
                                                    else:
                                                        st.info("No distribution to show.")

                                                summary_rows_fc = []
                                                if len(y_hist):
                                                    summary_rows_fc.append({
                                                        "Series": "Historical (AOI daily mean)",
                                                        "Mean": float(y_hist.mean()),
                                                        "Min": float(y_hist.min()),
                                                        "Max": float(y_hist.max()),
                                                        "Std": float(y_hist.std()),
                                                    })
                                                for method, predictions in forecast_results.items():
                                                    arr = np.asarray(predictions, dtype=float)
                                                    summary_rows_fc.append({
                                                        "Series": f"Forecast — {method}",
                                                        "Mean": float(np.mean(arr)),
                                                        "Min": float(np.min(arr)),
                                                        "Max": float(np.max(arr)),
                                                        "Std": float(np.std(arr)),
                                                    })
                                                st.subheader("Summary statistics")
                                                st.dataframe(
                                                    pd.DataFrame(summary_rows_fc).round(3),
                                                    use_container_width=True,
                                                    hide_index=True,
                                                )
                                            # Prepare export data
                                            export_df = daily_avg[['date', forecast_target]].copy()
                                            export_df['type'] = 'historical'
                                                
                                            for method, predictions in forecast_results.items():
                                                method_df = pd.DataFrame({
                                                    'date': future_dates,
                                                    forecast_target: predictions,
                                                    'type': f'forecast_{method}'
                                                })
                                                export_df = pd.concat([export_df, method_df], ignore_index=True)
                                                
                                            st.session_state.combined_export_data = export_df
                                                
                                            # Store forecast frames for animation
                                            st.session_state.forecast_frames = {
                                                'dates': future_dates,
                                                'forecasts': forecast_results,
                                                'bounds': bounds
                                            }
                                            
                                        else:
                                            st.error("❌ All forecast methods failed")
                                
                            else:
                                # Regular time series (no forecast)
                                daily_avg = gdf_aoi.groupby('date')[selected_metric].mean().reset_index()
                                latest_date = gdf_aoi['date'].max()
                                gdf_latest = gdf_aoi[gdf_aoi['date'] == latest_date]
                                values_grid = None
                                heatmap_values = None
                                if not gdf_latest.empty:
                                    ts_ref_std = float(daily_avg[selected_metric].std())
                                    if not np.isfinite(ts_ref_std):
                                        ts_ref_std = 0.0
                                    values_grid, heatmap_values = create_heatmap_data(
                                        gdf_latest,
                                        bounds,
                                        selected_metric,
                                        reference_std=ts_ref_std,
                                    )

                                with spatial_placeholder.container():
                                    ts1a, ts1b = st.columns(2, gap="large")
                                    with ts1a:
                                        st.subheader("Time series")
                                        fig = px.line(
                                            daily_avg,
                                            x='date',
                                            y=selected_metric,
                                            title=f"{selected_metric} — AOI daily mean",
                                            labels={'date': 'Date', selected_metric: AVAILABLE_PARAMETERS.get(selected_metric, selected_metric)}
                                        )
                                        fig.update_layout(height=480, margin=dict(t=50, b=40))
                                        st.plotly_chart(fig, use_container_width=True)
                                    with ts1b:
                                        st.subheader("Spatial — latest day")
                                        if values_grid is not None and heatmap_values:
                                            if map_type == "Detailed":
                                                m_timeseries = folium.Map(
                                                    location=[(bounds[1] + bounds[3])/2, (bounds[0] + bounds[2])/2],
                                                    zoom_start=map_zoom_level
                                                )
                                                folium.TileLayer(
                                                    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                                                    attr='Esri',
                                                    name='Esri Satellite'
                                                ).add_to(m_timeseries)
                                                folium.TileLayer(
                                                    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}',
                                                    attr='Esri',
                                                    name='Labels',
                                                    overlay=True
                                                ).add_to(m_timeseries)
                                            elif map_type == "Terrain":
                                                m_timeseries = folium.Map(
                                                    location=[(bounds[1] + bounds[3])/2, (bounds[0] + bounds[2])/2],
                                                    zoom_start=map_zoom_level,
                                                    tiles="OpenStreetMap"
                                                )
                                            else:
                                                m_timeseries = folium.Map(
                                                    location=[(bounds[1] + bounds[3])/2, (bounds[0] + bounds[2])/2],
                                                    zoom_start=map_zoom_level,
                                                    tiles="CartoDB positron"
                                                )
                                            folium.Rectangle(
                                                bounds=[[bounds[1], bounds[0]], [bounds[3], bounds[2]]],
                                                color="red",
                                                weight=2,
                                                fill_opacity=0.0,
                                                popup=f"Latest: {latest_date.strftime('%Y-%m-%d')}"
                                            ).add_to(m_timeseries)
                                            metric_description = AVAILABLE_PARAMETERS.get(selected_metric, "Fire Weather Index" if selected_metric == "FWI" else "Australian Fire Danger Rating" if selected_metric == "AFDR" else selected_metric)
                                            heatmap_layer, legend_html = create_continuous_heatmap(
                                                bounds, values_grid,
                                                opacity=heatmap_opacity,
                                                metric_name=metric_description
                                            )
                                            heatmap_layer.add_to(m_timeseries)
                                            m_timeseries.get_root().html.add_child(folium.Element(legend_html))
                                            st_folium(
                                                m_timeseries,
                                                key="timeseries_spatial_map",
                                                width=VIZ_FOLIUM_CELL_W,
                                                height=VIZ_FOLIUM_CELL_H,
                                                returned_objects=[],
                                            )
                                            st.caption(f"Latest date: {latest_date.strftime('%Y-%m-%d')}")
                                        else:
                                            st.info("No spatial heatmap available for the latest date.")

                                    y_ts = daily_avg[selected_metric].dropna().astype(float)
                                    acf_ts = None
                                    if len(y_ts) > 14:
                                        acf_ts = _acf_plotly(
                                            y_ts.values,
                                            title=f"ACF — {selected_metric} (AOI daily mean)",
                                            max_lag=min(40, max(5, len(y_ts) // 3)),
                                        )
                                    metric_description = AVAILABLE_PARAMETERS.get(
                                        selected_metric,
                                        "Fire Weather Index" if selected_metric == "FWI" else "Australian Fire Danger Rating"
                                    )
                                    ts2a, ts2b = st.columns(2, gap="large")
                                    with ts2a:
                                        st.subheader("ACF")
                                        if acf_ts is not None:
                                            st.plotly_chart(acf_ts, use_container_width=True)
                                        else:
                                            st.info("Not enough historical points for ACF (need more than 14 days).")
                                    with ts2b:
                                        st.subheader("Distribution")
                                        time_series_values = daily_avg[selected_metric].dropna().tolist()
                                        dist_fig = create_distribution_plot(time_series_values, selected_metric, metric_description)
                                        if dist_fig:
                                            dist_fig.update_layout(
                                                title=f"{selected_metric} — AOI daily mean",
                                                height=420,
                                                margin=dict(t=50, b=40),
                                            )
                                            st.plotly_chart(dist_fig, use_container_width=True)
                                        else:
                                            st.info("No distribution to show.")

                                    time_series_values = daily_avg[selected_metric].dropna().tolist()
                                    if time_series_values:
                                        summary_ts = [{
                                            "Series": f"{selected_metric} (AOI daily mean)",
                                            "Mean": float(np.mean(time_series_values)),
                                            "Std": float(np.std(time_series_values)),
                                            "Min": float(np.min(time_series_values)),
                                            "Max": float(np.max(time_series_values)),
                                            "Median": float(np.median(time_series_values)),
                                            "P25": float(np.percentile(time_series_values, 25)),
                                            "P75": float(np.percentile(time_series_values, 75)),
                                        }]
                                        if selected_metric == "AFDR":
                                            avg_afdr = float(np.mean(time_series_values))
                                            category, emoji = get_afdr_category(avg_afdr)
                                            st.info(f"{emoji} **Average AFDR category:** {category}")
                                        st.subheader("Summary statistics")
                                        st.dataframe(
                                            pd.DataFrame(summary_ts).round(3),
                                            use_container_width=True,
                                            hide_index=True,
                                        )
                    elif viz_mode == "Forecast Animation":
                        if st.session_state.forecast_frames is None:
                            st.info("👆 Please view Time Series first to generate forecast")
                        else:
                            frames_data = st.session_state.forecast_frames
                            future_dates = frames_data['dates']
                            forecast_results = frames_data['forecasts']
                            bounds = frames_data['bounds']
                                
                            # Select forecast method to animate
                            selected_forecast_method = st.selectbox(
                                "Select Forecast Method",
                                options=list(forecast_results.keys()),
                                key="forecast_animation_method"
                            )
                                
                            predictions = forecast_results[selected_forecast_method]
                                
                            # Animation controls
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button("▶️ Start Forecast", key="start_forecast_anim"):
                                    st.session_state.animation_running = True
                                    st.session_state.current_frame = 0
                                    st.rerun()
                                
                            with col2:
                                if st.button("⏸️ Pause", key="pause_forecast_anim"):
                                    st.session_state.animation_running = False
                                    st.rerun()
                                
                            # Frame selector
                            current_frame = st.slider(
                                "Select Forecast Day",
                                min_value=0,
                                max_value=len(predictions) - 1,
                                value=st.session_state.get('current_frame', 0),
                                key="frame_slider"
                            )
                                
                            st.session_state.current_frame = current_frame
                                
                            # Display current forecast
                            current_date = future_dates[current_frame]
                            current_value = predictions[current_frame]
                                
                            st.write(f"**Forecast Date:** {current_date.strftime('%Y-%m-%d')}")
                            st.write(f"**Predicted {forecast_target}:** {current_value:.2f}")
                                
                            # Statistics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Current Value", f"{current_value:.2f}")
                            with col2:
                                if current_frame > 0:
                                    prev_value = predictions[current_frame - 1]
                                    delta = current_value - prev_value
                                    st.metric("Change from Previous", f"{delta:+.2f}")
                            with col3:
                                st.metric("Period Max", f"{np.max(predictions):.2f}")
                                
                            # FIXED: Create continuous heatmap for forecast
                            values_grid = create_forecast_heatmap_grid(bounds, current_value, grid_size=50)
                                
                            with spatial_placeholder.container():
                                if map_type == "Detailed":
                                    m_forecast = folium.Map(
                                        location=[(bounds[1] + bounds[3])/2, (bounds[0] + bounds[2])/2],
                                        zoom_start=map_zoom_level
                                    )
                                    folium.TileLayer(
                                        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                                        attr='Esri',
                                        name='Esri Satellite'
                                    ).add_to(m_forecast)
                                    folium.TileLayer(
                                        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}',
                                        attr='Esri',
                                        name='Labels',
                                        overlay=True
                                    ).add_to(m_forecast)
                                elif map_type == "Terrain":
                                    m_forecast = folium.Map(
                                        location=[(bounds[1] + bounds[3])/2, (bounds[0] + bounds[2])/2],
                                        zoom_start=map_zoom_level,
                                        tiles="OpenStreetMap"
                                    )
                                else:
                                    m_forecast = folium.Map(
                                        location=[(bounds[1] + bounds[3])/2, (bounds[0] + bounds[2])/2],
                                        zoom_start=map_zoom_level,
                                        tiles="CartoDB positron"
                                    )
                                    
                                folium.Rectangle(
                                    bounds=[[bounds[1], bounds[0]], [bounds[3], bounds[2]]],
                                    color="red",
                                    weight=2,
                                    fill_opacity=0.0,
                                    popup=f"Forecast: {current_date.strftime('%Y-%m-%d')}"
                                ).add_to(m_forecast)
                                    
                                # Add continuous heatmap layer with dynamic legend
                                metric_description = AVAILABLE_PARAMETERS.get(forecast_target, "Fire Weather Index" if forecast_target == "FWI" else "Australian Fire Danger Rating" if forecast_target == "AFDR" else forecast_target)
                                heatmap_layer, legend_html = create_continuous_heatmap(
                                    bounds, values_grid,
                                    opacity=heatmap_opacity,
                                    metric_name=f"Forecast: {metric_description}"
                                )
                                heatmap_layer.add_to(m_forecast)
                                    
                                # Add legend to map
                                m_forecast.get_root().html.add_child(folium.Element(legend_html))
                                    
                                st_folium(
                                    m_forecast,
                                    key=f"forecast_map_{current_frame}",
                                    width=VIZ_FOLIUM_CELL_W,
                                    height=VIZ_FOLIUM_CELL_H,
                                    returned_objects=[],
                                )
                                
                            # Auto-advance if animating
                            if st.session_state.get('animation_running', False):
                                time.sleep(animation_speed)
                                st.session_state.current_frame = (st.session_state.current_frame + 1) % len(predictions)
                                    
                                if st.session_state.current_frame == 0:
                                    st.session_state.animation_running = False
                                    st.success("Animation cycle complete!")
                                    
                                if st.session_state.animation_running:
                                    st.rerun()
                    
                    # Export functionality
                    st.subheader("💾 Export Data")
                    
                    if forecast_mode and 'combined_export_data' in st.session_state:
                        st.write(f"Export includes historical {forecast_target} and all forecast models")
                        
                        if st.button("📥 Export Historical + Forecast CSV", key="export_forecast"):
                            export_data = st.session_state.combined_export_data
                            
                            csv_buffer = StringIO()
                            export_data.to_csv(csv_buffer, index=False)
                            csv_str = csv_buffer.getvalue()
                            
                            st.download_button(
                                label="⬇️ Download Historical + Forecast CSV",
                                data=csv_str,
                                file_name=f"{forecast_target}_forecast_{start_date.strftime('%Y%m%d')}_{forecast_horizon}days.csv",
                                mime="text/csv"
                            )
                            
                            st.success(f"Prepared {len(export_data)} records")
                    
                    else:
                        export_params = st.multiselect(
                            "Select parameters to export",
                            options=list(AVAILABLE_PARAMETERS.keys()) + ["FWI", "AFDR"],
                            default=[selected_metric],
                            key="export_params_regular"
                        )
                        
                        if st.button("📥 Export to CSV", key="export_regular"):
                            if export_params:
                                export_cols = ['lat', 'lon', 'date'] + export_params
                                export_data = gdf_filtered[export_cols].copy()
                                
                                csv_buffer = StringIO()
                                export_data.to_csv(csv_buffer, index=False)
                                csv_str = csv_buffer.getvalue()
                                
                                st.download_button(
                                    label="⬇️ Download CSV",
                                    data=csv_str,
                                    file_name=f"power_data_{start_date}_{end_date}.csv",
                                    mime="text/csv"
                                )
                                
                                st.success(f"Prepared {len(export_data)} records for export")
                            else:
                                st.warning("Select at least one parameter")


st.markdown("---")
st.subheader("Model reference")
c1, c2 = st.columns(2)
with c1:
    with st.expander("Linear Regression"):
        st.markdown("Ordinary least squares baseline; fast, interpretable, limited flexibility for nonlinear seasonality.")
    with st.expander("Random Forest"):
        st.markdown("Bagged decision trees; strong default for tabular weather features, handles nonlinearities.")
    with st.expander("Gradient Boosting / XGBoost"):
        st.markdown("Sequential tree ensembles with regularization (XGBoost); strong accuracy on structured features.")
    with st.expander("Prophet"):
        st.markdown("Additive time-series decomposition with seasonality; uses the target series and calendar effects.")
    with st.expander("SARIMA"):
        st.markdown("Seasonal ARIMA on the target series; classical statistical model with explicit seasonal orders.")
with c2:
    with st.expander("Ensemble"):
        st.markdown("Averages all selected non-ensemble method outputs for a consensus trajectory.")
    with st.expander("FBLiR (fuzzy Bayesian)"):
        st.markdown(
            "Gaussian fuzzy numbers + Bayesian-style coefficient uncertainty; **fits once per forecast run**, "
            "then predicts recursively like other tabular models. Tuning `adapt_steps` / `burnin_steps` / `thinning_steps` "
            "controls posterior sample count."
        )
    with st.expander("LLM Forecaster"):
        st.markdown(
            "Sends **long daily history** plus **monthly statistics** to an OpenAI model, with instructions to respect "
            "seasonality (not extrapolate short end spikes). The raw trajectory is **blended (~72%)** with a "
            "**day-of-year profile** from all past years. Requires `openai` and an API key; if the API is missing, "
            "the forecast is the seasonal profile only."
        )
    with st.expander("Useful insights"):
        st.markdown(
            "After each successful forecast, bullets appear under the title: **trend**, **seasonality (ACF)**, "
            "**peaks/troughs**, and **per-model forecasts**. A **fixed** insights model (`gpt-4o-mini` by default) "
            "refines wording when an API key is set; override with Streamlit secret `insights_openai_model` or "
            "env `INSIGHTS_OPENAI_MODEL`. Heuristic bullets always appear even without the API."
        )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
🎯 FAIM - Forecasting Analyzer of Ignition Metrics v1.5.3 | Built with Streamlit<br>
Powered by NASA POWER data from Goddard Earth Sciences Data and Information Services Center (GES DISC)
</div>
""", unsafe_allow_html=True)
