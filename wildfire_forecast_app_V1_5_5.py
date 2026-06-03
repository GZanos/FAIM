import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, Point, box
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import requests
from datetime import datetime, timedelta
import time
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from io import StringIO, BytesIO
import json
import hashlib
import html as html_module
import re
from pathlib import Path

import streamlit.components.v1 as components
from sklearn.linear_model import ARDRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.seasonal import STL
import warnings
import inspect
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


def parse_tau_sigma_0_params(params=None):
    """Read tau and sigma_0_squared from model_params dict."""
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


def sanitize_float_matrix(X, clip=1e10):
    """Replace non-finite values and clip extremes for sklearn / linear algebra."""
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


def sanitize_float_vector(y, clip=1e10):
    y = np.asarray(y, dtype=float).ravel()
    ok = np.isfinite(y)
    if not ok.all():
        fill = float(np.median(y[ok])) if ok.any() else 0.0
        y = np.where(ok, y, fill)
    if clip is not None and clip > 0:
        y = np.clip(y, -float(clip), float(clip))
    return y


def _conjugate_posterior_linear(X, y, tau=1.0, sigma_0_squared=1.0):
    X = sanitize_float_matrix(X)
    y = sanitize_float_vector(y)
    n, p = X.shape
    if n < 2:
        raise ValueError("Need at least 2 samples for Bayesian linear regression.")
    tau = max(float(tau), 1e-8)
    sigma_0_squared = max(float(sigma_0_squared), 1e-12)
    Xd = np.column_stack([np.ones(n, dtype=float), X])
    prior_var = np.concatenate([[sigma_0_squared], np.full(p, tau ** 2, dtype=float)])
    prior_prec = np.diag(1.0 / prior_var)
    y_hat_init = Xd @ (np.linalg.lstsq(Xd, y, rcond=None)[0])
    residuals = y - y_hat_init
    sigma_squared = float(max(np.var(residuals), 1e-12))
    prec_post = (Xd.T @ Xd) / sigma_squared + prior_prec
    prec_post = prec_post + np.eye(prec_post.shape[0], dtype=float) * 1e-10
    cov_post = np.linalg.inv(prec_post)
    mean_post = cov_post @ (Xd.T @ y / sigma_squared)
    return mean_post, cov_post, sigma_squared


class ConjugateBayesianLinearRegression:
    """Bayesian Linear Regression (BLiR) with explicit tau and sigma_0^2 priors."""

    def __init__(self, tau=1.0, sigma_0_squared=1.0):
        self.tau = float(tau)
        self.sigma_0_squared = float(sigma_0_squared)
        self.coef_ = None
        self.posterior_cov_ = None
        self.sigma_squared_ = None

    @classmethod
    def from_params(cls, params=None):
        tau, sigma_0_squared = parse_tau_sigma_0_params(params)
        return cls(tau=tau, sigma_0_squared=sigma_0_squared)

    def fit(self, X, y):
        self.coef_, self.posterior_cov_, self.sigma_squared_ = _conjugate_posterior_linear(
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


IWFR_DISPLAY_NAME = "Intelligent Wildfire Forecaster (IWFR)"

# Configure page
st.set_page_config(
    page_title=IWFR_DISPLAY_NAME,
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Match Streamlit dark UI: native Plotly dark template (avoids white plot + light text).
pio.templates.default = "plotly_dark"

# Larger close control on `st.dialog` modals (easier to tap than default icon).
st.markdown(
    """
    <style>
    div[data-testid="stDialog"] header button {
        min-width: 3rem !important;
        min-height: 3rem !important;
        width: 3rem !important;
        height: 3rem !important;
        padding: 0.35rem !important;
    }
    div[data-testid="stDialog"] header button svg {
        width: 1.75rem !important;
        height: 1.75rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
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
    from faim_guide_markdown import GUIDE_MARKDOWN as _GUIDE_MARKDOWN_RAW
except ImportError:
    _GUIDE_MARKDOWN_RAW = "**Guide text not found.** Add `faim_guide_markdown.py` next to the app."

_GUIDE_FBLIR_ANCHOR = "## 🎛️ FBLiR (Fuzzy Bayesian Linear Regression) Process"
_GUIDE_FBLIR_SUB = "### How FBLiR Handles Seasonality:"

# Forecast UX: show notice when horizon exceeds this many days (skill typically drops with lead time).
LONG_FORECAST_HORIZON_WARNING_DAYS = 365


def _strip_legacy_ascii_fblir_flowchart(md: str) -> str:
    """Remove legacy STEP 1–6 ASCII diagram (fenced or plain) if still present in guide text."""
    out = md
    out = re.sub(
        r"```[^\n]*\r?\n(?=[\s\S]*?STEP\s*1\s*:\s*DATA\s*PREPARATION)[\s\S]*?```\s*",
        "",
        out,
        flags=re.IGNORECASE,
    )
    token = "STEP 1: DATA PREPARATION"
    while token in out:
        i = out.index(token)
        window_start = max(0, i - 4000)
        win = out[window_start:i]
        j = win.rfind("┌")
        if j == -1:
            break
        line_start = out.rfind("\n", 0, window_start + j) + 1
        tail = out[i : min(len(out), i + 20000)]
        k = tail.rfind("└")
        if k == -1:
            break
        k2 = tail.find("┘", k)
        if k2 == -1:
            break
        e = i + k2 + 1
        while e < len(out) and out[e] in "\r\n":
            e += 1
        out = out[:line_start] + out[e:]
    return out


GUIDE_MARKDOWN = _strip_legacy_ascii_fblir_flowchart(_GUIDE_MARKDOWN_RAW)


def _split_guide_markdown_for_dialog(md: str):
    """Split full guide so the FBLiR diagram can sit under that section (no extra imports from guide file)."""
    i0 = md.find(_GUIDE_FBLIR_ANCHOR)
    i1 = md.find(_GUIDE_FBLIR_SUB)
    if i0 != -1 and i1 != -1 and i1 > i0:
        return md[:i0].rstrip(), md[i0:i1].rstrip(), md[i1:].lstrip()
    return md.rstrip(), "", ""


GUIDE_MARKDOWN_PREFIX, GUIDE_FBLIR_SECTION_HEADER_AND_INTRO, GUIDE_MARKDOWN_FBLIR_TAIL = (
    _split_guide_markdown_for_dialog(GUIDE_MARKDOWN)
)

_APP_DIR = Path(__file__).resolve().parent
_GUIDES_DIR = _APP_DIR / "Guides"
_GUIDE_VIDEO_NASA = _GUIDES_DIR / "Video 1.mov"
_GUIDE_VIDEO_FORECAST = _GUIDES_DIR / "Video 2.mov"

# Default Guide Helper videos (unlisted YouTube). Override via secrets / env if links change.
_DEFAULT_GUIDE_YT_NASA = "https://www.youtube.com/watch?v=EpOL1qipZKk"
_DEFAULT_GUIDE_YT_FORECAST = "https://www.youtube.com/watch?v=1cc4jNZHrAk"

_FBLIR_DIAGRAM_NAMES = (
    "fblir_flowchart.png",
    "FBLiR_flowchart.png",
    "FBLIR_flowchart.png",
)


def _resolve_fblir_diagram_path():
    """Find diagram on disk (Linux case-sensitive; layout differs local vs Cloud).

    On GitHub the PNG is often committed next to `faim_guide_markdown.py` (repo root),
    while `wildfire_forecast_app_*.py` may live in a subfolder — so we resolve via the
    guide module path and the repo parent, not only `Guides/`.
    """
    try:
        import faim_guide_markdown as _fgm

        guide_dir = Path(_fgm.__file__).resolve().parent
        for name in _FBLIR_DIAGRAM_NAMES:
            p = guide_dir / name
            if p.is_file():
                return p
    except Exception:
        pass

    search_dirs = (
        _GUIDES_DIR,
        _APP_DIR,
        _APP_DIR.parent,
        _APP_DIR / "assets",
        _APP_DIR / "static",
        _APP_DIR / "images",
    )
    for d in search_dirs:
        for name in _FBLIR_DIAGRAM_NAMES:
            p = d / name
            if p.is_file():
                return p
    return None


def _secret_or_env(secret_key: str, env_key: str):
    """Read Streamlit secret or process env (for assets too large to commit to GitHub)."""
    v = None
    try:
        v = st.secrets.get(secret_key)
    except Exception:
        pass
    if v is None or (isinstance(v, str) and not str(v).strip()):
        v = os.environ.get(env_key)
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def _video_format_for_source(src: str) -> str:
    s = src.lower().split("?", 1)[0]
    if s.endswith(".mov"):
        return "video/quicktime"
    if s.endswith(".webm"):
        return "video/webm"
    return "video/mp4"


def _youtube_video_id(url: str):
    """Extract 11-char YouTube id from watch, youtu.be, or embed URLs."""
    if not url or not isinstance(url, str):
        return None
    u = url.strip()
    m = re.search(
        r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([A-Za-z0-9_-]{11})",
        u,
    )
    return m.group(1) if m else None


def _render_youtube_embed(watch_or_embed_url: str, height: int = 420):
    """YouTube watch URLs are pages, not media files — embed the player (muted autoplay where allowed)."""
    vid = _youtube_video_id(watch_or_embed_url)
    if not vid:
        st.warning("Could not parse that YouTube link.")
        return
    src = (
        f"https://www.youtube-nocookie.com/embed/{vid}"
        "?mute=1&autoplay=1&playsinline=1&rel=0&modestbranding=1"
    )
    components.html(
        f'<iframe width="100%" height="{height}" src="{html_module.escape(src)}" '
        'title="Guide video" frameborder="0" '
        'allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" '
        "referrerpolicy=\"strict-origin-when-cross-origin\" "
        "allowfullscreen></iframe>",
        height=height + 24,
        scrolling=True,
    )


def _render_guide_video(local_path: Path, secret_key: str, env_key: str, fallback_youtube_url: str | None):
    """Prefer local file; else URL from secrets/env; else default YouTube (embed). Direct MP4/MOV URLs use st.video."""
    url = _secret_or_env(secret_key, env_key)
    if local_path.is_file():
        st.video(
            str(local_path),
            format=_video_format_for_source(str(local_path)),
            muted=True,
            autoplay=True,
        )
        return
    if url and _youtube_video_id(url):
        _render_youtube_embed(url)
        return
    if url:
        st.video(
            url,
            format=_video_format_for_source(url),
            muted=True,
            autoplay=True,
        )
        return
    if fallback_youtube_url and _youtube_video_id(fallback_youtube_url):
        _render_youtube_embed(fallback_youtube_url)
        return
    st.info(
        "No guide video available. Add a local file in **`Guides/`**, set **`"
        + secret_key
        + "`** in Streamlit secrets (or **`"
        + env_key
        + "`** in the environment), or use the built-in default YouTube links in the app."
    )


if hasattr(st, "dialog"):

    @st.dialog(f"{IWFR_DISPLAY_NAME} — How to use", width="large")
    def faim_howto_dialog():
        c1, c2 = st.columns(2)
        with c1:
            if st.button("NASA POWER API Functions Guide", use_container_width=True, key="howto_video_nasa"):
                st.session_state["_howto_video"] = "nasa"
        with c2:
            if st.button("Forecasting Guide", use_container_width=True, key="howto_video_forecast"):
                st.session_state["_howto_video"] = "forecast"

        choice = st.session_state.get("_howto_video")
        if choice == "nasa":
            _render_guide_video(
                _GUIDE_VIDEO_NASA,
                secret_key="guides_video_nasa_url",
                env_key="GUIDES_VIDEO_NASA_URL",
                fallback_youtube_url=_DEFAULT_GUIDE_YT_NASA,
            )
        elif choice == "forecast":
            _render_guide_video(
                _GUIDE_VIDEO_FORECAST,
                secret_key="guides_video_forecast_url",
                env_key="GUIDES_VIDEO_FORECAST_URL",
                fallback_youtube_url=_DEFAULT_GUIDE_YT_FORECAST,
            )
        else:
            st.caption("Tip: use the buttons above to play the NASA POWER or Forecasting guide videos (muted).")

        if GUIDE_FBLIR_SECTION_HEADER_AND_INTRO and GUIDE_MARKDOWN_FBLIR_TAIL:
            st.markdown(GUIDE_MARKDOWN_PREFIX)
            st.markdown(GUIDE_FBLIR_SECTION_HEADER_AND_INTRO)

            diagram_path = _resolve_fblir_diagram_path()
            diagram_url = _secret_or_env("guides_fblir_diagram_url", "GUIDES_FBLIR_DIAGRAM_URL")
            if diagram_path is not None:
                _half1, _half2 = st.columns([1, 1], gap="small")
                with _half1:
                    st.image(
                        str(diagram_path),
                        caption="FBLiR pipeline (model fit layer + fuzzy inference layer)",
                        use_container_width=True,
                    )
                _half2.empty()
            elif diagram_url:
                _half1, _half2 = st.columns([1, 1], gap="small")
                with _half1:
                    st.image(
                        diagram_url,
                        caption="FBLiR pipeline (model fit layer + fuzzy inference layer)",
                        use_container_width=True,
                    )
                _half2.empty()
            else:
                st.warning(
                    "FBLiR diagram image not found in the deployed app. "
                    "Commit **`fblir_flowchart.png`** next to `faim_guide_markdown.py` (or under **`Guides/`**), "
                    "or set secret **`guides_fblir_diagram_url`** / env **`GUIDES_FBLIR_DIAGRAM_URL`** to a direct image URL."
                )

            st.markdown(GUIDE_MARKDOWN_FBLIR_TAIL)
        else:
            st.markdown(GUIDE_MARKDOWN_PREFIX)

else:

    def faim_howto_dialog():
        st.sidebar.warning("Upgrade Streamlit to 1.33+ for the guide popup.")

# Title and description
st.title(f"🎯 {IWFR_DISPLAY_NAME}")
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

DATA_SOURCE_QUICK_HISTORICAL = "Quick Historical Data"

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

@st.cache_resource
def load_global_land_geometry():
    """Load world land polygons once for AOI land/ocean validation."""
    try:
        world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    except Exception:
        world = None
    if world is None:
        try:
            import requests
            import tempfile
            import zipfile
            from pathlib import Path
            from io import BytesIO

            url = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            with tempfile.TemporaryDirectory() as td:
                zf = zipfile.ZipFile(BytesIO(resp.content))
                zf.extractall(td)
                shp = next(Path(td).glob("*.shp"), None)
                if shp is None:
                    return None
                world = gpd.read_file(str(shp))
        except Exception:
            return None
    world = world.to_crs("EPSG:4326")
    world = world[world.geometry.notna()].copy()
    if world.empty:
        return None
    return world.unary_union


def evaluate_aoi_land_coverage(bounds, land_geom):
    """
    Return dict with AOI polygon, land overlap ratio and validity flags.
    Uses equal-area projection for robust area ratios.
    """
    minx, miny, maxx, maxy = bounds
    aoi_poly = box(minx, miny, maxx, maxy)
    if land_geom is None:
        return {
            "aoi_poly": aoi_poly,
            "land_ratio": 1.0,
            "is_non_land": False,
            "has_partial_ocean": False,
            "land_geom_aoi": aoi_poly,
        }
    try:
        land_geom_aoi = aoi_poly.intersection(land_geom)
        gdf_tmp = gpd.GeoDataFrame(
            {"kind": ["aoi", "land"]},
            geometry=[aoi_poly, land_geom_aoi],
            crs="EPSG:4326",
        ).to_crs("EPSG:6933")
        aoi_area = float(gdf_tmp.geometry.iloc[0].area)
        land_area = float(gdf_tmp.geometry.iloc[1].area)
        ratio = 0.0 if aoi_area <= 0 else max(0.0, min(1.0, land_area / aoi_area))
    except Exception:
        ratio = 1.0
        land_geom_aoi = aoi_poly
    return {
        "aoi_poly": aoi_poly,
        "land_ratio": ratio,
        "is_non_land": ratio < 0.05,
        "has_partial_ocean": 0.05 <= ratio < 0.98,
        "land_geom_aoi": land_geom_aoi,
    }


def filter_points_to_land(gdf, land_geom_aoi):
    """Keep only points on land for selected AOI."""
    if gdf is None or gdf.empty or land_geom_aoi is None or getattr(land_geom_aoi, "is_empty", False):
        return gdf
    try:
        keep = gdf.geometry.apply(lambda p: bool(land_geom_aoi.covers(p)))
        return gdf[keep].copy()
    except Exception:
        return gdf

def create_continuous_heatmap(
    bounds,
    values_grid,
    gradient_colormap='RdYlBu_r',
    opacity=0.4,
    metric_name="Value",
    full_map_bounds=None,
    color_scheme="blue",
):
    """
    Folium HeatMap overlay with blue or red gradient (normalized by AOI min/max).
    Returns: (heatmap_layer, legend_html)
    """
    minx, miny, maxx, maxy = bounds
    grid_height, grid_width = values_grid.shape

    blue_gradient = {
        0.0: '#E3F2FD',
        0.2: '#90CAF9',
        0.4: '#42A5F5',
        0.6: '#1E88E5',
        0.8: '#1565C0',
        1.0: '#0D47A1',
    }
    blue_css = "#E3F2FD,#90CAF9,#42A5F5,#1E88E5,#1565C0,#0D47A1"
    red_gradient = {
        0.0: '#FFF8F8',
        0.2: '#FFCDD2',
        0.4: '#E57373',
        0.6: '#E53935',
        0.8: '#C62828',
        1.0: '#7F0000',
    }
    red_css = "#FFF8F8,#FFCDD2,#E57373,#E53935,#C62828,#7F0000"
    scheme = str(color_scheme).lower().strip()
    gradient_map = red_gradient if scheme == "red" else blue_gradient
    legend_css = red_css if scheme == "red" else blue_css

    vmin = np.nanmin(values_grid)
    vmax = np.nanmax(values_grid)
    vrange = vmax - vmin

    heatmap_data = []
    for i in range(grid_height):
        for j in range(grid_width):
            value = values_grid[i, j]
            if not np.isnan(value):
                lat = miny + (i / grid_height) * (maxy - miny)
                lon = minx + (j / grid_width) * (maxx - minx)
                if vmax > vmin:
                    normalized_intensity = (value - vmin) / (vmax - vmin)
                else:
                    normalized_intensity = 0.5
                heatmap_data.append([lat, lon, normalized_intensity])

    heatmap_layer = folium.FeatureGroup(name='Heatmap')

    HeatMap(
        heatmap_data,
        min_opacity=0.3,
        max_opacity=0.7,
        radius=25,
        blur=20,
        gradient=gradient_map,
    ).add_to(heatmap_layer)

    safe_title = html_module.escape(str(metric_name))
    mean_g = float(np.nanmean(values_grid))
    std_g = float(np.nanstd(values_grid))
    inner_legend = (
        f'<p style="margin:0 0 8px 0;font-weight:bold;text-align:center;color:#000;font-size:14px;line-height:1.2;">{safe_title}</p>'
        '<div style="margin-bottom:6px;">'
        '<div style="width:100%;height:18px;background:linear-gradient(to right,'
        f'{legend_css});border:2px solid #333;border-radius:3px;"></div>'
        '</div>'
        '<div style="display:flex;justify-content:space-between;font-size:12px;margin-top:4px;color:#000;font-weight:600;">'
        f'<span><b>Min:</b> {vmin:.1f}</span><span><b>Max:</b> {vmax:.1f}</span>'
        '</div>'
        '<div style="text-align:center;font-size:11px;margin-top:6px;padding-top:6px;border-top:2px solid #ccc;color:#000;font-weight:600;">'
        f'<div><b>Range:</b> {vrange:.1f}</div><div><b>Mean:</b> {mean_g:.1f}</div><div><b>Std:</b> {std_g:.1f}</div>'
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
    anchor.style.cssText = 'position:absolute;top:10px;left:50%;transform:translateX(-50%);width:220px;max-width:78%;z-index:6500;font-size:13px;pointer-events:none;';
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


def create_heatmap_data(
    gdf,
    aoi_bounds,
    metric,
    grid_size=80,
    n_anchor_lon=5,
    n_anchor_lat=4,
    reference_std=None,
    land_geom_aoi=None,
):
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

    # Mask ocean cells so heatmap only covers land inside AOI.
    if land_geom_aoi is not None and not getattr(land_geom_aoi, "is_empty", False):
        land_mask = np.zeros_like(values_grid, dtype=bool)
        for i, lat in enumerate(lat_c):
            for j, lon in enumerate(lon_c):
                try:
                    land_mask[i, j] = bool(land_geom_aoi.covers(Point(float(lon), float(lat))))
                except Exception:
                    land_mask[i, j] = True
        values_grid = np.where(land_mask, values_grid, np.nan)

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
        template="plotly_dark",
        font=dict(size=12),
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(255,255,255,0.12)")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(255,255,255,0.12)")
    
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
            df_clean[col] = (
                df_clean[col]
                .replace([np.inf, -np.inf], np.nan)
                .ffill()
                .bfill()
                .fillna(0)
            )
    
    return df_clean, feature_names


def _apply_target_smoothing(y_values, smoothing_method, window_days, dates=None):
    """Optionally smooth target series before model fitting."""
    y = np.asarray(y_values, dtype=float).ravel()
    if smoothing_method == "None (no smoothing)":
        return y, ""

    if smoothing_method == "Seasonal smoothing (STL)":
        return _apply_stl_target_smoothing(y, dates)

    w = int(max(2, window_days))
    s = pd.Series(y)
    if smoothing_method == "Rolling mean":
        y_smooth = s.rolling(window=w, min_periods=1).mean().values
        return np.asarray(y_smooth, dtype=float), f"Target smoothing: rolling mean ({w} days)."
    if smoothing_method == "Rolling median":
        y_smooth = s.rolling(window=w, min_periods=1).median().values
        return np.asarray(y_smooth, dtype=float), f"Target smoothing: rolling median ({w} days)."
    if smoothing_method == "Exponential moving average (EMA)":
        y_smooth = s.ewm(span=w, adjust=False, min_periods=1).mean().values
        return np.asarray(y_smooth, dtype=float), f"Target smoothing: EMA (span={w} days)."
    return y, ""


def _forward_target_model_scale(smoothed_y: np.ndarray, use_log: bool):
    """Apply optional ln(1+y) on nonnegative smoothed target for model fitting."""
    y = np.asarray(smoothed_y, dtype=float).copy()
    y[~np.isfinite(y)] = np.nan
    if use_log:
        y = np.where(np.isfinite(y), np.maximum(y, 0.0), np.nan)
        y = np.nan_to_num(y, nan=0.0)
        return np.log1p(y), (
            "Target uses ln(1+y) before fitting; forecast traces use original scale."
        )
    return y, None


def _inverse_target_transform_predictions(predictions, use_log: bool) -> np.ndarray:
    """Map model-scale forecasts back to original units for plotting and export."""
    p = np.asarray(predictions, dtype=float)
    if not use_log:
        return p
    return np.expm1(p)


def _apply_stl_target_smoothing(y, dates):
    """STL trend + seasonal (annual) for long daily series; removes irregular component."""
    if dates is None:
        return np.asarray(y, dtype=float), "Target smoothing: STL skipped (no dates on series)."

    y = np.asarray(y, dtype=float).ravel()
    d_raw = pd.to_datetime(pd.Series(dates, copy=False), errors="coerce")
    if len(d_raw) != len(y):
        return y, "Target smoothing: STL skipped (dates length mismatch)."

    if not d_raw.notna().any():
        return y, "Target smoothing: STL skipped (invalid dates)."

    ok = d_raw.notna().to_numpy()
    df = pd.DataFrame({"date": d_raw[ok].to_numpy(), "y": y[ok]}).sort_values("date")
    df = df.drop_duplicates(subset=["date"], keep="last")
    if df.empty:
        return y, "Target smoothing: STL skipped (invalid dates)."

    dmin, dmax = df["date"].min(), df["date"].max()
    full_idx = pd.date_range(dmin, dmax, freq="D")
    ser = df.set_index("date")["y"].astype(float).reindex(full_idx)
    if ser.notna().sum() < 10:
        return y, "Target smoothing: STL skipped (too few valid daily values)."

    ser = ser.interpolate(method="linear", limit_direction="both").bfill().ffill()

    period = 365
    if len(ser) < 2 * period:
        return np.asarray(y, dtype=float), (
            f"Target smoothing: STL needs at least {2 * period} daily points (~2 years); "
            "left target unchanged."
        )

    try:
        stl = STL(ser, period=period, seasonal=15, robust=True)
        res = stl.fit()
        smoothed_daily = pd.Series(
            np.asarray(res.trend, dtype=float).ravel() + np.asarray(res.seasonal, dtype=float).ravel(),
            index=ser.index,
        )
        mapped = smoothed_daily.reindex(pd.DatetimeIndex(d_raw))
        out = np.where(mapped.notna().to_numpy(), mapped.to_numpy(dtype=float), y)
        return np.asarray(out, dtype=float), (
            "Target smoothing: STL (trend + annual seasonality, period=365 days)."
        )
    except Exception:
        return np.asarray(y, dtype=float), "Target smoothing: STL failed; left target unchanged."

def _build_blir_regressor(params=None):
    """ARDRegression — stable BLiR path used before paper-prior experiment."""
    params = params or {}
    return ARDRegression(
        alpha_1=float(params.get("alpha_1", 1e-6)),
        alpha_2=float(params.get("alpha_2", 1e-6)),
        lambda_1=float(params.get("lambda_1", 1e-6)),
        lambda_2=float(params.get("lambda_2", 1e-6)),
        max_iter=700,
        tol=1e-4,
        threshold_lambda=1e12,
        compute_score=False,
    )


def _seasonal_target_reference(df_ml, forecast_target, target_date, window=12):
    """Historical median target for the same time of year (drives seasonal recursive forecasts)."""
    if df_ml is None or forecast_target not in df_ml.columns or "day_of_year" not in df_ml.columns:
        return None
    doy = int(pd.Timestamp(target_date).dayofyear)
    hist_doy = df_ml["day_of_year"].astype(int).to_numpy()
    delta = np.abs(hist_doy - doy)
    delta = np.minimum(delta, 366 - delta)
    mask = delta <= int(window)
    vals = df_ml.loc[mask, forecast_target].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(vals) < 3:
        vals = df_ml[forecast_target].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(vals) == 0:
        return None
    return float(np.median(vals.to_numpy()))


def _auto_tune_blir_params(X_scaled, y, holdout_days=28):
    """Pick ARD hyperparameters by one-step holdout MAE on the most recent days."""
    defaults = {
        "alpha_1": 1e-6,
        "alpha_2": 1e-6,
        "lambda_1": 1e-6,
        "lambda_2": 1e-6,
    }
    y_arr = np.asarray(y, dtype=float).ravel()
    holdout = int(max(14, min(holdout_days, len(y_arr) // 4)))
    if len(y_arr) < holdout + 40:
        return defaults, None

    X_tr = X_scaled.iloc[:-holdout]
    y_tr = y_arr[:-holdout]
    X_ho = X_scaled.iloc[-holdout:]
    y_ho = y_arr[-holdout:]

    candidates = []
    for exp in (-8, -6, -4, -3):
        v = float(10.0 ** exp)
        candidates.append({"alpha_1": v, "alpha_2": v, "lambda_1": v, "lambda_2": v})
    candidates.extend([
        {"alpha_1": 1e-4, "alpha_2": 1e-4, "lambda_1": 1e-6, "lambda_2": 1e-6},
        {"alpha_1": 1e-6, "alpha_2": 1e-6, "lambda_1": 1e-4, "lambda_2": 1e-4},
        {"alpha_1": 1e-5, "alpha_2": 1e-5, "lambda_1": 1e-7, "lambda_2": 1e-7},
    ])

    best_mae = float("inf")
    best = defaults
    for cand in candidates:
        try:
            model = _build_blir_regressor(cand)
            model.fit(X_tr, y_tr)
            pred = np.asarray(model.predict(X_ho), dtype=float).ravel()
            mae = float(np.mean(np.abs(y_ho - pred)))
            if np.isfinite(mae) and mae < best_mae:
                best_mae = mae
                best = cand
        except Exception:
            continue

    note = f"BLiR auto-tuned on last {holdout} days (holdout MAE={best_mae:.4f})."
    return best, note


def _fblir_supports_prescaled(model):
    try:
        return "input_prescaled" in inspect.signature(model.fit).parameters
    except (TypeError, ValueError):
        return False


def _fblir_fit_model(model, X_scaled, y, X_raw, blir_model=None):
    """Fit FBLiR with scaled features when supported; otherwise unscaled once (legacy module)."""
    model._iwfr_fblir_backbone = False
    if blir_model is not None and hasattr(model, "fit_with_blir_backbone"):
        try:
            model.fit_with_blir_backbone(blir_model, X_scaled, y, input_prescaled=True)
            model._iwfr_uses_prescaled = True
            model._iwfr_fblir_backbone = True
            return model
        except TypeError:
            pass
    if _fblir_supports_prescaled(model):
        model.fit(X_scaled, y, input_prescaled=True)
        model._iwfr_uses_prescaled = True
    else:
        model.fit(X_raw, y)
        model._iwfr_uses_prescaled = False
    return model


def _fblir_predict_values(model, X_scaled, X_raw=None, linear_only=False):
    """Predict with FBLiR; compatible with legacy modules lacking input_prescaled."""
    use_prescaled = getattr(model, "_iwfr_uses_prescaled", _fblir_supports_prescaled(model))
    X_in = X_scaled if use_prescaled else (X_raw if X_raw is not None else X_scaled)

    if linear_only:
        if hasattr(model, "predict_linear_mean"):
            try:
                return np.asarray(
                    model.predict_linear_mean(X_in, input_prescaled=use_prescaled),
                    dtype=float,
                ).ravel()
            except TypeError:
                return np.asarray(model.predict_linear_mean(X_in), dtype=float).ravel()
        try:
            return np.asarray(
                model.predict(X_in, input_prescaled=use_prescaled, linear_only=True),
                dtype=float,
            ).ravel()
        except TypeError:
            pass

    if use_prescaled:
        try:
            return np.asarray(model.predict(X_in, input_prescaled=True), dtype=float).ravel()
        except TypeError:
            return np.asarray(model.predict(X_in), dtype=float).ravel()
    if X_raw is None:
        X_raw = X_scaled
    return np.asarray(model.predict(X_raw), dtype=float).ravel()


def _stabilize_fblir_recursive_prediction(
    pred,
    seasonal_ref,
    prev_pred,
    step_idx,
    forecast_horizon,
    linear_z_lo,
    linear_z_hi,
):
    """Tame FBLiR recursive drift: more seasonal anchoring and step-to-step limits later in the horizon."""
    pred = float(pred)
    if not np.isfinite(pred):
        pred = float(prev_pred) if prev_pred is not None and np.isfinite(prev_pred) else pred

    if seasonal_ref is not None and np.isfinite(seasonal_ref):
        horizon = max(int(forecast_horizon), 1)
        progress = float(step_idx) / float(max(horizon - 1, 1))
        w_season = min(0.72, 0.38 + 0.34 * progress)
        pred = (1.0 - w_season) * pred + w_season * float(seasonal_ref)

    if prev_pred is not None and np.isfinite(prev_pred):
        span = max(abs(prev_pred), 1.0)
        max_delta = max(0.35, 0.12 * span)
        pred = float(np.clip(pred, prev_pred - max_delta, prev_pred + max_delta))

    return float(np.clip(pred, linear_z_lo, linear_z_hi))


def _auto_tune_fblir_params(X_scaled, y, X_raw, base_params, holdout_days=21, blir_model=None):
    """Light grid search for FBLiR GFN settings (one-step holdout, BLiR backbone fixed)."""
    params = dict(base_params or {})
    y_arr = np.asarray(y, dtype=float).ravel()
    holdout = int(max(14, min(holdout_days, len(y_arr) // 4)))
    if len(y_arr) < holdout + 40 or not FBLIR_AVAILABLE or FuzzyBayesianRegression is None:
        return params, None

    if blir_model is None:
        blir_model = _build_blir_regressor({})
        blir_model.fit(X_scaled.iloc[:-holdout], y_arr[:-holdout])

    X_tr_s, X_ho_s = X_scaled.iloc[:-holdout], X_scaled.iloc[-holdout:]
    X_tr_r, X_ho_r = X_raw.iloc[:-holdout], X_raw.iloc[-holdout:]
    y_tr, y_ho = y_arr[:-holdout], y_arr[-holdout:]

    m_vals = [0.05, 0.1, 0.15]
    fuzz_vals = [0.02, 0.05, 0.08]

    best_mae = float("inf")
    best = params
    for m_val in m_vals:
        for fuzz_val in fuzz_vals:
            trial = dict(params)
            trial.update({"m": m_val, "fuzzification_factor": fuzz_val})
            try:
                model = _build_fblir_regressor(trial)
                _fblir_fit_model(model, X_tr_s, y_tr, X_tr_r, blir_model=blir_model)
                pred = _fblir_predict_values(model, X_ho_s, X_ho_r, linear_only=False)
                mae = float(np.mean(np.abs(y_ho - pred)))
                if np.isfinite(mae) and mae < best_mae:
                    best_mae = mae
                    best = trial
            except Exception:
                continue

    note = f"FBLiR GFN auto-tuned on last {holdout} days (holdout MAE={best_mae:.4f}, BLiR core fixed)."
    return best, note


def get_seasonal_feature_value(historical_data, feature_name, target_date, lookback_years=3, deterministic=False):
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
        seasonal_value = float(matching_rows[feature_name].mean())
        if deterministic:
            return seasonal_value
        # Small random variation (±5%) for realism in exploratory runs only.
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

def _fblir_n_samples_from_params(params: dict) -> int:
    adapt = int(params.get("adapt_steps", 200))
    burnin = int(params.get("burnin_steps", 200))
    n_chains = max(1, int(params.get("N_chains", 2)))
    thinning = max(1, int(params.get("thinning_steps", 7)))
    return min(max(100, ((adapt + burnin) * n_chains) // thinning), 2200)


def _build_fblir_regressor(params: dict | None):
    """Construct FuzzyBayesianRegression with GFN + prior hyperparameters (tau, sigma_0^2)."""
    params = params or {}
    tau, sigma_0_squared = parse_tau_sigma_0_params(params)
    n_samples = _fblir_n_samples_from_params(params)
    base_kw = dict(
        n_samples=n_samples,
        symmetry_threshold=params.get("symmetry_threshold", 0.5),
        k=params.get("k", 0.5),
        m=params.get("m", 0.1),
        fuzzify_variance=params.get("fuzzification_factor", 0.05),
        use_quadratic=True,
        small_delta_threshold=params.get("symmetry_threshold", 0.4),
    )
    try:
        return FuzzyBayesianRegression(tau=tau, sigma_0_squared=sigma_0_squared, **base_kw)
    except TypeError:
        return FuzzyBayesianRegression(**base_kw)


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
            'N_chains': 2,
            'adapt_steps': 200,
            'burnin_steps': 200,
            'thinning_steps': 7,
            'tau': 1.0,
            'sigma_0_squared': 1.0,
        }
    
    try:
        # Check if we have FuzzyBayesianRegression (V2/V3) or only FuzzyBayesianRegressionTuned (original)
        if FuzzyBayesianRegression is not None:
            model = _build_fblir_regressor(params)
            _fblir_fit_model(model, X_train, y_train, X_train)
            predictions = _fblir_predict_values(model, X_future, X_future)
            return predictions
        else:
            # Fallback to original FuzzyBayesianRegressionTuned (doesn't support all parameters)
            # Use reasonable n_samples calculation
            base_samples = 500
            n_chains = max(1, params.get('N_chains', 2))
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


def _acf_plotly(
    series,
    title="Autocorrelation",
    max_lag=40,
    adjust_slow_acf=True,
    height=300,
    margin=None,
):
    y_raw = np.asarray(series, dtype=float)
    y_raw = y_raw[np.isfinite(y_raw)]
    adjusted = False
    if adjust_slow_acf and len(y_raw) >= 90:
        y_work, adjusted = _acf_series_remove_slow_variation(y_raw)
    else:
        y_work = y_raw - np.mean(y_raw)

    acf, ml = _acf_numpy(y_work, max_lag=max_lag)
    lags = np.arange(len(acf))
    if len(lags) > 1:
        lags = lags[1:]
        acf = acf[1:]
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
        height=int(height),
        showlegend=False,
        margin=margin if margin is not None else dict(t=60, b=40),
        autosize=True,
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


def _simple_forecast_findings_summary(forecast_results, y_hist, forecast_target, future_dates):
    """2-3 plain-language sentences for non-technical readers."""
    if not forecast_results:
        return ""
    hist_mean = float(np.mean(y_hist)) if len(y_hist) else None
    method_means = {m: float(np.mean(np.asarray(v, dtype=float))) for m, v in forecast_results.items()}
    sorted_methods = sorted(method_means.items(), key=lambda kv: kv[1])
    low_m, low_v = sorted_methods[0]
    high_m, high_v = sorted_methods[-1]
    span = high_v - low_v
    if hist_mean is not None:
        closest_m = min(method_means, key=lambda m: abs(method_means[m] - hist_mean))
        dir_word = "above" if method_means[closest_m] >= hist_mean else "below"
        s1 = (
            f"Across models, expected {forecast_target} stays in a similar overall band, "
            f"with {low_m} lowest and {high_m} highest on average."
        )
        s2 = (
            f"The spread between models is moderate ({span:.2f} units), and {closest_m} tracks "
            f"closest to recent historical conditions ({dir_word} the historical average)."
        )
    else:
        s1 = (
            f"Across models, expected {forecast_target} remains in a consistent range, "
            f"from {low_m} (lower) to {high_m} (higher)."
        )
        s2 = f"Model-to-model spread is about {span:.2f} units, indicating moderate uncertainty."
    if len(future_dates):
        s3 = f"This outlook covers {len(future_dates)} days ({future_dates[0].strftime('%Y-%m-%d')} to {future_dates[-1].strftime('%Y-%m-%d')})."
    else:
        s3 = "The forecast horizon uses the user-selected future window."
    return f"{s1}\n\n{s2}\n\n{s3}"


def _forecast_method_export_column_name(method: str) -> str:
    """Stable CSV column name for a forecast method (e.g. Random Forest -> forecast_Random_Forest)."""
    base = str(method).strip().replace(" ", "_").replace("/", "_")
    return f"forecast_{base}"


def build_forecast_export_wide(
    daily_avg: pd.DataFrame,
    selected_features: list,
    forecast_target: str,
    future_dates,
    forecast_results: dict,
) -> pd.DataFrame:
    """
    Wide export: date | selected feature AOI means | target | one column per model | row_type.
    Historical rows have NaN in forecast_* columns; future rows have NaN in feature/target observables.
    """
    feat_cols = [c for c in selected_features if c in daily_avg.columns]
    hist_cols = ["date"] + feat_cols + [forecast_target]
    hist = daily_avg[hist_cols].copy()
    for m in forecast_results:
        hist[_forecast_method_export_column_name(m)] = np.nan
    hist["row_type"] = "historical"

    fut = pd.DataFrame({"date": future_dates})
    for c in feat_cols:
        fut[c] = np.nan
    fut[forecast_target] = np.nan
    for method, pred in forecast_results.items():
        fut[_forecast_method_export_column_name(method)] = np.asarray(pred, dtype=float)
    fut["row_type"] = "forecast"

    out = pd.concat([hist, fut], ignore_index=True)
    out = out.sort_values("date").reset_index(drop=True)
    fc_ordered = [_forecast_method_export_column_name(m) for m in forecast_results.keys()]
    col_order = ["date"] + feat_cols + [forecast_target] + fc_ordered + ["row_type"]
    out = out[[c for c in col_order if c in out.columns]]
    return out


def _on_long_forecast_horizon_dismiss() -> None:
    """Called when the user closes the long-horizon dialog (X, Esc, or click-outside)."""
    st.session_state["long_forecast_horizon_dialog_dismissed"] = True


@st.dialog("Long forecast horizon", on_dismiss=_on_long_forecast_horizon_dismiss)
def _long_forecast_horizon_dialog(forecast_horizon: int, threshold_days: int) -> None:
    st.markdown(
        f"You selected a **{forecast_horizon}-day** forecast (more than **{threshold_days}** days). "
        "**Precision and skill typically decrease** as the horizon lengthens—long-range values are best treated "
        "as **indicative scenarios**, not high-confidence point predictions."
    )
    st.caption("Close this notice with the **×** in the top-right (or **Esc**).")


def _in_sample_seasonal_baseline_fit(df_ml, forecast_target, min_train=30):
    """Fallback in-sample fitted values using rolling seasonal baseline."""
    df = df_ml[["date", forecast_target]].dropna().sort_values("date").reset_index(drop=True)
    if len(df) < min_train + 5:
        return None
    fitted = np.full(len(df), np.nan, dtype=float)
    for i in range(min_train, len(df)):
        hist = df.iloc[:i]
        tdoy = int(df.loc[i, "date"].timetuple().tm_yday)
        h = hist.copy()
        h["doy"] = h["date"].dt.dayofyear
        diff = (h["doy"] - tdoy).abs()
        diff = np.minimum(diff, 366 - diff)
        block = h.loc[diff <= 10, forecast_target]
        if len(block) == 0:
            fitted[i] = float(hist[forecast_target].mean())
        else:
            fitted[i] = float(block.mean())
    valid = np.isfinite(fitted)
    if valid.sum() < 10:
        return None
    return fitted[valid], df.loc[valid, forecast_target].to_numpy(dtype=float)


def _fit_in_sample_predictions(method, X_scaled, y, df_ml, forecast_target, model_params):
    """Compute fitted (in-sample) predictions for residual diagnostics."""
    y_arr = np.asarray(y, dtype=float)
    if method == "Linear Regression":
        # Ridge on standardized features; aligns with iterative forecast path.
        m = Ridge(alpha=1.5, random_state=42)
        m.fit(X_scaled, y_arr)
        return np.asarray(m.predict(X_scaled), dtype=float)
    if method == "Bayesian Linear Regression":
        mp = model_params.get("Bayesian Linear Regression") or {}
        m = _build_blir_regressor(mp)
        m.fit(X_scaled, y_arr)
        return np.asarray(m.predict(X_scaled), dtype=float)
    if method == "Random Forest":
        mp = model_params.get("Random Forest") or {}
        m = RandomForestRegressor(
            n_estimators=mp.get("n_estimators", 100),
            max_depth=mp.get("max_depth", 10),
            min_samples_split=mp.get("min_samples_split", 2),
            random_state=42,
            n_jobs=-1,
        )
        m.fit(X_scaled, y_arr)
        return np.asarray(m.predict(X_scaled), dtype=float)
    if method == "Gradient Boosting":
        mp = model_params.get("Gradient Boosting") or {}
        m = GradientBoostingRegressor(
            n_estimators=mp.get("n_estimators", 100),
            max_depth=mp.get("max_depth", 5),
            learning_rate=mp.get("learning_rate", 0.1),
            random_state=42,
        )
        m.fit(X_scaled, y_arr)
        return np.asarray(m.predict(X_scaled), dtype=float)
    if method == "XGBoost":
        import xgboost as xgb

        mp = model_params.get("XGBoost") or {}
        m = xgb.XGBRegressor(
            n_estimators=mp.get("n_estimators", 100),
            max_depth=mp.get("max_depth", 6),
            learning_rate=mp.get("learning_rate", 0.1),
            random_state=42,
            n_jobs=-1,
        )
        m.fit(X_scaled, y_arr)
        return np.asarray(m.predict(X_scaled), dtype=float)
    if method == "FBLiR":
        params = dict(model_params.get("FBLiR") or {})
        X_raw = df_ml[list(X_scaled.columns)]
        blr_mp = dict(model_params.get("Bayesian Linear Regression") or {})
        blir_core = _build_blir_regressor(blr_mp)
        blir_core.fit(X_scaled, y_arr)
        if FuzzyBayesianRegression is not None:
            m = _build_fblir_regressor(params)
            _fblir_fit_model(m, X_scaled, y_arr, X_raw, blir_model=blir_core)
            return _fblir_predict_values(
                m,
                X_scaled,
                X_raw,
                linear_only=getattr(m, "_iwfr_fblir_backbone", False),
            )
        split_idx = int(len(X_scaled) * 0.8)
        X_train_f = X_scaled.iloc[:split_idx]
        y_train_f = pd.Series(y_arr).iloc[:split_idx]
        X_val_f = X_scaled.iloc[split_idx:]
        y_val_f = pd.Series(y_arr).iloc[split_idx:]
        base_samples = min(max(100, int(params.get("adapt_steps", 200)) + int(params.get("burnin_steps", 200))), 1500)
        m = FuzzyBayesianRegressionTuned(n_samples=base_samples, use_quadratic=True)
        try:
            m.fit(X_train_f, y_train_f, X_val_f, y_val_f, input_prescaled=True)
            m._iwfr_uses_prescaled = True
        except TypeError:
            m.fit(X_raw.iloc[:split_idx], y_train_f, X_raw.iloc[split_idx:], y_val_f)
            m._iwfr_uses_prescaled = False
        return _fblir_predict_values(m, X_scaled, X_raw)
    if method == "Prophet":
        df_prophet = df_ml[["date", forecast_target]].copy()
        df_prophet.columns = ["ds", "y"]
        from prophet import Prophet

        pp = model_params.get("Prophet") or {}
        m = Prophet(
            daily_seasonality=pp.get("daily_seasonality", True),
            weekly_seasonality=pp.get("weekly_seasonality", True),
            yearly_seasonality=pp.get("yearly_seasonality", True),
            seasonality_mode=pp.get("seasonality_mode", "multiplicative"),
        )
        m.fit(df_prophet)
        fitted = m.predict(df_prophet[["ds"]])["yhat"].values
        return np.asarray(fitted, dtype=float)
    if method == "LLM Forecaster":
        baseline = _in_sample_seasonal_baseline_fit(df_ml, forecast_target, min_train=30)
        if baseline is None:
            return None
        fitted_valid, y_valid = baseline
        full = np.full(len(y_arr), np.nan, dtype=float)
        full[-len(fitted_valid):] = fitted_valid
        return full
    return None


def _residual_diagnostic_bundle(y_true, fitted, method):
    """Build residual plots + Shapiro-Wilk test for one model."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    fitted = np.asarray(fitted, dtype=float).ravel()
    n = min(len(y_true), len(fitted))
    if n < 10:
        return {"error": "Need at least 10 points for residual diagnostics."}
    y_true = y_true[-n:]
    fitted = fitted[-n:]
    residuals = y_true - fitted
    if not np.isfinite(residuals).all():
        mask = np.isfinite(residuals) & np.isfinite(fitted)
        residuals = residuals[mask]
        fitted = fitted[mask]
    if len(residuals) < 10:
        return {"error": "Not enough finite residuals after cleaning."}

    try:
        from scipy import stats
    except Exception:
        return {"error": "Residual tests require scipy. Add scipy to requirements.txt."}

    shapiro_stat, shapiro_p = stats.shapiro(residuals if len(residuals) <= 5000 else residuals[-5000:])
    osm, osr = stats.probplot(residuals, dist="norm", fit=False)
    qq_fig = go.Figure()
    qq_fig.add_trace(go.Scatter(x=osm, y=osr, mode="markers", name="Residual quantiles"))
    slope, intercept, r_val = stats.linregress(osm, osr)[:3]
    xx = np.linspace(np.min(osm), np.max(osm), 80)
    qq_fig.add_trace(go.Scatter(x=xx, y=slope * xx + intercept, mode="lines", name="Reference line"))
    qq_fig.update_layout(
        title=f"QQ plot — {method} (Shapiro-Wilk p={shapiro_p:.4f})",
        xaxis_title="Theoretical quantiles",
        yaxis_title="Residual quantiles",
        height=320,
    )

    resid_fig = go.Figure()
    resid_fig.add_trace(go.Scatter(x=fitted, y=residuals, mode="markers", name="Residuals"))
    resid_fig.add_hline(y=0, line_dash="dash", line_color="gray")
    resid_fig.update_layout(title=f"Residuals vs fitted — {method}", xaxis_title="Fitted", yaxis_title="Residual", height=320)

    acf_fig = _acf_plotly(residuals, title=f"Residual ACF — {method}", max_lag=min(30, max(5, len(residuals) // 3)), adjust_slow_acf=False)
    return {
        "residuals": residuals,
        "fitted": fitted,
        "qq_fig": qq_fig,
        "resid_fig": resid_fig,
        "acf_fig": acf_fig,
        "shapiro_stat": float(shapiro_stat),
        "shapiro_p": float(shapiro_p),
        "qq_r": float(r_val),
    }


def _estimate_effective_k(method, model_params, n_features, n_selected_methods):
    """Approximate model complexity for information criteria benchmarking."""
    if method in ("Linear Regression", "Bayesian Linear Regression"):
        return int(n_features) + 1
    # Tree/boosting: using n_estimators as literal parameter count makes AIC penalties enormous vs linear models
    # and misleading when comparing rows. Use a capped effective-parameter proxy for exploratory IC display only.
    if method == "Random Forest":
        return max(5, min(int(n_features) + 8, 45))
    if method == "Gradient Boosting":
        return max(5, min(int(n_features) + 8, 45))
    if method == "XGBoost":
        return max(5, min(int(n_features) + 8, 45))
    if method == "Prophet":
        return 10
    if method == "FBLiR":
        return int(n_features) + 3
    if method == "LLM Forecaster":
        return max(3, int(n_features) + 1)
    if method == "Ensemble":
        return max(1, int(n_selected_methods))
    return max(1, int(n_features))


def _aic_bic_from_fit(y_true, fitted, k):
    """Compute AIC/BIC from residual variance approximation."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    fitted = np.asarray(fitted, dtype=float).ravel()
    n = min(len(y_true), len(fitted))
    if n < 3:
        return np.nan, np.nan
    err = y_true[-n:] - fitted[-n:]
    err = err[np.isfinite(err)]
    n = len(err)
    if n < 3:
        return np.nan, np.nan
    rss = float(np.sum(err ** 2))
    if rss <= 0:
        rss = 1e-12
    sigma2 = rss / n
    k = max(1, int(k))
    aic = n * np.log(sigma2) + 2 * k
    bic = n * np.log(sigma2) + k * np.log(n)
    return float(aic), float(bic)


def _best_forecast_method_by_information_criteria(
    fitted_by_method: dict,
    forecast_results: dict,
    y_true_diag: np.ndarray,
    model_params: dict,
    n_features: int,
) -> str:
    """
    Choose one method (excluding Ensemble) with lowest AIC, then BIC on in-sample fits.
    Falls back to the first non-Ensemble key in forecast_results if no finite IC scores.
    """
    y_true_diag = np.asarray(y_true_diag, dtype=float).ravel()
    n_sel = max(1, len([m for m in forecast_results.keys() if m != "Ensemble"]))
    scored: list[tuple[str, float, float]] = []
    for method, fitted in fitted_by_method.items():
        if method == "Ensemble":
            continue
        if fitted is None:
            continue
        fit_arr = np.asarray(fitted, dtype=float).ravel()
        k_eff = _estimate_effective_k(
            method,
            model_params,
            n_features=int(n_features),
            n_selected_methods=n_sel,
        )
        aic_val, bic_val = _aic_bic_from_fit(y_true_diag, fit_arr, k_eff)
        if np.isfinite(aic_val) and np.isfinite(bic_val):
            scored.append((method, float(aic_val), float(bic_val)))
    if scored:
        scored.sort(key=lambda t: (t[1], t[2]))
        return scored[0][0]
    for m in forecast_results.keys():
        if m != "Ensemble":
            return m
    return next(iter(forecast_results.keys()))


def _gdf_forecast_day_spatial_pattern(
    valid_data: gpd.GeoDataFrame,
    forecast_target: str,
    fc_date,
    forecast_aoi_mean: float,
    historical_doy_window: int = 7,
) -> gpd.GeoDataFrame | None:
    """
    Point-level field for a forecast day: forecast AOI mean plus historical spatial anomalies
    from a day-of-year seasonal window (±window) averaged at each grid location.
    """
    if valid_data is None or valid_data.empty:
        return None
    fc_ts = pd.Timestamp(fc_date).normalize()
    vd = valid_data.copy()
    vd["_dts"] = pd.to_datetime(vd["date"], errors="coerce")
    vd = vd[vd["_dts"].notna()].copy()
    if vd.empty:
        return None
    vd["_doy"] = vd["_dts"].dt.dayofyear.astype(int)
    doy_t = int(fc_ts.dayofyear)
    doy = vd["_doy"].to_numpy(dtype=int)
    diff = np.abs(doy - doy_t)
    diff = np.minimum(diff, np.maximum(1, 366 - diff))
    mask = diff <= int(historical_doy_window)
    sub = vd.loc[mask].copy()
    if len(sub) < max(8, len(vd) // 200):
        sub = vd.copy()
    gx = np.round(sub.geometry.x.astype(float), 5)
    gy = np.round(sub.geometry.y.astype(float), 5)
    sub["_gx"] = gx
    sub["_gy"] = gy
    grp = sub.groupby(["_gx", "_gy"], as_index=False)[forecast_target].mean()
    try:
        geom = gpd.points_from_xy(grp["_gx"], grp["_gy"], crs=valid_data.crs)
        out = gpd.GeoDataFrame(grp, geometry=geom, crs=valid_data.crs)
    except Exception:
        return None
    vals = out[forecast_target].to_numpy(dtype=float)
    mu_spatial = float(np.nanmean(vals))
    if not np.isfinite(mu_spatial):
        mu_spatial = 0.0
    out[forecast_target] = float(forecast_aoi_mean) + (out[forecast_target].astype(float) - mu_spatial)
    out["date"] = fc_ts
    return out


def _iwfr_forecast_run_signature(
    bounds,
    start_date,
    end_date,
    forecast_target,
    selected_features,
    forecast_methods,
    forecast_horizon,
    target_smoothing_method,
    target_smoothing_window,
    log_transform_target,
    model_params,
    gdf_aoi,
):
    """Stable hash so we skip refitting when only lightweight widgets (e.g. heatmap date) change."""
    mp = json.dumps(model_params or {}, sort_keys=True, default=str)
    try:
        dmin = pd.Timestamp(gdf_aoi["date"].min())
        dmax = pd.Timestamp(gdf_aoi["date"].max())
        nx = float(np.nansum(gdf_aoi.geometry.x.values))
        ny = float(np.nansum(gdf_aoi.geometry.y.values))
    except Exception:
        dmin, dmax, nx, ny = ("", "", 0.0, 0.0)
    tup = (
        tuple(bounds) if bounds is not None else (),
        str(start_date),
        str(end_date),
        str(forecast_target),
        tuple(sorted(selected_features or [])),
        tuple(sorted(forecast_methods or [])),
        int(forecast_horizon),
        str(target_smoothing_method),
        int(target_smoothing_window),
        bool(log_transform_target),
        mp,
        int(len(gdf_aoi)),
        str(dmin),
        str(dmax),
        round(nx, 4),
        round(ny, 4),
    )
    return hashlib.sha256(repr(tup).encode("utf-8")).hexdigest()


def _plotly_to_png_bytes(fig):
    try:
        width = 1400
        height = 820
        try:
            if hasattr(fig, "layout") and getattr(fig.layout, "width", None):
                width = max(1200, int(fig.layout.width))
            if hasattr(fig, "layout") and getattr(fig.layout, "height", None):
                height = max(700, int(fig.layout.height))
        except Exception:
            pass
        return fig.to_image(format="png", width=width, height=height, scale=2)
    except Exception:
        return None


def _spatial_grid_plotly(values_grid, bounds, title, colorscale="Blues"):
    """Static spatial heatmap figure for PDF export."""
    try:
        minx, miny, maxx, maxy = bounds
        fig = go.Figure(
            data=go.Heatmap(
                z=np.asarray(values_grid, dtype=float),
                colorscale=colorscale,
                colorbar=dict(title="Intensity"),
            )
        )
        fig.update_layout(
            title=title,
            height=480,
            xaxis_title=f"Longitude ({minx:.2f} to {maxx:.2f})",
            yaxis_title=f"Latitude ({miny:.2f} to {maxy:.2f})",
            margin=dict(t=60, b=45, l=55, r=25),
        )
        return fig
    except Exception:
        return None


def _build_forecast_pdf_report_bytes(payload):
    """Create a polished PDF report using reportlab (with fallback)."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            SimpleDocTemplate,
            Paragraph,
            Spacer,
            Image as RLImage,
            Table,
            TableStyle,
            PageBreak,
        )
    except Exception:
        # Fallback path for environments where reportlab wheels are unavailable.
        return _build_forecast_pdf_report_bytes_pillow(payload)
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=1.5 * cm,
        rightMargin=1.5 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
    )
    styles = getSampleStyleSheet()
    story = []

    def add_title(text):
        story.append(Paragraph(f"<b>{html_module.escape(str(text))}</b>", styles["Title"]))
        story.append(Spacer(1, 0.25 * cm))

    def add_para(text):
        story.append(Paragraph(html_module.escape(str(text)).replace("\n", "<br/>"), styles["BodyText"]))
        story.append(Spacer(1, 0.18 * cm))

    def add_plot(fig, caption):
        png = _plotly_to_png_bytes(fig)
        if png is None:
            add_para(f"[Plot unavailable: {caption}]")
            return
        story.append(Paragraph(f"<b>{html_module.escape(caption)}</b>", styles["Heading4"]))
        img = RLImage(BytesIO(png))
        img.drawWidth = doc.width
        img.drawHeight = doc.width * 0.56
        story.append(img)
        story.append(Spacer(1, 0.22 * cm))

    add_title(f"{IWFR_DISPLAY_NAME} Forecast Report")
    if payload.get("plain_summary"):
        add_para(payload["plain_summary"])
    if payload.get("insights_md"):
        add_para("Key insights:")
        for ln in str(payload["insights_md"]).splitlines():
            if ln.strip():
                add_para(ln.lstrip("- ").strip())

    summary_rows = payload.get("summary_rows") or []
    if summary_rows:
        story.append(Paragraph("<b>Summary statistics</b>", styles["Heading3"]))
        headers = ["Series", "Mean", "Min", "Max", "Std"]
        data = [headers]
        for r in summary_rows:
            data.append(
                [
                    str(r.get("Series", "")),
                    f"{float(r.get('Mean', np.nan)):.2f}",
                    f"{float(r.get('Min', np.nan)):.2f}",
                    f"{float(r.get('Max', np.nan)):.2f}",
                    f"{float(r.get('Std', np.nan)):.2f}",
                ]
            )
        tbl = Table(data, repeatRows=1, colWidths=[doc.width * 0.42, doc.width * 0.145, doc.width * 0.145, doc.width * 0.145, doc.width * 0.145])
        tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF7")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                    ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B8C2D3")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ]
            )
        )
        story.append(tbl)
        story.append(Spacer(1, 0.35 * cm))

    add_plot(payload.get("forecast_fig"), "Forecast time series")
    add_plot(payload.get("spatial_fig"), "Spatial heatmap snapshot")
    add_plot(payload.get("hist_acf_fig"), "Historical ACF")
    add_plot(payload.get("hist_dist_fig"), "Historical distribution")

    residuals = payload.get("residual_diagnostics") or {}
    if residuals:
        story.append(PageBreak())
        story.append(Paragraph("<b>Residual diagnostics</b>", styles["Heading2"]))
        story.append(Spacer(1, 0.2 * cm))
        for method, diag in residuals.items():
            story.append(Paragraph(f"<b>{html_module.escape(method)}</b>", styles["Heading3"]))
            if "error" in diag:
                add_para(diag["error"])
                continue
            add_para(
                f"Shapiro-Wilk: W={diag.get('shapiro_stat', np.nan):.4f}, "
                f"p={diag.get('shapiro_p', np.nan):.4f}"
            )
            add_plot(diag.get("qq_fig"), f"{method} — QQ plot")
            add_plot(diag.get("resid_fig"), f"{method} — Residuals vs fitted")
            add_plot(diag.get("acf_fig"), f"{method} — Residual ACF")

    doc.build(story)
    buf.seek(0)
    return buf.getvalue(), None


def _build_forecast_pdf_report_bytes_pillow(payload):
    """Fallback PDF builder using Pillow only (works when reportlab isn't available)."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return None, "PDF export requires reportlab or pillow (pip install reportlab pillow)."

    # A4-ish page at ~150 DPI
    page_w, page_h = 1240, 1754
    margin = 35
    line_h = 22
    pages = []
    page = Image.new("RGB", (page_w, page_h), "white")
    draw = ImageDraw.Draw(page)
    font = ImageFont.load_default()
    y = margin

    def flush_page():
        nonlocal page, draw, y
        pages.append(page)
        page = Image.new("RGB", (page_w, page_h), "white")
        draw = ImageDraw.Draw(page)
        y = margin

    def write_line(txt):
        nonlocal y
        if y > page_h - margin - line_h:
            flush_page()
        draw.text((margin, y), str(txt)[:150], fill="black", font=font)
        y += line_h

    def add_plot(fig, title):
        nonlocal y
        png = _plotly_to_png_bytes(fig)
        if png is None:
            write_line(f"[Plot unavailable: {title}]")
            return
        try:
            img = Image.open(BytesIO(png)).convert("RGB")
        except Exception:
            write_line(f"[Plot decode failed: {title}]")
            return
        max_w = page_w - 2 * margin
        max_h = 360
        img.thumbnail((max_w, max_h))
        if y + 22 + img.height > page_h - margin:
            flush_page()
        write_line(title)
        page.paste(img, (margin, y))
        y += img.height + 18

    write_line(f"{IWFR_DISPLAY_NAME} Forecast Report")
    write_line("-" * 40)
    for ln in str(payload.get("plain_summary", "")).split("\n"):
        if ln.strip():
            write_line(ln.strip())
    write_line("")
    write_line("Summary statistics:")
    for row in payload.get("summary_rows", []):
        write_line(
            f"{row.get('Series','')}: mean={row.get('Mean', np.nan):.2f}, min={row.get('Min', np.nan):.2f}, "
            f"max={row.get('Max', np.nan):.2f}, std={row.get('Std', np.nan):.2f}"
        )

    if payload.get("forecast_fig") is not None:
        add_plot(payload["forecast_fig"], "Forecast time series")
    if payload.get("hist_acf_fig") is not None:
        add_plot(payload["hist_acf_fig"], "Historical ACF")
    if payload.get("hist_dist_fig") is not None:
        add_plot(payload["hist_dist_fig"], "Historical distribution")

    for method, diag in (payload.get("residual_diagnostics") or {}).items():
        write_line("")
        write_line(f"Residual diagnostics — {method}")
        if "error" in diag:
            write_line(f"  {diag['error']}")
            continue
        write_line(
            f"  Shapiro-Wilk: W={diag.get('shapiro_stat', np.nan):.4f}, p={diag.get('shapiro_p', np.nan):.4f}"
        )
        add_plot(diag.get("qq_fig"), f"{method}: QQ plot")
        add_plot(diag.get("resid_fig"), f"{method}: residuals vs fitted")
        add_plot(diag.get("acf_fig"), f"{method}: residual ACF")

    pages.append(page)
    out = BytesIO()
    try:
        pages[0].save(out, format="PDF", save_all=True, append_images=pages[1:])
    except Exception as e:
        return None, f"Pillow PDF generation failed: {e}"
    out.seek(0)
    return out.getvalue(), None


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

    # Recursive linear models can extrapolate without bound in model space (e.g. ln(1+y)).
    # Small upward drift in z becomes enormous after expm1 when plotting on the original scale.
    y_fit = np.asarray(y, dtype=float).ravel()
    y_fit = y_fit[np.isfinite(y_fit)]
    train_z_min = float(np.min(y_fit)) if len(y_fit) else -np.inf
    train_z_max = float(np.max(y_fit)) if len(y_fit) else np.inf
    z_span = max(train_z_max - train_z_min, 1e-6)
    # Allow modest extrapolation beyond the training band but block pathological blow-ups.
    linear_recursive_margin = float(min(1.0, 0.2 + 0.35 * z_span))
    linear_z_lo = train_z_min - linear_recursive_margin
    linear_z_hi = train_z_max + linear_recursive_margin

    if method == "Linear Regression":
        # L2 shrinkage reduces coefficient blow-ups in recursive multi-step forecasts.
        model = Ridge(alpha=1.5, random_state=42)
        model.fit(X_scaled, y)

        def predict_one(row_scaled_df):
            return float(model.predict(row_scaled_df)[0])

    elif method == "Bayesian Linear Regression":
        mp = dict(model_params.get("Bayesian Linear Regression") or {})
        if mp.pop("auto_tune", False):
            tuned, tune_note = _auto_tune_blir_params(
                X_scaled, y, holdout_days=int(mp.pop("holdout_days", 28))
            )
            mp.update(tuned)
            if tune_note:
                st.session_state["_iwfr_last_blir_tune"] = tune_note
        model = _build_blir_regressor(mp)
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
        params = dict(model_params.get("FBLiR") or {})
        X_raw = df_ml[feature_names]
        blr_mp = dict(model_params.get("Bayesian Linear Regression") or {})
        blir_core = _build_blir_regressor(blr_mp)
        blir_core.fit(X_scaled, y)
        if params.pop("auto_tune", False):
            tuned, tune_note = _auto_tune_fblir_params(
                X_scaled,
                y,
                X_raw,
                params,
                holdout_days=int(params.pop("holdout_days", 21)),
                blir_model=blir_core,
            )
            params.update(tuned)
            if tune_note:
                st.session_state["_iwfr_last_fblir_tune"] = tune_note
        adapt = int(params.get("adapt_steps", 200))
        burnin = int(params.get("burnin_steps", 200))
        if FuzzyBayesianRegression is not None:
            model = _build_fblir_regressor(params)
            _fblir_fit_model(model, X_scaled, y, X_raw, blir_model=blir_core)
        else:
            split_idx = int(len(X_scaled) * 0.8)
            X_train_f = X_scaled.iloc[:split_idx]
            y_train_f = y.iloc[:split_idx]
            X_val_f = X_scaled.iloc[split_idx:]
            y_val_f = y.iloc[split_idx:]
            X_train_r = X_raw.iloc[:split_idx]
            base_samples = min(max(100, adapt + burnin), 1500)
            model = FuzzyBayesianRegressionTuned(n_samples=base_samples, use_quadratic=True)
            try:
                model.fit(X_train_f, y_train_f, X_val_f, y_val_f, input_prescaled=True)
                model._iwfr_uses_prescaled = True
            except TypeError:
                model.fit(X_train_r, y_train_f, X_raw.iloc[split_idx:], y_val_f)
                model._iwfr_uses_prescaled = False
    else:
        raise ValueError(f"Unknown iterative ML method: {method}")

    is_fblir = method == "FBLiR"
    fblir_uses_backbone = is_fblir and getattr(model, "_iwfr_fblir_backbone", False)

    for step_idx in range(forecast_horizon):
        next_date = current_data["date"].iloc[-1] + timedelta(days=1)
        seasonal_ref = _seasonal_target_reference(df_ml, forecast_target, next_date)

        last_row = current_data.iloc[[-1]][feature_names]
        last_row = pd.DataFrame(
            sanitize_float_matrix(last_row.values),
            columns=feature_names,
        )
        last_row_scaled = pd.DataFrame(
            scaler.transform(last_row),
            columns=feature_names,
        )
        if is_fblir:
            pred = float(
                _fblir_predict_values(model, last_row_scaled, last_row, linear_only=fblir_uses_backbone)[0]
            )
        else:
            pred = predict_one(last_row_scaled)

        if method == "Linear Regression":
            tail_z = current_data[forecast_target].tail(21)
            anchor = float(np.nanmean(tail_z)) if len(tail_z) else pred
            pred = 0.86 * pred + 0.14 * anchor
            pred = float(np.clip(pred, linear_z_lo, linear_z_hi))
        elif method in ("Bayesian Linear Regression", "FBLiR"):
            if seasonal_ref is not None:
                pred = 0.58 * pred + 0.42 * seasonal_ref
            tail_z = current_data[forecast_target].tail(14)
            anchor = float(np.nanmean(tail_z)) if len(tail_z) else pred
            pred = 0.92 * pred + 0.08 * anchor
            pred = float(np.clip(pred, linear_z_lo, linear_z_hi))
        predictions.append(pred)

        new_date = next_date
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
                deterministic=True,
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
                std_val = np.std(recent_values) if len(recent_values) > 1 else 0.0
                new_row[std_col] = std_val
        current_data = pd.concat([current_data, pd.DataFrame([new_row])], ignore_index=True)

    return np.array(predictions, dtype=float)


# Initialize session state
if 'map_type' not in st.session_state:
    st.session_state.map_type = "Street"
if 'forecast_insights_md' not in st.session_state:
    st.session_state.forecast_insights_md = None
if 'forecast_report_payload' not in st.session_state:
    st.session_state.forecast_report_payload = None

# Sidebar controls
st.sidebar.header("⚙️ Controls")

# Data source selection
data_source = st.sidebar.radio(
    "Data Source",
    [DATA_SOURCE_QUICK_HISTORICAL, "Forecasting"],
)
if data_source != "Forecasting":
    st.session_state.forecast_insights_md = None

# Date range selection
if data_source == DATA_SOURCE_QUICK_HISTORICAL:
    select_last_30_hist = st.sidebar.checkbox(
        "Select last 30 days",
        value=False,
        help="Sets the historical window to the most recent 30 available days (end date lagged ~3 days like NASA latency handling).",
    )
    if select_last_30_hist:
        end_date = (datetime.now() - timedelta(days=3)).date()
        start_date = end_date - timedelta(days=29)
        st.sidebar.caption(f"📅 Date range: **{start_date}** → **{end_date}**")
    else:
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime(2024, 1, 1),
                max_value=datetime.now() - timedelta(days=3),
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime(2024, 1, 30),
                max_value=datetime.now() - timedelta(days=3),
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
    available_forecast_methods = ["Linear Regression", "Bayesian Linear Regression", "Random Forest", "Gradient Boosting", "XGBoost",
                                   "Prophet", "LLM Forecaster", "Ensemble"]
    
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
    
    forecast_horizon = st.sidebar.number_input(
        "Forecast Horizon (in days)",
        min_value=7,
        max_value=730,
        value=30,
        step=1
    )
    if forecast_horizon <= LONG_FORECAST_HORIZON_WARNING_DAYS:
        st.session_state.pop("long_forecast_horizon_dialog_dismissed", None)
    else:
        st.sidebar.caption(
            f"Horizons over **{LONG_FORECAST_HORIZON_WARNING_DAYS}** days: forecast precision usually "
            "**decreases** the further ahead you predict."
        )
        if not st.session_state.get("long_forecast_horizon_dialog_dismissed"):
            _long_forecast_horizon_dialog(forecast_horizon, LONG_FORECAST_HORIZON_WARNING_DAYS)

    with st.sidebar.expander("📐 Transformation & Smoothing Options", expanded=False):
        log_transform_target = st.checkbox(
            "Log-transform target (ln(1+y), pre-fit)",
            value=False,
            help=(
                "Fit models on ln(1+y) after optional smoothing; residual diagnostics use this scale. "
                "Forecast time series and CSV exports use the original scale."
            ),
        )
        target_smoothing_method = st.selectbox(
            "Target smoothing (pre-fit)",
            options=[
                "None (no smoothing)",
                "Rolling mean",
                "Rolling median",
                "Exponential moving average (EMA)",
                "Seasonal smoothing (STL)",
            ],
            index=0,
            help=(
                "Smooth the target before fitting. EMA reacts faster to shifts than rolling windows. "
                "STL fits annual seasonality on long daily series (~2+ years recommended)."
            ),
            key="target_smooth_method_outer",
        )
        target_smoothing_window = 7
        if target_smoothing_method == "Seasonal smoothing (STL)":
            st.caption(
                "STL uses period=365 days. Needs at least ~730 consecutive calendar days in range "
                "(gaps are linearly interpolated)."
            )
        elif target_smoothing_method != "None (no smoothing)":
            target_smoothing_window = st.number_input(
                "Smoothing window (days)",
                min_value=2,
                max_value=60,
                value=7,
                step=1,
                help="Rolling window for mean/median; EMA span (typical effective memory ~this many days).",
                key="target_smooth_window_outer",
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

        if "Bayesian Linear Regression" in forecast_methods:
            st.markdown("**Bayesian Linear Regression**")
            st.markdown("**ARD prior hyperparameters (sklearn):**")
            blr_alpha1 = st.number_input(
                "alpha_1",
                min_value=1e-10,
                max_value=1.0,
                value=1e-6,
                format="%.2e",
                key="blr_alpha1",
                help="Shape parameter for the Gamma prior on inverse noise precision.",
            )
            blr_alpha2 = st.number_input(
                "alpha_2",
                min_value=1e-10,
                max_value=1.0,
                value=1e-6,
                format="%.2e",
                key="blr_alpha2",
                help="Rate parameter for the Gamma prior on inverse noise precision.",
            )
            blr_lambda1 = st.number_input(
                "lambda_1",
                min_value=1e-10,
                max_value=1.0,
                value=1e-6,
                format="%.2e",
                key="blr_lambda1",
                help="Shape parameter for the Gamma prior on weight precisions.",
            )
            blr_lambda2 = st.number_input(
                "lambda_2",
                min_value=1e-10,
                max_value=1.0,
                value=1e-6,
                format="%.2e",
                key="blr_lambda2",
                help="Rate parameter for the Gamma prior on weight precisions.",
            )
            model_params["Bayesian Linear Regression"] = {
                "alpha_1": float(blr_alpha1),
                "alpha_2": float(blr_alpha2),
                "lambda_1": float(blr_lambda1),
                "lambda_2": float(blr_lambda2),
                "auto_tune": st.checkbox(
                    "Auto-tune on recent holdout (28 days)",
                    value=True,
                    key="blr_auto_tune",
                    help="Search ARD priors on the last ~28 days before forecasting.",
                ),
                "holdout_days": 28,
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
        
        if 'FBLiR' in forecast_methods and FBLIR_AVAILABLE:
            st.markdown("**FBLiR (Fuzzy Bayesian Linear Regression)**")

            st.markdown("**Bayesian prior hyperparameters:**")
            fblir_tau = st.number_input(
                "tau (prior std for slopes β_j)",
                min_value=1e-4,
                max_value=100.0,
                value=1.0,
                step=0.1,
                format="%.4f",
                key="fblir_tau",
                help="Prior: β_j ~ N(0, τ²) for j = 1, …, p* (model-fit layer, before GFN fuzzification).",
            )
            fblir_sigma0 = st.number_input(
                "sigma_0_squared (prior variance for intercept β_0)",
                min_value=1e-6,
                max_value=100.0,
                value=1.0,
                step=0.1,
                format="%.4f",
                key="fblir_sigma0",
                help="Prior: β_0 ~ N(0, σ₀²).",
            )
            
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
            fblir_n_chains = st.number_input("N_chains", min_value=1, max_value=10, value=2, step=1, key="fblir_n_chains",
                                            help="Number of MCMC chains for Bayesian inference")
            fblir_adapt_steps = st.number_input("adapt_steps", min_value=10, max_value=1000, value=200, step=10, key="fblir_adapt_steps",
                                               help="Number of adaptation steps for MCMC")
            fblir_burnin_steps = st.number_input("burnin_steps", min_value=10, max_value=1000, value=200, step=10, key="fblir_burnin_steps",
                                                 help="Number of burn-in steps for MCMC")
            fblir_thinning_steps = st.number_input("thinning_steps", min_value=1, max_value=50, value=7, step=1, key="fblir_thinning_steps",
                                                   help="Thinning interval for MCMC samples")
            
            model_params['FBLiR'] = {
                'tau': float(fblir_tau),
                'sigma_0_squared': float(fblir_sigma0),
                'm': fblir_m,
                'k': fblir_k,
                'fuzzification_factor': fblir_fuzz,
                'symmetry_threshold': fblir_symmetry,
                'N_chains': fblir_n_chains,
                'adapt_steps': fblir_adapt_steps,
                'burnin_steps': fblir_burnin_steps,
                'thinning_steps': fblir_thinning_steps,
                'auto_tune': st.checkbox(
                    "Auto-tune on recent holdout (21 days)",
                    value=True,
                    key="fblir_auto_tune",
                    help="Grid-search m, fuzzification factor, and tau on the last ~21 days.",
                ),
                'holdout_days': 21,
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
            'response': """Hello! 👋 I'm your IWFR Assistant. I can help you learn how to use the app!
            
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
6. Use **Transformation & Smoothing Options** for optional smoothing / log transform (fits use this; plots export original scale)
7. Configure model parameters if needed
8. Draw an AOI on the map — forecasting runs when viewing Spatial Distribution with Time Series

**Forecast Methods:**
- **Random Forest**: Fast, good for most cases
- **XGBoost**: Powerful gradient boosting
- **FBLiR**: Uncertainty-aware, best for noisy data (takes 6-10 min)
- **Prophet**: Time series forecasting
- **Ensemble**: Combines all selected methods

**Note:** Forecasting takes 20-40 seconds (FBLiR: 6-10 minutes)

**Scale:** Original-series preprocessing applies **before fitting**; forecast time series use **original units**."""
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

**Access:** Sidebar → **📐 Transformation & Smoothing Options** (smoothing / log) and **⚙️ Configure Model Parameters**

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
1. **Quick Historical Data:** NASA POWER records over your chosen history window (optional **last 30 days** shortcut)
2. **Forecasting:** Same retrieval plus forecasting, diagnostics, and exports

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

    @st.dialog("🤖 IWFR Assistant", width="large")
    def faim_assistant_dialog():
        st.caption(f"Tips and how-to for {IWFR_DISPLAY_NAME}. Close with the dialog X when you are done.")
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
                        "content": "👋 Hello! I'm your IWFR Assistant. I can help you learn how to use the app. Ask me anything!",
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
            "content": "👋 Hello! I'm your IWFR Assistant. I can help you learn how to use the app. Ask me anything!",
        }
    ]

st.sidebar.markdown("---")
if st.sidebar.button("🤖 AI Assistant", use_container_width=True, key="open_ai_assistant"):
    faim_assistant_dialog()

if st.sidebar.button("📖 Guide Helper", use_container_width=True, key="open_guide_helper"):
    faim_howto_dialog()


def _reset_faim_app() -> None:
    """Clear session state so the app returns to a clean start (widgets reset on next run)."""
    for k in list(st.session_state.keys()):
        try:
            del st.session_state[k]
        except Exception:
            pass
    st.rerun()


if st.sidebar.button("🔄 Reset App", use_container_width=True, key="reset_faim_app"):
    _reset_faim_app()

# Main content — full width (maps first, then visualizations below)
main_viz = st.container()
VIZ_FOLIUM_CELL_W = 720
VIZ_FOLIUM_CELL_H = 440
# Matched height/margins for side-by-side ACF + distribution in forecast / quick historical views.
PAIR_DIAG_PLOT_HEIGHT = 420
PAIR_DIAG_PLOT_MARGIN = dict(t=50, b=40, l=55, r=25)

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
land_mask_geom_for_aoi = None
land_geom_global = load_global_land_geometry()

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
        aoi_eval = evaluate_aoi_land_coverage(bounds, land_geom_global)
        st.session_state.aoi_land_ratio = float(aoi_eval["land_ratio"])
        st.session_state.aoi_land_geom = aoi_eval["land_geom_aoi"]
        if aoi_eval["is_non_land"]:
            st.error("You have selected a non-land mass. Please select an appropriate land mass.")
            st.session_state.show_selection_map = True
            st.stop()
        if aoi_eval["has_partial_ocean"]:
            st.warning("data and forecasting will be carried out only for the land mass area selected")
        land_mask_geom_for_aoi = aoi_eval["land_geom_aoi"]

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
if "aoi_land_geom" in st.session_state:
    land_mask_geom_for_aoi = st.session_state.aoi_land_geom

# Re-validate AOI against land mask on every run so ocean-only selections are always blocked.
if processed_bounds is not None:
    if land_geom_global is None:
        st.error("You have selected a non-land mass. Please select an appropriate land mass.")
        st.session_state.show_selection_map = True
        st.stop()
    aoi_eval = evaluate_aoi_land_coverage(processed_bounds, land_geom_global)
    land_mask_geom_for_aoi = aoi_eval["land_geom_aoi"]
    st.session_state.aoi_land_ratio = float(aoi_eval["land_ratio"])
    st.session_state.aoi_land_geom = land_mask_geom_for_aoi
    if aoi_eval["is_non_land"]:
        st.error("You have selected a non-land mass. Please select an appropriate land mass.")
        st.session_state.show_selection_map = True
        st.stop()
    if aoi_eval["has_partial_ocean"] and st.session_state.get("show_selection_map", False):
        st.warning("data and forecasting will be carried out only for the land mass area selected")

# Show spatial distribution content when toggled and we have AOI
if not st.session_state.show_selection_map and processed_bounds:
    bounds = processed_bounds

    with main_viz:
        st.info(f"Viewing AOI: {bounds[1]:.3f}°N to {bounds[3]:.3f}°N, {bounds[0]:.3f}°E to {bounds[2]:.3f}°E")
            
        # Create placeholder for spatial distribution maps
        spatial_placeholder = st.empty()
            
        # Load/fetch data based on source
        if data_source == DATA_SOURCE_QUICK_HISTORICAL:
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
                            status_text.text(
                                f"Fetching data for point {current_point}/{total_points} ({lat:.3f}°, {lon:.3f}°)"
                            )

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
                        st.info(
                            f"📊 Data quality: {valid_data_count}/{len(gdf_filtered)} records have valid {selected_metric} data"
                        )
                    else:
                        st.error("❌ Failed to fetch NASA POWER data. Please try again.")
                        gdf_filtered = None

                except Exception as e:
                    st.error(f"❌ Error fetching NASA POWER data: {str(e)}")
                    st.info("💡 Check your internet connection and date range, then retry.")
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

            mask = (
                (gdf_filtered.geometry.x >= bounds[0]) & (gdf_filtered.geometry.x <= bounds[2]) &
                (gdf_filtered.geometry.y >= bounds[1]) & (gdf_filtered.geometry.y <= bounds[3])
            )
            gdf_aoi = gdf_filtered[mask]
            gdf_aoi = filter_points_to_land(gdf_aoi, land_mask_geom_for_aoi)

            if not gdf_aoi.empty:
                if forecast_mode and selected_features:
                    # Prepare data for forecasting
                    daily_avg = gdf_aoi.groupby('date').agg({
                        forecast_target: 'mean',
                        **{feat: 'mean' for feat in selected_features if feat in gdf_aoi.columns}
                    }).reset_index()
                                    
                    daily_avg = daily_avg.dropna(subset=[forecast_target])
                    daily_avg_model = daily_avg.copy()
                    smoothed_target_values, smoothing_note = _apply_target_smoothing(
                        daily_avg_model[forecast_target].astype(float).values,
                        target_smoothing_method,
                        target_smoothing_window,
                        dates=daily_avg_model["date"].values,
                    )
                    daily_avg_model[forecast_target] = smoothed_target_values
                    if smoothing_note:
                        st.caption(smoothing_note)
                    model_target_vals, log_note = _forward_target_model_scale(
                        daily_avg_model[forecast_target].astype(float).values,
                        log_transform_target,
                    )
                    daily_avg_model[forecast_target] = model_target_vals
                    if log_note:
                        st.caption(log_note)

                    if len(daily_avg) < 30:
                        st.error("⚠️ Insufficient historical data. Need at least 30 days.")
                    else:
                        run_sig = _iwfr_forecast_run_signature(
                            bounds,
                            start_date,
                            end_date,
                            forecast_target,
                            selected_features,
                            forecast_methods,
                            forecast_horizon,
                            target_smoothing_method,
                            target_smoothing_window,
                            log_transform_target,
                            model_params,
                            gdf_aoi,
                        )
                        fc_cached = st.session_state.get("_iwfr_fc")
                        use_cached_fc = (
                            fc_cached is not None
                            and fc_cached.get("sig") == run_sig
                            and fc_cached.get("forecast_results")
                        )

                        forecast_results = {}
                        forecast_results_display = {}

                        if use_cached_fc:
                            daily_avg = fc_cached["daily_avg"]
                            daily_avg_model = fc_cached["daily_avg_model"]
                            df_ml = fc_cached["df_ml"]
                            feature_names = fc_cached["feature_names"]
                            X_scaled = fc_cached["X_scaled"]
                            y = fc_cached["y"]
                            scaler = fc_cached["scaler"]
                            future_dates = fc_cached["future_dates"]
                            forecast_results = fc_cached["forecast_results"]
                            forecast_results_display = fc_cached["forecast_results_display"]
                            st.session_state.forecast_insights_md = fc_cached.get("forecast_insights_md")
                        else:
                            st.info("🔮 Generating forecast... This may take a moment.")
                            df_ml, feature_names = prepare_ml_features(
                                daily_avg_model,
                                forecast_target,
                                selected_features,
                                lag_days=7
                            )

                            if len(df_ml) < 20:
                                st.error(
                                    f"⚠️ Not enough data after feature engineering. Got {len(df_ml)} rows, need at least 20."
                                )
                                st.info(
                                    "💡 Try: (1) Using fewer features, (2) Longer historical period, "
                                    "or (3) Different forecast target"
                                )
                            else:
                                X = df_ml[feature_names]
                                y = df_ml[forecast_target]

                                X_values = sanitize_float_matrix(X.values)
                                X = pd.DataFrame(X_values, columns=feature_names, index=X.index)
                                y_values = sanitize_float_vector(y.values)
                                y = pd.Series(y_values, index=y.index, name=forecast_target)

                                scaler = StandardScaler()
                                X_scaled = pd.DataFrame(
                                    scaler.fit_transform(X),
                                    columns=X.columns,
                                    index=X.index
                                )

                                last_date = daily_avg["date"].max()
                                future_dates = pd.date_range(
                                    start=last_date + timedelta(days=1),
                                    periods=forecast_horizon,
                                    freq="D",
                                )

                                st.session_state.forecast_insights_md = None
                                forecast_results = {}

                                for method in forecast_methods:
                                    if method == "Ensemble":
                                        continue

                                    try:
                                        if method == "Prophet":
                                            df_prophet = daily_avg_model[["date", forecast_target]].copy()
                                            df_prophet.columns = ["ds", "y"]
                                            predictions = train_prophet(
                                                df_prophet, forecast_horizon, model_params.get("Prophet")
                                            )

                                        elif method == "LLM Forecaster":
                                            predictions = train_llm_horizon_forecast(
                                                daily_avg_model,
                                                forecast_target,
                                                forecast_horizon,
                                                future_dates,
                                                model_params.get("LLM Forecaster"),
                                            )

                                        else:
                                            fblir_status = None
                                            if method == "FBLiR":
                                                fblir_status = st.empty()
                                                fblir_status.info(
                                                    "🔄 FBLiR is training (single fit for the full horizon)..."
                                                )
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

                                if "Ensemble" in forecast_methods and len(forecast_results) > 0:
                                    ensemble_pred = np.mean(list(forecast_results.values()), axis=0)
                                    forecast_results["Ensemble"] = ensemble_pred

                                forecast_results_display = {
                                    m: _inverse_target_transform_predictions(pred, log_transform_target)
                                    for m, pred in forecast_results.items()
                                }

                                if forecast_results:
                                    tune_notes = []
                                    if st.session_state.get("_iwfr_last_blir_tune"):
                                        tune_notes.append(st.session_state.pop("_iwfr_last_blir_tune"))
                                    if st.session_state.get("_iwfr_last_fblir_tune"):
                                        tune_notes.append(st.session_state.pop("_iwfr_last_fblir_tune"))
                                    for note in tune_notes:
                                        st.caption(note)

                                    st.session_state.forecast_insights_md = generate_forecast_insights_markdown(
                                        forecast_results_display,
                                        daily_avg,
                                        forecast_target,
                                        future_dates,
                                    )
                                else:
                                    st.session_state.forecast_insights_md = None

                                if forecast_results:
                                    st.session_state["_iwfr_fc"] = {
                                        "sig": run_sig,
                                        "daily_avg": daily_avg,
                                        "daily_avg_model": daily_avg_model,
                                        "df_ml": df_ml,
                                        "feature_names": feature_names,
                                        "X_scaled": X_scaled,
                                        "y": y,
                                        "scaler": scaler,
                                        "future_dates": future_dates,
                                        "forecast_results": forecast_results,
                                        "forecast_results_display": forecast_results_display,
                                        "forecast_insights_md": st.session_state.get("forecast_insights_md"),
                                    }

                        if forecast_results:
                            # Display forecast visualization in spatial_placeholder (col_map)
                            with spatial_placeholder.container():
                                plain_summary = _simple_forecast_findings_summary(
                                    forecast_results_display,
                                    daily_avg[forecast_target].astype(float).values,
                                    forecast_target,
                                    future_dates,
                                )
                                _ins = st.session_state.get("forecast_insights_md")
                                if _ins:
                                    st.subheader("💡 Useful insights")
                                    if plain_summary:
                                        st.info(plain_summary)
                                    st.markdown(_ins)
                                    st.markdown("---")
                                y_hist = daily_avg[forecast_target].dropna().astype(float)
                                acf_fig_fc = None
                                if len(y_hist) > 14:
                                    acf_fig_fc = _acf_plotly(
                                        y_hist.values,
                                        title=f"ACF — {forecast_target} (historical AOI daily mean)",
                                        max_lag=min(40, max(5, len(y_hist) // 3)),
                                        height=PAIR_DIAG_PLOT_HEIGHT,
                                        margin=PAIR_DIAG_PLOT_MARGIN,
                                    )
                                metric_description_hist = AVAILABLE_PARAMETERS.get(
                                    forecast_target,
                                    "Fire Weather Index" if forecast_target == "FWI" else "Australian Fire Danger Rating" if forecast_target == "AFDR" else forecast_target,
                                )

                                y_true_diag = np.asarray(y, dtype=float)
                                residual_diagnostics = {}
                                fitted_by_method = {}
                                for method in forecast_results.keys():
                                    if method == "Ensemble":
                                        continue
                                    try:
                                        fitted = _fit_in_sample_predictions(
                                            method,
                                            X_scaled,
                                            y_true_diag,
                                            df_ml,
                                            forecast_target,
                                            model_params,
                                        )
                                        if fitted is None:
                                            residual_diagnostics[method] = {
                                                "error": "Residual diagnostics unavailable for this model."
                                            }
                                        else:
                                            fitted = np.asarray(fitted, dtype=float).ravel()
                                            fitted_by_method[method] = fitted
                                            residual_diagnostics[method] = _residual_diagnostic_bundle(
                                                y_true_diag, fitted, method
                                            )
                                    except Exception as diag_e:
                                        residual_diagnostics[method] = {"error": str(diag_e)}

                                if "Ensemble" in forecast_results:
                                    available = [fitted_by_method.get(m) for m in fitted_by_method if m != "Ensemble"]
                                    available = [np.asarray(a, dtype=float) for a in available if a is not None]
                                    if available:
                                        ens_fit = np.mean(np.vstack(available), axis=0)
                                        fitted_by_method["Ensemble"] = ens_fit
                                        residual_diagnostics["Ensemble"] = _residual_diagnostic_bundle(
                                            y_true_diag, ens_fit, "Ensemble"
                                        )
                                    else:
                                        residual_diagnostics["Ensemble"] = {
                                            "error": "No base-model fitted values available for ensemble residual diagnostics."
                                        }

                                best_spatial_method = _best_forecast_method_by_information_criteria(
                                    fitted_by_method,
                                    forecast_results,
                                    y_true_diag,
                                    model_params,
                                    len(feature_names),
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
                                    for idx, (method, predictions) in enumerate(forecast_results_display.items()):
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
                                    st.plotly_chart(fig, use_container_width=True, theme=None)

                                with row1c2:
                                    mask_aoi = (
                                        (gdf_filtered.geometry.x >= bounds[0]) & (gdf_filtered.geometry.x <= bounds[2]) &
                                        (gdf_filtered.geometry.y >= bounds[1]) & (gdf_filtered.geometry.y <= bounds[3])
                                    )
                                    gdf_spatial_data = gdf_filtered[mask_aoi]
                                    if not gdf_spatial_data.empty:
                                        valid_data = gdf_spatial_data[gdf_spatial_data[forecast_target].notna()]
                                        if not valid_data.empty:
                                            vf = valid_data.copy()
                                            vf["_dts"] = pd.to_datetime(vf["date"])
                                            sd_hist = pd.Timestamp(start_date)
                                            ed_hist = pd.Timestamp(end_date)
                                            in_rng = (vf["_dts"] >= sd_hist) & (vf["_dts"] <= ed_hist)
                                            avail_map_dates = sorted(vf.loc[in_rng, "date"].unique())
                                            if not avail_map_dates:
                                                avail_map_dates = sorted(vf["date"].unique())

                                            hist_ts_norm = sorted(
                                                {pd.Timestamp(d).normalize() for d in avail_map_dates}
                                            )
                                            fc_ts_norm = [
                                                pd.Timestamp(x).normalize()
                                                for x in pd.to_datetime(future_dates, errors="coerce")
                                            ]
                                            heatmap_date_options = sorted(set(hist_ts_norm) | set(fc_ts_norm))

                                            st.caption(
                                                f"Forecast days on the map use **{best_spatial_method}** "
                                                "(lowest in-sample **AIC**, then **BIC** among fitted models; Ensemble excluded). "
                                                "Values = that model’s AOI-mean forecast plus a **±7 day-of-year** historical spatial pattern."
                                            )
                                            chosen_ts = st.selectbox(
                                                "Heatmap date (historical or forecast)",
                                                options=heatmap_date_options,
                                                index=len(heatmap_date_options) - 1,
                                                format_func=lambda x: pd.Timestamp(x).strftime("%Y-%m-%d"),
                                                key="forecast_spatial_hist_date",
                                                help=(
                                                    f"Historical coverage inside {start_date}–{end_date}, "
                                                    "plus each day in the forecast horizon. Forecast maps use the best IC model’s daily AOI forecast "
                                                    "with typical same-season spatial variability from NASA grid points."
                                                ),
                                            )
                                            chosen_ts = pd.Timestamp(chosen_ts).normalize()
                                            fd_set = set(fc_ts_norm)
                                            is_forecast_heatmap = chosen_ts in fd_set

                                            if is_forecast_heatmap:
                                                st.subheader("Spatial — forecast heatmap")
                                                fc_idx = fc_ts_norm.index(chosen_ts)
                                                y_hat_day = float(
                                                    np.asarray(
                                                        forecast_results_display[best_spatial_method],
                                                        dtype=float,
                                                    )[fc_idx]
                                                )
                                                gdf_latest = _gdf_forecast_day_spatial_pattern(
                                                    valid_data,
                                                    forecast_target,
                                                    chosen_ts,
                                                    y_hat_day,
                                                )
                                            else:
                                                st.subheader("Spatial — historical heatmap")
                                                dmatch = pd.to_datetime(valid_data["date"], errors="coerce").dt.normalize()
                                                gdf_latest = valid_data[dmatch == chosen_ts].copy()

                                            hist_std = float(y_hist.std()) if len(y_hist) > 1 else 0.0
                                            if not np.isfinite(hist_std):
                                                hist_std = 0.0

                                            if gdf_latest is None or gdf_latest.empty:
                                                st.info("Insufficient spatial points for this date.")
                                            else:
                                                values_grid, heatmap_values = create_heatmap_data(
                                                    gdf_latest,
                                                    bounds,
                                                    forecast_target,
                                                    reference_std=hist_std,
                                                    land_geom_aoi=land_mask_geom_for_aoi,
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
                                                    popup_txt = (
                                                        f"Forecast: {chosen_ts.strftime('%Y-%m-%d')} · {best_spatial_method}"
                                                        if is_forecast_heatmap
                                                        else f"Historical: {chosen_ts.strftime('%Y-%m-%d')}"
                                                    )
                                                    folium.Rectangle(
                                                        bounds=[[bounds[1], bounds[0]], [bounds[3], bounds[2]]],
                                                        color="red",
                                                        weight=2,
                                                        fill_opacity=0.0,
                                                        popup=popup_txt,
                                                    ).add_to(m_forecast_spatial)
                                                    if is_forecast_heatmap:
                                                        metric_leg = (
                                                            f"{metric_description_hist} — {chosen_ts.strftime('%Y-%m-%d')} "
                                                            f"({best_spatial_method} forecast)"
                                                        )
                                                    else:
                                                        metric_leg = (
                                                            f"{metric_description_hist} — {chosen_ts.strftime('%Y-%m-%d')}"
                                                        )
                                                    heatmap_layer, legend_html = create_continuous_heatmap(
                                                        bounds,
                                                        values_grid,
                                                        opacity=heatmap_opacity,
                                                        metric_name=metric_leg,
                                                        color_scheme="red",
                                                    )
                                                    heatmap_layer.add_to(m_forecast_spatial)
                                                    m_forecast_spatial.get_root().html.add_child(folium.Element(legend_html))
                                                    st_folium(
                                                        m_forecast_spatial,
                                                        key=f"fc_spatial_{chosen_ts}_{int(is_forecast_heatmap)}_{best_spatial_method}",
                                                        width=VIZ_FOLIUM_CELL_W,
                                                        height=VIZ_FOLIUM_CELL_H,
                                                        returned_objects=[],
                                                    )
                                                    vmin_ext = float(np.nanmin(values_grid))
                                                    vmax_ext = float(np.nanmax(values_grid))
                                                    vrange_ext = vmax_ext - vmin_ext
                                                    st.markdown(
                                                        f"""
                                                        <div style="margin-top:6px;padding:8px 10px;border:1px solid #fecaca;border-radius:8px;background:#ffffff;">
                                                          <div style="font-size:12px;font-weight:700;text-align:center;margin-bottom:6px;color:#0f172a;">
                                                            {metric_leg}
                                                          </div>
                                                          <div style="height:12px;border-radius:6px;background:linear-gradient(to right,#FFF8F8,#FFCDD2,#E57373,#E53935,#C62828,#7F0000);"></div>
                                                          <div style="display:flex;justify-content:space-between;font-size:12px;margin-top:6px;color:#111827;font-weight:700;">
                                                            <span>Min: {vmin_ext:.2f}</span>
                                                            <span>Max: {vmax_ext:.2f}</span>
                                                            <span>Range: {vrange_ext:.2f}</span>
                                                          </div>
                                                        </div>
                                                        """,
                                                        unsafe_allow_html=True,
                                                    )
                                                    if is_forecast_heatmap:
                                                        st.caption(
                                                            f"{chosen_ts.strftime('%Y-%m-%d')} · {best_spatial_method} "
                                                            f"· {len(gdf_latest)} locations · seasonal spatial pattern"
                                                        )
                                                    else:
                                                        st.caption(f"{chosen_ts.strftime('%Y-%m-%d')} · {len(gdf_latest)} points")
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
                                        st.plotly_chart(acf_fig_fc, use_container_width=True, theme=None)
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
                                            height=PAIR_DIAG_PLOT_HEIGHT,
                                            margin=PAIR_DIAG_PLOT_MARGIN,
                                            autosize=True,
                                        )
                                        st.plotly_chart(dist_fig_fc, use_container_width=True, theme=None)
                                    else:
                                        st.info("No distribution to show.")

                                st.info(
                                    "Please note, modelling diagnostics (and therefore forecasting behaviour) can vary "
                                    "depending on area and date range selected, because these inputs affect training data."
                                )
                                st.subheader("Residual diagnostics")
                                st.caption("Open each model tab to inspect QQ plot, residual-vs-fitted, residual ACF, and residual time series.")
                                tab_names = list(residual_diagnostics.keys()) if residual_diagnostics else []
                                if tab_names:
                                    diag_tabs = st.tabs(tab_names)
                                    for t_i, m_name in enumerate(tab_names):
                                        with diag_tabs[t_i]:
                                            d = residual_diagnostics[m_name]
                                            if "error" in d:
                                                st.warning(d["error"])
                                            else:
                                                st.caption(f"Shapiro-Wilk W={d['shapiro_stat']:.4f}, p={d['shapiro_p']:.4f}")
                                                r1, r2 = st.columns(2)
                                                with r1:
                                                    st.plotly_chart(d["qq_fig"], use_container_width=True, theme=None)
                                                with r2:
                                                    st.plotly_chart(d["resid_fig"], use_container_width=True, theme=None)
                                                r3, r4 = st.columns(2)
                                                with r3:
                                                    st.plotly_chart(d["acf_fig"], use_container_width=True, theme=None)
                                                with r4:
                                                    residual_ts_fig = go.Figure()
                                                    residual_ts_fig.add_trace(go.Scatter(y=d["residuals"], mode="lines+markers", name="Residuals"))
                                                    residual_ts_fig.add_hline(y=0, line_dash="dash", line_color="gray")
                                                    residual_ts_fig.update_layout(
                                                        title=f"Residual time series — {m_name}",
                                                        xaxis_title="Index",
                                                        yaxis_title="Residual",
                                                        height=300,
                                                    )
                                                    st.plotly_chart(residual_ts_fig, use_container_width=True, theme=None)
                                else:
                                    st.info("Residual diagnostics unavailable for the selected models.")

                                summary_rows_fc = []
                                if len(y_hist):
                                    summary_rows_fc.append({
                                        "Series": "Historical (AOI daily mean)",
                                        "Mean": float(y_hist.mean()),
                                        "Min": float(y_hist.min()),
                                        "Max": float(y_hist.max()),
                                        "Std": float(y_hist.std()),
                                        "AIC (train)": np.nan,
                                        "BIC (train)": np.nan,
                                    })
                                for method, predictions in forecast_results_display.items():
                                    arr = np.asarray(predictions, dtype=float)
                                    fit_arr = fitted_by_method.get(method)
                                    k_eff = _estimate_effective_k(
                                        method,
                                        model_params,
                                        n_features=len(feature_names),
                                        n_selected_methods=max(1, len([m for m in forecast_results.keys() if m != "Ensemble"])),
                                    )
                                    if fit_arr is not None:
                                        aic_val, bic_val = _aic_bic_from_fit(y_true_diag, fit_arr, k_eff)
                                    else:
                                        aic_val, bic_val = (np.nan, np.nan)
                                    summary_rows_fc.append({
                                        "Series": f"Forecast — {method}",
                                        "Mean": float(np.mean(arr)),
                                        "Min": float(np.min(arr)),
                                        "Max": float(np.max(arr)),
                                        "Std": float(np.std(arr)),
                                        "AIC (train)": aic_val,
                                        "BIC (train)": bic_val,
                                    })
                                st.subheader("Summary statistics")
                                st.dataframe(
                                    pd.DataFrame(summary_rows_fc).round(3),
                                    use_container_width=True,
                                    hide_index=True,
                                )
                                st.session_state.forecast_report_payload = {
                                    "plain_summary": plain_summary,
                                    "insights_md": _ins,
                                    "summary_rows": summary_rows_fc,
                                    "forecast_fig": fig,
                                    "spatial_fig": _spatial_grid_plotly(
                                        values_grid if "values_grid" in locals() else None,
                                        bounds,
                                        "Spatial heatmap (historical snapshot)",
                                        colorscale="Reds",
                                    ) if "values_grid" in locals() and values_grid is not None else None,
                                    "hist_acf_fig": acf_fig_fc,
                                    "hist_dist_fig": dist_fig_fc if "dist_fig_fc" in locals() else None,
                                    "residual_diagnostics": residual_diagnostics,
                                }
                            # Prepare export data (wide: date, features, target, forecast_*, row_type)
                            st.session_state.combined_export_data = build_forecast_export_wide(
                                daily_avg,
                                selected_features,
                                forecast_target,
                                future_dates,
                                forecast_results_display,
                            )

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
                            st.plotly_chart(fig, use_container_width=True, theme=None)
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
                                    bounds,
                                    values_grid,
                                    opacity=heatmap_opacity,
                                    metric_name=metric_description,
                                    color_scheme="red",
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
                                vmin_ext = float(np.nanmin(values_grid))
                                vmax_ext = float(np.nanmax(values_grid))
                                vrange_ext = vmax_ext - vmin_ext
                                st.markdown(
                                    f"""
                                    <div style="margin-top:6px;padding:8px 10px;border:1px solid #fecaca;border-radius:8px;background:#ffffff;">
                                      <div style="font-size:12px;font-weight:700;text-align:center;margin-bottom:6px;color:#0f172a;">
                                        {metric_description} — {latest_date.strftime('%Y-%m-%d')}
                                      </div>
                                      <div style="height:12px;border-radius:6px;background:linear-gradient(to right,#FFF8F8,#FFCDD2,#E57373,#E53935,#C62828,#7F0000);"></div>
                                      <div style="display:flex;justify-content:space-between;font-size:12px;margin-top:6px;color:#111827;font-weight:700;">
                                        <span>Min: {vmin_ext:.2f}</span>
                                        <span>Max: {vmax_ext:.2f}</span>
                                        <span>Range: {vrange_ext:.2f}</span>
                                      </div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                                st.caption(
                                    f"{latest_date.strftime('%Y-%m-%d')} · {len(gdf_latest)} points"
                                )
                            else:
                                st.info("No spatial heatmap available for the latest date.")

                        y_ts = daily_avg[selected_metric].dropna().astype(float)
                        acf_ts = None
                        if len(y_ts) > 14:
                            acf_ts = _acf_plotly(
                                y_ts.values,
                                title=f"ACF — {selected_metric} (AOI daily mean)",
                                max_lag=min(40, max(5, len(y_ts) // 3)),
                                height=PAIR_DIAG_PLOT_HEIGHT,
                                margin=PAIR_DIAG_PLOT_MARGIN,
                            )
                        metric_description = AVAILABLE_PARAMETERS.get(
                            selected_metric,
                            "Fire Weather Index" if selected_metric == "FWI" else "Australian Fire Danger Rating"
                        )
                        ts2a, ts2b = st.columns(2, gap="large")
                        with ts2a:
                            st.subheader("ACF")
                            if acf_ts is not None:
                                st.plotly_chart(acf_ts, use_container_width=True, theme=None)
                            else:
                                st.info("Not enough historical points for ACF (need more than 14 days).")
                        with ts2b:
                            st.subheader("Distribution")
                            time_series_values = daily_avg[selected_metric].dropna().tolist()
                            dist_fig = create_distribution_plot(time_series_values, selected_metric, metric_description)
                            if dist_fig:
                                dist_fig.update_layout(
                                    title=f"{selected_metric} — AOI daily mean",
                                    height=PAIR_DIAG_PLOT_HEIGHT,
                                    margin=PAIR_DIAG_PLOT_MARGIN,
                                    autosize=True,
                                )
                                st.plotly_chart(dist_fig, use_container_width=True, theme=None)
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
            # Export functionality
            st.subheader("💾 Export Data")
                    
            if forecast_mode and 'combined_export_data' in st.session_state:
                st.write(
                    "Wide CSV: **`date`**, selected **feature** columns (AOI daily means), "
                    f"**`{forecast_target}`** (target), one **`forecast_*`** column per model, "
                    "and **`row_type`** (`historical` / `forecast`). "
                    "Future rows have missing observed features and target—model outputs are in **`forecast_*`**."
                )
                export_data = st.session_state.combined_export_data
                csv_buffer = StringIO()
                export_data.to_csv(csv_buffer, index=False)
                csv_str = csv_buffer.getvalue()

                st.download_button(
                    label="⬇️ Export Historical + Forecast CSV",
                    data=csv_str,
                    file_name=f"{forecast_target}_forecast_{start_date.strftime('%Y%m%d')}_{forecast_horizon}days.csv",
                    mime="text/csv",
                    key="download_forecast_csv",
                )
                    
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
st.markdown(f"""
<div style='text-align: center; color: gray;'>
🎯 {IWFR_DISPLAY_NAME} v1.5.3 | Built with Streamlit<br>
Powered by NASA POWER data from Goddard Earth Sciences Data and Information Services Center (GES DISC)
</div>
""", unsafe_allow_html=True)
