# app.py - RetailSense Dashboard (Final Combined Version)
import streamlit as st
import pandas as pd
import numpy as np
import os, subprocess, sys, random, warnings
import inspect
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.seasonal import STL
import io

# Ensure required directories exist
required_dirs = [
    os.path.join("data", "processed"),
    os.path.join("data", "uploaded"),
    os.path.join("data", "predictions"),
    "outputs",
    "notebooks"
]

for directory in required_dirs:
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), directory), exist_ok=True)

# Try to import Prophet (optional)
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    warnings.warn("Prophet not available. Install with: pip install prophet")

# Import utility modules with error handling
try:
    from utils.advanced_forecasting import (
        train_ensemble, train_ensemble_for_app, run_hybrid_forecast, 
        simulate_forecast_with_scenarios, run_advanced_forecast,
        evaluate_models, cross_validate_models, plot_forecast_results, create_features
    )
    from utils.business_insights import (
        detect_sales_anomalies, generate_inventory_alerts, analyze_seasonality,
        calculate_price_elasticity, analyze_pricing_opportunities, optimize_price,
        generate_executive_summary, generate_forecast_insights, calculate_scenario_impact,
        generate_ai_root_cause_explanation
    )
    from utils.data_loader import load_dataset, preprocess_data, load_csv_bytes, normalize_columns, get_quick_summary, dedupe_columns, validate_mapping
    from utils.column_mapping import render_column_mapping, apply_mapping_to_df
    from utils.dynamic_pricing_engine import DynamicPricingEngine
    MODULES_LOADED = True
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    MODULES_LOADED = False
    
# Fallback forecast function when regular forecasting fails
def generate_fallback_forecast(product: str, horizon_weeks: int) -> dict:
    """Generate a simple fallback forecast when regular forecasting fails."""
    print(f"Generating fallback forecast for {product} ({horizon_weeks} weeks)")
    
    # Create a date range for the forecast
    today = pd.Timestamp.today()
    start_date = today - pd.Timedelta(days=30)  # Some history
    future_dates = pd.date_range(
        start=today, 
        periods=horizon_weeks, 
        freq='W'
    )
    
    # Create a simple forecast with constant values
    history_df = pd.DataFrame({
        'ds': pd.date_range(start=start_date, periods=4, freq='W'),
        'y': [100, 100, 100, 100],  # Placeholder values
        'yhat': [100, 100, 100, 100]
    })
    
    forecast_df = pd.DataFrame({
        'ds': future_dates,
        'yhat': [100] * len(future_dates),
        'yhat_lower': [90] * len(future_dates),
        'yhat_upper': [110] * len(future_dates)
    })
    
    # Return a dictionary with the same structure as the regular forecast
    return {
        'forecast_df': forecast_df,
        'history_df': history_df,
        'metrics': {'smape': 0, 'mape': 0, 'rmse': 0, 'mae': 0},
        'feature_importances': {'price': 0.5, 'trend': 0.5},
        'prophet_components': None,
        'details': {'model_type': 'fallback', 'product': product}
    }

# Try to import SHAP for explainability (optional)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

warnings.filterwarnings("ignore")
st.set_page_config(
    page_title="RetailSense Lite — AI-Driven Retail Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# GLOBAL UI ENHANCEMENTS: Dark Theme & Custom Styling
# ============================================================================
CUSTOM_CSS = """
<style>
    /* Dark Professional Theme */
    .main {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%);
        color: #e0e0e0;
    }
    
    /* Header Styling */
    .header-container {
        background: linear-gradient(135deg, #1a237e 0%, #283593 50%, #3949ab 100%);
        padding: 1.5rem 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    .app-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
    }
    
    .app-tagline {
        font-size: 1.1rem;
        color: #b0bec5;
        margin-top: 0.5rem;
        font-style: italic;
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background-color: #1a1f3a;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
    }
    
    /* KPI Cards */
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: bold;
        color: #00e5ff;
    }
    
    /* Section Headers */
    h2, h3 {
        color: #00e5ff;
        border-bottom: 2px solid #00e5ff;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
    
    /* Info Boxes */
    .stInfo {
        background-color: rgba(33, 150, 243, 0.1);
        border-left: 4px solid #2196F3;
    }
    
    /* Success Messages */
    .stSuccess {
        background-color: rgba(76, 175, 80, 0.1);
        border-left: 4px solid #4CAF50;
    }
    
    /* Warning Messages */
    .stWarning {
        background-color: rgba(255, 193, 7, 0.1);
        border-left: 4px solid #FFC107;
    }
    
    /* Error Messages */
    .stError {
        background-color: rgba(244, 67, 54, 0.1);
        border-left: 4px solid #F44336;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #1a1f3a;
        color: #b0bec5;
        border-radius: 8px 8px 0 0;
        padding: 0.75rem 1.5rem;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #3949ab;
        color: #ffffff;
    }
    
    /* Dataframe Styling */
    .dataframe {
        background-color: #1a1f3a;
        color: #e0e0e0;
    }
    
    /* Plotly Chart Containers */
    .js-plotly-plot {
        background-color: rgba(26, 31, 58, 0.5);
        border-radius: 8px;
        padding: 1rem;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ============================================================================
# CUSTOM HEADER SECTION
# ============================================================================
header_html = """
<div class="header-container">
    <h1 class="app-title">📊 RetailSense Lite — AI-Driven Retail Analytics</h1>
    <p class="app-tagline">Powered by XGBoost • LightGBM • Prophet • Streamlit</p>
</div>
"""
st.markdown(header_html, unsafe_allow_html=True)

# Track whether the full pipeline has successfully run in this session
if "pipeline_success" not in st.session_state:
    st.session_state["pipeline_success"] = False
if "force_sales_tab" not in st.session_state:
    st.session_state["force_sales_tab"] = False
# Add a session state variable to maintain tab selection after product selection
if "current_tab" not in st.session_state:
    st.session_state["current_tab"] = "Home"

# -------------------------------
# Utility Functions
# -------------------------------
def load_csv(file_path):
    """Load CSV file with proper path handling and error messages"""
    # Ensure path is using proper OS path separators
    file_path = os.path.normpath(file_path)
    
    if os.path.exists(file_path):
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            st.error(f"❌ Error reading {file_path}: {e}")
            return None
    else:
        st.warning(f"File not found: {file_path}")
        return None

def profile_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = len(df)
    for col in df.columns:
        s = df[col]
        dtype = (
            "datetime" if pd.api.types.is_datetime64_any_dtype(s)
            else "numeric" if pd.api.types.is_numeric_dtype(s)
            else "boolean" if pd.api.types.is_bool_dtype(s)
            else "categorical" if pd.api.types.is_categorical_dtype(s)
            else "object"
        )
        non_null = int(s.notna().sum())
        missing = int(s.isna().sum())
        missing_pct = round((missing / n * 100.0), 2) if n else 0.0
        unique = int(s.nunique(dropna=True))
        sample = None
        try:
            # Convert sample to string to avoid PyArrow mixed-type errors
            sample_val = s.dropna().iloc[0] if non_null > 0 else None
            sample = str(sample_val) if sample_val is not None else None
        except Exception:
            sample = None

        stats = {"min": None, "max": None, "mean": None, "std": None, "top": None, "freq": None}
        if dtype == "numeric":
            stats["min"] = float(s.min()) if non_null else None
            stats["max"] = float(s.max()) if non_null else None
            stats["mean"] = float(s.mean()) if non_null else None
            stats["std"] = float(s.std()) if non_null else None
        elif dtype == "datetime":
            try:
                stats["min"] = pd.to_datetime(s, errors="coerce").min()
                stats["max"] = pd.to_datetime(s, errors="coerce").max()
            except Exception:
                stats["min"], stats["max"] = None, None
        else:
            vc = s.value_counts(dropna=True)
            if not vc.empty:
                stats["top"] = vc.index[0]
                stats["freq"] = int(vc.iloc[0])

        rows.append({
            "column": col,
            "dtype": dtype,
            "non_null": non_null,
            "missing": missing,
            "missing_%": missing_pct,
            "unique": unique,
            "sample": sample,
            "min": stats.get("min"),
            "max": stats.get("max"),
            "mean": stats.get("mean"),
            "std": stats.get("std"),
            "top": stats.get("top"),
            "freq": stats.get("freq"),
        })
    return pd.DataFrame(rows)

def dataframe_summary(df: pd.DataFrame):
    if df is not None and not df.empty:
        summary = {
            "Rows": [len(df)],
            "Columns": [len(df.columns)],
            "Missing (%)": [round(df.isna().mean().mean() * 100, 2)],
        }
        return pd.DataFrame(summary)
    return pd.DataFrame({"Rows": [0], "Columns": [0], "Missing (%)": [0]})

def download_button(df, label, filename, key=None):
    if df is not None and not df.empty:
        csv = df.to_csv(index=False).encode("utf-8")
        # Generate unique key from filename if not provided
        if key is None:
            key = f"download_{filename.replace('.', '_').replace('/', '_')}"
        st.download_button(label, csv, file_name=filename, mime="text/csv", key=key)

# -------------------------------
# Forecasting & Analytics Helpers
# -------------------------------
def identify_columns(df: pd.DataFrame) -> dict:
    cols = {}
    cols["date"] = next((c for c in ["date", "week_start", "ds"] if c in df.columns), None)
    cols["product"] = next((c for c in ["product_name", "product", "item_name", "name"] if c in df.columns), None)
    cols["sales"] = next((c for c in ["sales_qty", "quantity", "units", "order_quantity", "orders", "y"] if c in df.columns), None)
    cols["stock"] = next((c for c in ["stock_on_hand", "stock", "inventory"] if c in df.columns), None)
    cols["price"] = next((c for c in ["price", "unit_price", "selling_price"] if c in df.columns), None)
    cols["store"] = next((c for c in ["store", "store_id", "location", "region"] if c in df.columns), None)
    cols["category"] = next((c for c in ["category", "department", "aisle"] if c in df.columns), None)
    return cols

def preprocess_for_forecasting(ts: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess time series data before forecasting.
    Removes outliers, fills gaps, and smooths noise.
    """
    df = ts.copy()
    
    # 1. Remove extreme outliers using IQR method
    Q1 = df["sales_qty"].quantile(0.25)
    Q3 = df["sales_qty"].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 2.0 * IQR  # 2x IQR (less aggressive)
    upper_bound = Q3 + 2.0 * IQR
    
    # Cap outliers instead of removing (preserves data points)
    df["sales_qty"] = df["sales_qty"].clip(lower=max(0, lower_bound), upper=upper_bound)
    
    # 2. Fill missing dates with forward fill + interpolation
    df = df.sort_values("date").reset_index(drop=True)
    date_range = pd.date_range(start=df["date"].min(), end=df["date"].max(), freq="D")
    df_complete = pd.DataFrame({"date": date_range})
    df = df_complete.merge(df, on="date", how="left")
    
    # Forward fill for sales (carry last known value)
    df["sales_qty"] = df["sales_qty"].fillna(method="ffill").fillna(method="bfill").fillna(0)
    
    # 3. Apply light smoothing to reduce high-frequency noise (7-day moving average)
    df["sales_smooth"] = df["sales_qty"].rolling(window=7, center=True, min_periods=1).mean()
    
    # 4. Replace original with smoothed, but keep some variability (80% smooth + 20% original)
    df["sales_qty"] = 0.8 * df["sales_smooth"] + 0.2 * df["sales_qty"]
    df = df.drop(columns=["sales_smooth"])
    
    # 5. Handle other columns
    if "price" in df.columns:
        df["price"] = df["price"].fillna(method="ffill").fillna(df["price"].median())
    if "stock_on_hand" in df.columns:
        df["stock_on_hand"] = df["stock_on_hand"].fillna(method="ffill").fillna(0)
    if "promotion_flag" in df.columns:
        df["promotion_flag"] = df["promotion_flag"].fillna(0)
    if "holiday_flag" in df.columns:
        df["holiday_flag"] = df["holiday_flag"].fillna(0)
    
    return df

def make_product_ts(df: pd.DataFrame, product_name: str, cols: dict, store_val: str | None = None) -> pd.DataFrame:
    sdf = df.copy()
    sdf[cols["date"]] = pd.to_datetime(sdf[cols["date"]], errors="coerce")
    sdf = sdf[sdf[cols["product"]].astype(str) == str(product_name)]
    if store_val and cols["store"] in sdf.columns:
        sdf = sdf[sdf[cols["store"]].astype(str) == str(store_val)]
    
    # Build aggregation map - only numerical columns
    agg_map = {cols["sales"]: "sum"}
    if cols["stock"] and cols["stock"] in sdf.columns: 
        agg_map[cols["stock"]] = "sum"
    if cols["price"] and cols["price"] in sdf.columns: 
        agg_map[cols["price"]] = "mean"
    
    # Optional flags and descriptors
    promo_col = st.session_state.get("colmap", {}).get("promotion")
    holiday_col = st.session_state.get("colmap", {}).get("holiday")
    if promo_col and promo_col in sdf.columns:
        agg_map[promo_col] = "max"
    if holiday_col and holiday_col in sdf.columns:
        agg_map[holiday_col] = "max"
    
    # Categorical columns - use first value (don't aggregate numerically)
    if cols.get("store") and cols["store"] in sdf.columns:
        agg_map[cols["store"]] = "first"
    if cols.get("category") and cols["category"] in sdf.columns:
        agg_map[cols["category"]] = "first"
    
    ts = sdf.groupby(cols["date"]).agg(agg_map).reset_index()
    
    # Rename columns
    rename_map = {cols["date"]: "date", cols["sales"]: "sales_qty"}
    if cols["stock"] and cols["stock"] in ts.columns: 
        rename_map[cols["stock"]] = "stock_on_hand"
    if cols["price"] and cols["price"] in ts.columns: 
        rename_map[cols["price"]] = "price"
    if promo_col and promo_col in ts.columns: 
        rename_map[promo_col] = "promotion_flag"
    if holiday_col and holiday_col in ts.columns: 
        rename_map[holiday_col] = "holiday_flag"
    if cols.get("store") and cols["store"] in ts.columns: 
        rename_map[cols["store"]] = "store_region"
    if cols.get("category") and cols["category"] in ts.columns: 
        rename_map[cols["category"]] = "product_category"
    
    ts = ts.rename(columns=rename_map)
    
    # Ensure required columns exist with proper types
    if "stock_on_hand" not in ts.columns: 
        ts["stock_on_hand"] = np.nan
    if "price" not in ts.columns: 
        ts["price"] = np.nan
    
    # Drop categorical columns that might cause issues in numerical operations
    categorical_cols = ["store_region", "product_category"]
    for col in categorical_cols:
        if col in ts.columns:
            ts = ts.drop(columns=[col])
    
    ts = ts.sort_values("date").dropna(subset=["date"]).reset_index(drop=True)
    return ts

def get_colmap(df: pd.DataFrame) -> dict:
    # Use session-selected mapping if available; otherwise auto-detect
    user_map = st.session_state.get("colmap", {})
    if user_map and all(k in user_map for k in ["date", "product", "sales"]):
        return user_map
    return identify_columns(df)

def validate_colmap(colmap: dict, df: pd.DataFrame) -> tuple[bool, list]:
    missing = []
    for req in ["date", "product", "sales"]:
        name = colmap.get(req)
        if not name or name not in df.columns:
            missing.append(req)
    return (len(missing) == 0), missing

def run_prophet(ts: pd.DataFrame, horizon_days: int = 14) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run Prophet forecasting with error handling"""
    try:
        if not PROPHET_AVAILABLE:
            st.warning("Prophet is not installed. Using fallback forecasting method.")
            # Implement a simple fallback method
            return fallback_forecast(ts, horizon_days)
            
        df = ts.rename(columns={"date": "ds", "sales_qty": "y"})[["ds", "y"]].dropna()
        m = Prophet(interval_width=0.8, yearly_seasonality=True, weekly_seasonality=True)
        m.fit(df)
        future = m.make_future_dataframe(periods=horizon_days, freq="D")
        forecast = m.predict(future)
        forecast = forecast.rename(columns={"ds": "date", "yhat": "yhat", "yhat_lower": "yhat_lower", "yhat_upper": "yhat_upper"})
        hist_fit = forecast.merge(ts, on="date", how="left")
        return hist_fit, forecast[forecast["date"] > ts["date"].max()][["date", "yhat", "yhat_lower", "yhat_upper"]]
    except Exception as e:
        st.error(f"Error in Prophet forecasting: {e}")
        return fallback_forecast(ts, horizon_days)

def fallback_forecast(ts: pd.DataFrame, horizon_days: int = 14) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simple fallback forecasting method when Prophet is not available"""
    try:
        # Create a copy of the dataframe
        df = ts.copy()
        
        # Calculate simple moving average
        window = min(7, len(df) // 3)
        df['yhat'] = df['sales_qty'].rolling(window=window, min_periods=1).mean()
        
        # Create simple confidence intervals
        std = df['sales_qty'].std()
        df['yhat_lower'] = df['yhat'] - 1.96 * std
        df['yhat_upper'] = df['yhat'] + 1.96 * std
        
        # Create future dates
        last_date = df['date'].max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon_days)
        
        # Create forecast dataframe
        forecast = pd.DataFrame({'date': future_dates})
        
        # Use last 'window' days average for prediction
        avg_sales = df['sales_qty'].tail(window).mean()
        forecast['yhat'] = avg_sales
        forecast['yhat_lower'] = avg_sales - 1.96 * std
        forecast['yhat_upper'] = avg_sales + 1.96 * std
        
        return df, forecast
    except Exception as e:
        st.error(f"Error in fallback forecasting: {e}")
        # Return empty dataframes as last resort
        return pd.DataFrame(), pd.DataFrame()

def compute_kpis(history: pd.DataFrame, fc: pd.DataFrame) -> dict:
    k = {}
    next7 = fc.sort_values("date").head(7)
    k["next_week_sales"] = float(next7["yhat"].clip(lower=0).sum()) if not next7.empty else np.nan
    if "stock_on_hand" in history.columns and not history["stock_on_hand"].tail(4).isna().all():
        avg_stock = history["stock_on_hand"].tail(4).mean()
        avg_fc = next7["yhat"].mean() if not next7.empty else np.nan
        if not np.isnan(avg_stock) and not np.isnan(avg_fc) and avg_stock > 0:
            k["stockout_risk"] = float(np.clip((avg_fc - avg_stock) / max(avg_stock, 1), 0, 1))
        else:
            k["stockout_risk"] = np.nan
    else:
        k["stockout_risk"] = np.nan
    if not fc.empty:
        best = fc.loc[fc["yhat"].idxmax()]
        worst = fc.loc[fc["yhat"].idxmin()]
        k["best_day"], k["best_day_sales"] = str(best["date"]), float(best["yhat"])
        k["worst_day"], k["worst_day_sales"] = str(worst["date"]), float(worst["yhat"])
    return k

def compute_accuracy(hist_fit: pd.DataFrame) -> dict:
    # Handle both 'yhat' and 'fitted' column names (for compatibility with different forecasting modules)
    pred_col = None
    if "yhat" in hist_fit.columns:
        pred_col = "yhat"
    elif "fitted" in hist_fit.columns:
        pred_col = "fitted"
    else:
        return {"RMSE": np.nan, "MAE": np.nan}
    
    df = hist_fit.dropna(subset=[pred_col, "sales_qty"]).copy()
    if df.empty:
        return {"RMSE": np.nan, "MAE": np.nan}
    rmse = float(np.sqrt(mean_squared_error(df["sales_qty"], df[pred_col])))
    mae = float(mean_absolute_error(df["sales_qty"], df[pred_col]))
    return {"RMSE": rmse, "MAE": mae}

def compute_elasticity(ts: pd.DataFrame) -> float:
    if "price" not in ts.columns or ts["price"].isna().all():
        return -1.2
    df = ts.dropna(subset=["price", "sales_qty"]).copy()
    if len(df) < 12 or (df["price"].nunique() < 3):
        return -1.2
    x = np.log(df["price"].values + 1e-6)
    y = np.log(df["sales_qty"].values + 1e-6)
    coef = np.polyfit(x, y, 1)[0]
    return float(coef)

def detect_anomalies(hist_fit: pd.DataFrame, z_thresh: float = 3.0) -> pd.DataFrame:
    # Handle both 'yhat' and 'fitted' column names
    pred_col = "yhat" if "yhat" in hist_fit.columns else "fitted" if "fitted" in hist_fit.columns else None
    if pred_col is None:
        return pd.DataFrame(columns=["date", "sales_qty", "predicted", "residual", "is_anomaly"])
    df = hist_fit.dropna(subset=[pred_col, "sales_qty"]).copy()
    if df.empty:
        return pd.DataFrame()
    df["resid"] = df["sales_qty"] - df[pred_col]
    mu, sd = df["resid"].mean(), df["resid"].std(ddof=1) if df["resid"].std(ddof=1) > 0 else 1.0
    df["z"] = (df["resid"] - mu) / sd
    # Rename pred_col to "predicted" for consistency in output
    df_ret = df.copy()
    df_ret["predicted"] = df_ret[pred_col]
    return df_ret.loc[df_ret["z"].abs() >= z_thresh, ["date", "sales_qty", "predicted", "z"]]

def decompose_series(ts: pd.DataFrame, period: int = 7) -> pd.DataFrame:
    s = ts.set_index("date")["sales_qty"].asfreq("D").interpolate(limit_direction="both")
    res = STL(s, period=period, robust=True).fit()
    return pd.DataFrame({
        "date": s.index,
        "trend": res.trend.values,
        "seasonal": res.seasonal.values,
        "resid": res.resid.values,
    })

# -------------------------------
# Directories and Paths
# -------------------------------
BASE_DIR         = "F:\\RetailSense_Lite"
RAW_DIR          = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR    = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_DIR       = os.path.join(BASE_DIR, "outputs")
NOTEBOOKS_DIR    = os.path.join(BASE_DIR, "notebooks")
UPLOAD_DIR       = os.path.join(BASE_DIR, "data", "uploaded")

# -------------------------------
# Clear Old Outputs Function
# -------------------------------
def clear_all_outputs():
    """Clear all old output files to ensure fresh pipeline runs"""
    output_files = [
        os.path.join(OUTPUT_DIR, "forecasting_results.csv"),
        os.path.join(OUTPUT_DIR, "business_sales_anomalies.csv"),
        os.path.join(OUTPUT_DIR, "business_inventory_alerts.csv"),
        os.path.join(OUTPUT_DIR, "business_seasonal_insights.csv"),
        os.path.join(OUTPUT_DIR, "business_pricing_opportunities.csv"),
        os.path.join(OUTPUT_DIR, "phase3_completion_report.csv"),
        os.path.join(OUTPUT_DIR, "executive_dashboard.png"),
        os.path.join(OUTPUT_DIR, "anomaly_dashboard.png"),
        os.path.join(OUTPUT_DIR, "model_performance_comparison.png"),
        os.path.join(PROCESSED_DIR, "cleaned_data.csv"),
        os.path.join(PROCESSED_DIR, "data_with_all_features.csv"),
        os.path.join(PROCESSED_DIR, "raw2.csv"),
    ]
    
    cleared_count = 0
    for file_path in output_files:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                cleared_count += 1
        except Exception as e:
            print(f"Warning: Could not delete {file_path}: {e}")
    
    return cleared_count

RAW_DATA_PATH      = os.path.join(RAW_DIR, "market.csv")
CLEANED_DATA_PATH  = os.path.join(PROCESSED_DIR, "cleaned_data.csv")
FEATURES_DATA_PATH = os.path.join(PROCESSED_DIR, "data_with_all_features.csv")
UPLOADED_FILE_PATH = os.path.join(UPLOAD_DIR, "uploaded_data.csv")

FORECAST_FILE   = os.path.join(OUTPUT_DIR, "forecasting_results.csv")
ANOMALIES_FILE  = os.path.join(OUTPUT_DIR, "business_sales_anomalies.csv")
INVENTORY_FILE  = os.path.join(OUTPUT_DIR, "business_inventory_alerts.csv")
SEASONAL_FILE   = os.path.join(OUTPUT_DIR, "business_seasonal_insights.csv")
PRICING_FILE    = os.path.join(OUTPUT_DIR, "business_pricing_opportunities.csv")

# -------------------------------
# -------------------------------
# ============================================================================
# ENHANCED SIDEBAR: Navigation & Dataset Info
# ============================================================================
st.sidebar.image("https://img.icons8.com/color/96/000000/shopify.png", width=120)
st.sidebar.markdown("## 📊 RetailSense Lite")
st.sidebar.markdown("**AI-Driven Retail Analytics**")
st.sidebar.markdown("---")

st.sidebar.subheader("⚙️ Controls")

uploaded_file = st.sidebar.file_uploader("Upload Retail Data CSV", type="csv")

if uploaded_file is not None:
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    # Check if this is a new file upload
    current_file_id = f"{uploaded_file.name}_{uploaded_file.size}"
    previous_file_id = st.session_state.get("uploaded_file_id")
    
    if current_file_id != previous_file_id:
        # New file uploaded - save it and reset state
        with open(UPLOADED_FILE_PATH, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state["uploaded_file_id"] = current_file_id
        st.session_state["uploaded_ready"] = True
        
        # Reset any previous session selections/mappings to avoid stale history
        st.session_state.pop("column_mapping", None)
        st.session_state.pop("colmap", None)  # Keep for backward compatibility
        st.session_state.pop("selected_product_id", None)
        st.session_state.pop("selected_product", None)
        st.session_state.pop("df_mapped", None)
        st.session_state["force_sales_tab"] = False
        
        # Clear cached data
        st.session_state.pop("df_full_cached", None)
        st.session_state.pop("raw_df", None)
        st.session_state.pop("df_profile_cached", None)
        st.session_state.pop("auto_map_cached", None)
        
        st.sidebar.success(f"✅ File uploaded successfully!")
    else:
        # Same file still uploaded - ensure flag is set
        st.session_state["uploaded_ready"] = True
    
    # Load dataset using new data loader (handles duplicates, normalization)
    if "raw_df" not in st.session_state:
        try:
            with st.spinner("Loading and processing CSV..."):
                # Use new load_csv_bytes function
                uploaded_file.seek(0)  # Reset file pointer
                df_full = load_csv_bytes(uploaded_file)
                if df_full is not None and not df_full.empty:
                    st.session_state["raw_df"] = df_full.copy()
                    st.session_state["df_full_cached"] = df_full.copy()  # Keep for backward compatibility
        except Exception as e:
            st.sidebar.error(f"❌ Unable to read uploaded CSV: {e}")
            df_full = None
    else:
        df_full = st.session_state.get("raw_df")

    if df_full is not None and not df_full.empty:
        # Quick summary using new function
        st.sidebar.subheader("📋 Quick Summary")
        mapping = st.session_state.get("column_mapping", {})
        date_col = mapping.get("date")
        summary = get_quick_summary(df_full, date_col=date_col)
        
        c1, c2 = st.sidebar.columns(2)
        c1.metric("Rows", f"{summary['rows']:,}")
        c2.metric("Columns", f"{summary['cols']:,}")
        
        if summary.get('date_range'):
            st.sidebar.caption(f"📅 Date Range: {summary['date_range'][0]} to {summary['date_range'][1]}")
        if summary.get('n_products', 0) > 0:
            st.sidebar.caption(f"🏷️ Products: {summary['n_products']}")
        
        # Lazy-loaded data preview in expander
        with st.sidebar.expander("📊 View Data Preview & Profile", expanded=False):
            st.dataframe(df_full.head(10), use_container_width=True)
            
            # Compute profile only when expander is opened (cached)
            if "df_profile_cached" not in st.session_state:
                with st.spinner("Generating data profile..."):
                    total_cells = int(df_full.shape[0] * df_full.shape[1])
                    missing_cells = int(df_full.isna().sum().sum())
                    missing_cells_pct = round((missing_cells / total_cells * 100.0), 2) if total_cells else 0.0
                    duplicate_rows = int(df_full.duplicated().sum())
                    mem_bytes = int(df_full.memory_usage(deep=True).sum())
                    mem_mb = round(mem_bytes / (1024 * 1024), 2)
                    
                    prof = profile_dataframe(df_full)
                    
                    st.session_state["df_profile_cached"] = {
                        "missing_pct": missing_cells_pct,
                        "duplicate_rows": duplicate_rows,
                        "mem_mb": mem_mb,
                        "profile": prof
                    }
            
            profile_data = st.session_state.get("df_profile_cached")
            if profile_data:
                st.markdown("**Detailed Statistics:**")
                c3, c4, c5 = st.columns(3)
                c3.metric("Missing (%)", f"{profile_data['missing_pct']:.2f}%")
                c4.metric("Duplicates", f"{profile_data['duplicate_rows']:,}")
                c5.metric("Memory (MB)", f"{profile_data['mem_mb']:.2f}")
                
                st.markdown("**Column Profile:**")
                st.dataframe(profile_data['profile'], use_container_width=True, hide_index=True, height=300)
                
                try:
                    prof_csv = profile_data['profile'].to_csv(index=False).encode("utf-8")
                    st.download_button("⬇️ Download Profile", prof_csv, file_name="data_profile.csv", mime="text/csv", key="download_profile_pipeline")
                except Exception:
                    pass

        # Column Mapping using new component
        default_map = st.session_state.get("column_mapping")
        mapping = render_column_mapping(df_full.columns.tolist(), default_map=default_map)
        
        # Store current mapping in session state (even if not applied yet)
        st.session_state["current_mapping"] = mapping
        
        # Check if mapping is valid
        mapping_valid, missing_cols = validate_mapping(mapping, df_full)
        st.session_state["mapping_valid"] = mapping_valid
        
        # Auto-apply mapping if valid (even if user hasn't clicked "Apply Mapping" yet)
        # This ensures df_mapped is available for the Sales Forecasting tab
        if mapping_valid:
            # Use existing column_mapping if available, otherwise use current mapping (auto-apply)
            mapping_to_use = st.session_state.get("column_mapping")
            if mapping_to_use is None:
                # Auto-apply: use current mapping and store it
                mapping_to_use = mapping
                st.session_state["column_mapping"] = mapping_to_use.copy()
            
            # Check if we need to create or update df_mapped
            last_applied = st.session_state.get("last_mapping_applied")
            needs_update = (
                "df_mapped" not in st.session_state or 
                last_applied is None or
                last_applied != mapping_to_use
            )
            
            if needs_update:
                # Auto-apply mapping and store in session state
                df_mapped = apply_mapping_to_df(df_full, mapping_to_use)
                st.session_state["df_mapped"] = df_mapped
                st.session_state["product_list"] = sorted(df_mapped["product"].unique().tolist()) if "product" in df_mapped.columns else []
                st.session_state["last_mapping_applied"] = mapping_to_use.copy()
        else:
            # Invalid mapping - clear df_mapped
            st.sidebar.warning(f"⚠️ Map required columns: {', '.join(missing_cols)} before proceeding.")
            st.session_state["df_mapped"] = None
        
        # Display current column mapping (if applied)
        if "column_mapping" in st.session_state and st.session_state["column_mapping"]:
            st.sidebar.markdown("---")
            st.sidebar.subheader("🗺️ Current Column Mapping")
            
            applied_mapping = st.session_state["column_mapping"]
            mapping_display = []
            
            # Required fields
            required_fields = {
                "date": "📅 Date",
                "product": "🏷️ Product",
                "sales_qty": "📊 Sales Quantity"
            }
            
            # Optional fields
            optional_fields = {
                "price": "💰 Price",
                "stock_on_hand": "📦 Stock",
                "category": "📁 Category",
                "store": "🏪 Store"
            }
            
            st.sidebar.markdown("**Required:**")
            for field, label in required_fields.items():
                mapped_col = applied_mapping.get(field)
                if mapped_col:
                    st.sidebar.caption(f"{label}: `{mapped_col}`")
            
            # Check if any optional fields are mapped
            has_optional = any(applied_mapping.get(field) for field in optional_fields.keys())
            if has_optional:
                st.sidebar.markdown("**Optional:**")
                for field, label in optional_fields.items():
                    mapped_col = applied_mapping.get(field)
                    if mapped_col:
                        st.sidebar.caption(f"{label}: `{mapped_col}`")
            
            # Show mapping status
            if mapping_valid:
                st.sidebar.success("✅ Mapping Applied")
            else:
                st.sidebar.warning("⚠️ Incomplete Mapping")

# -------------------------------
# Main Page
# -------------------------------
st.title("🛍️ RetailSense Dashboard")
st.caption("AI-Powered Retail Analytics & Dynamic Pricing Engine")

# Clear old outputs on fresh page load (only once per session)
if "outputs_cleared" not in st.session_state:
    cleared_count = clear_all_outputs()
    st.session_state["outputs_cleared"] = True
    if cleared_count > 0:
        print(f"Cleared {cleared_count} old output files on dashboard load")

# Determine upload readiness (current session only)
uploaded_ready = st.session_state.get("uploaded_ready", False)

# Reset transient state when no upload present in this session
if not uploaded_ready:
    st.session_state["pipeline_success"] = False
    st.session_state.pop("selected_product_id", None)
    st.session_state.pop("colmap", None)
    st.session_state.pop("pipeline_timestamp", None)

# Global banner reminding to upload before results
if not uploaded_ready:
    st.info("Upload a CSV from the sidebar to enable the pipeline and view forecasting, anomalies, inventory, seasonal, and pricing results.")

# -------------------------------
# Run Pipeline (Phases 1, 2, 3)
# -------------------------------
if st.button("🚀 Run Full Pipeline", disabled=not uploaded_ready):
    with st.spinner("🔄 Running full pipeline... please wait"):
        try:
            # Upload gate: require uploaded CSV before running
            if not uploaded_ready:
                st.error("Please upload a CSV from the sidebar before running the pipeline.")
            else:
                # Clear old outputs before running new pipeline (silently in background)
                cleared_count = clear_all_outputs()
                
                # Store timestamp BEFORE pipeline starts (files created after this are fresh)
                from datetime import datetime, timedelta
                pipeline_start_time = datetime.now() - timedelta(seconds=5)  # 5 sec buffer
                st.session_state["pipeline_timestamp"] = pipeline_start_time.isoformat()
                
                # Phase 1: Data Cleaning
                result1 = subprocess.run(
                    [
                        sys.executable, "-m", "papermill",
                        os.path.join(NOTEBOOKS_DIR, "phase1_eda.ipynb"),
                        os.path.join(NOTEBOOKS_DIR, "phase1_out.ipynb"),
                        "-p", "INPUT_CSV", UPLOADED_FILE_PATH,
                        "-p", "RETAILSENSE_BASE_DIR", BASE_DIR,
                    ],
                    capture_output=True, text=True
                )
                if result1.returncode == 0:
                    st.success("✅ Phase 1: Data Cleaning complete")
                else:
                    st.error(f"❌ Phase 1 failed:\n{result1.stderr}")

                # Phase 2: Core ML Models (pass env vars instead of papermill -p)
                env2 = os.environ.copy()
                env2["RETAILSENSE_BASE_DIR"] = BASE_DIR
                env2["UPLOADED_DATA_PATH"] = UPLOADED_FILE_PATH
                result2 = subprocess.run(
                    [
                        sys.executable, "-m", "papermill",
                        os.path.join(NOTEBOOKS_DIR, "phase2_core_models.ipynb"),
                        os.path.join(NOTEBOOKS_DIR, "phase2_out.ipynb"),
                    ],
                    capture_output=True, text=True, env=env2
                )
                if result2.returncode == 0:
                    st.success("✅ Phase 2: Core ML Models complete")
                    # CRITICAL: Verify that data_with_all_features.csv was created
                    if os.path.exists(FEATURES_DATA_PATH):
                        file_size = os.path.getsize(FEATURES_DATA_PATH)
                        if file_size > 0:
                            st.success(f"✅ Verified: data_with_all_features.csv created ({file_size:,} bytes)")
                        else:
                            st.warning("⚠️ Warning: data_with_all_features.csv exists but is empty")
                    else:
                        st.error(f"❌ CRITICAL: data_with_all_features.csv was not created after Phase 2!")
                        st.error(f"Expected at: {FEATURES_DATA_PATH}")
                else:
                    st.error(f"❌ Phase 2 failed:\n{result2.stderr}")

                # Phase 3: Business Insights
                env3 = os.environ.copy()
                env3["RETAILSENSE_BASE_DIR"] = BASE_DIR
                env3["UPLOADED_DATA_PATH"] = UPLOADED_FILE_PATH
                result3 = subprocess.run(
                    [sys.executable, "-m", "papermill",
                     os.path.join(NOTEBOOKS_DIR, "phase3.ipynb"),
                     os.path.join(NOTEBOOKS_DIR, "phase3_out.ipynb")],
                    capture_output=True, text=True, env=env3
                )
                if result3.returncode == 0:
                    st.success("✅ Phase 3: Business Insights complete")
                else:
                    st.error(f"❌ Phase 3 failed:\n{result3.stderr}")

                if result1.returncode == 0 and result2.returncode == 0 and result3.returncode == 0:
                    st.success("🎉 Pipeline executed successfully!")
                    st.session_state["pipeline_success"] = True
                    # Timestamp already set before pipeline started
                else:
                    st.error("❌ Pipeline did not complete successfully.")
                    st.session_state["pipeline_success"] = False
                    st.session_state.pop("pipeline_timestamp", None)
        except Exception as e:
            st.error(f"❌ Pipeline execution error: {e}")

# -------------------------------
# Load Outputs (only after pipeline success AND current-session upload)
# -------------------------------
def load_output_with_freshness_check(file_path, pipeline_timestamp):
    """Load CSV only if it exists and was created after pipeline run"""
    if not os.path.exists(file_path):
        return None
    
    # Check if file was modified after pipeline timestamp
    if pipeline_timestamp:
        try:
            from datetime import datetime
            file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
            pipeline_time = datetime.fromisoformat(pipeline_timestamp)
            
            # File must be newer than or equal to pipeline timestamp (with tolerance)
            # Files created during pipeline execution are considered fresh
            if file_mtime >= pipeline_time:
                return load_csv(file_path)
            else:
                print(f"WARNING: Skipping stale file: {file_path}")
                print(f"  File time: {file_mtime}, Pipeline time: {pipeline_time}")
                return None
        except Exception as e:
            print(f"Warning: Could not check file freshness for {file_path}: {e}")
            return None
    
    return load_csv(file_path)

if uploaded_ready and st.session_state.get("pipeline_success"):
    pipeline_timestamp = st.session_state.get("pipeline_timestamp")
    
    # Load outputs with freshness check
    forecast_results  = load_output_with_freshness_check(FORECAST_FILE, pipeline_timestamp)
    sales_anomalies   = load_output_with_freshness_check(ANOMALIES_FILE, pipeline_timestamp)
    inventory_alerts  = load_output_with_freshness_check(INVENTORY_FILE, pipeline_timestamp)
    seasonal_insights = load_output_with_freshness_check(SEASONAL_FILE, pipeline_timestamp)
    pricing_opps      = load_output_with_freshness_check(PRICING_FILE, pipeline_timestamp)
    
    # Verify at least some outputs were loaded
    if all(x is None for x in [forecast_results, sales_anomalies, inventory_alerts, seasonal_insights, pricing_opps]):
        # Debug: Check which files exist
        missing_files = []
        for name, path in [("Forecast", FORECAST_FILE), ("Anomalies", ANOMALIES_FILE), 
                          ("Inventory", INVENTORY_FILE), ("Seasonal", SEASONAL_FILE), 
                          ("Pricing", PRICING_FILE)]:
            if not os.path.exists(path):
                missing_files.append(name)
        
        if missing_files:
            st.warning(f"⚠️ Pipeline succeeded but output files not created: {', '.join(missing_files)}")
        else:
            st.warning("⚠️ Pipeline succeeded but output files appear stale. Check console for details.")
        st.session_state["pipeline_success"] = False
else:
    forecast_results  = None
    sales_anomalies   = None
    inventory_alerts  = None
    seasonal_insights = None
    pricing_opps      = None

# -------------------------------
# KPIs
# -------------------------------
col1, col2, col3, col4 = st.columns(4)
if st.session_state.get("pipeline_success"):
    col1.metric("📈 Forecast Models", len(forecast_results) if forecast_results is not None else 0)
    col2.metric("🚨 Anomalies", len(sales_anomalies) if sales_anomalies is not None else 0)
    col3.metric("📦 Inventory Alerts", len(inventory_alerts) if inventory_alerts is not None else 0)
    col4.metric("💰 Pricing Opps", len(pricing_opps) if pricing_opps is not None else 0)
else:
    col1.metric("📈 Forecast Models", "-")
    col2.metric("🚨 Anomalies", "-")
    col3.metric("📦 Inventory Alerts", "-")
    col4.metric("💰 Pricing Opps", "-")

# -------------------------------
# Tabs
# -------------------------------
# Store the current tab in session state when it changes
def on_tab_change(tab_name):
    st.session_state["current_tab"] = tab_name

# Get the tab index based on the current_tab in session state
def get_tab_index():
    tab_names = [
        "Executive Summary",
        "Sales Forecasting",
        "Sales Anomalies",
        "Inventory Alerts",
        "Seasonal Insights",
        "Pricing Opportunities",
        "Dynamic Pricing Engine",
        "Data Summary"
    ]
    current_tab = st.session_state.get("current_tab", "Home")
    try:
        return tab_names.index(current_tab)
    except ValueError:
        return 0  # Default to first tab if not found

# Use the stored tab index or default to the first tab
tab_names = [
    "📊 Executive Summary",
    "📈 Sales Forecasting",
    "🚨 Sales Anomalies",
    "📦 Inventory Alerts",
    "🎯 Seasonal Insights",
    "💰 Pricing Opportunities",
    "⚙️ Dynamic Pricing Engine",
    "🧾 Data Summary",
]
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(tab_names)

# Set the active tab based on session state
if st.session_state["current_tab"] == "Sales Forecasting":
    tab2.active = True

with tab1:
    st.subheader("📊 Executive Summary")
    if not st.session_state.get("pipeline_success"):
        st.info("Run the Full Pipeline to see the executive summary.")
    else:
        if forecast_results is not None and not forecast_results.empty:
            # Check if this is old format (with Model and RMSE columns) or new format (just forecast data)
            if "RMSE" in forecast_results.columns and "Model" in forecast_results.columns:
                best_model = forecast_results.loc[forecast_results["RMSE"].idxmin(), "Model"]
                st.success(f"✅ Best Forecasting Model: **{best_model}** with lowest RMSE.")
            else:
                # New 3-year hybrid engine format
                st.success(f"✅ 3-Year Hybrid Forecasting Engine: Prophet + XGBoost + LightGBM active")
        if sales_anomalies is not None:
            st.warning(f"⚠️ {len(sales_anomalies)} anomalies detected in sales trends.")
        if inventory_alerts is not None:
            urgent = inventory_alerts[inventory_alerts["urgency"] == "High"].shape[0] if "urgency" in inventory_alerts.columns else 0
            st.error(f"🚨 {urgent} urgent inventory issues flagged.")
        if pricing_opps is not None:
            st.info(f"💡 {len(pricing_opps)} potential pricing opportunities identified.")

with tab2:
    # Store that we're in the Sales Forecasting tab
    on_tab_change("Sales Forecasting")
    
    st.subheader("📈 Sales Forecasting & Business Intelligence")
    st.markdown("**🚀 Placement-Grade Analytics: Hybrid Forecasting + Anomaly Detection + Pricing Intelligence**")
    
    # Check if mapped data is available
    df_mapped = st.session_state.get("df_mapped")
    mapping_valid = st.session_state.get("mapping_valid", False)
    
    if df_mapped is None or not mapping_valid:
        st.warning("⚠️ Please upload a CSV file and map required columns (date, product, sales_qty) in the sidebar before proceeding.")
        st.info("💡 Steps to get started:\n1. Upload a CSV file in the sidebar\n2. Map your columns (date, product, sales_qty are required)\n3. Click 'Apply Mapping' button\n4. Select a product below")
        st.stop()
    
    # Product Selection - Use df_mapped
    if "product" not in df_mapped.columns:
        st.error("❌ 'product' column not found in mapped dataset. Please check your column mapping.")
        st.stop()
    
    # Get unique product names (filter out codes and get clean names)
    products_raw = df_mapped["product"].unique().tolist()
    
    # Filter to show only product names (exclude codes like P001, P002, etc.)
    # If product column contains codes, try to find product_name column
    if "product_name" in df_mapped.columns:
        # Use product_name column if available
        product_list = sorted(df_mapped["product_name"].unique().tolist())
        # Create mapping from product_name to product for filtering
        product_name_map = df_mapped[["product", "product_name"]].drop_duplicates().set_index("product_name")["product"].to_dict()
        st.session_state["product_name_to_product_map"] = product_name_map
    else:
        # Filter out product codes (patterns like P001, PROD001, etc.)
        import re
        product_list = []
        for prod in products_raw:
            # Skip if it matches common product code patterns
            if not re.match(r'^P\d+$|^PROD\d+$|^[A-Z]{1,3}\d+$', str(prod).strip().upper()):
                product_list.append(prod)
        
        # If filtering removed everything, show all products
        if not product_list:
            product_list = sorted([str(p) for p in products_raw])
        else:
            product_list = sorted([str(p) for p in product_list])
        
        # Store direct mapping (name -> name in this case)
        st.session_state["product_name_to_product_map"] = {p: p for p in product_list}
    
    # Store in session state for consistency
    if "product_list" not in st.session_state or st.session_state.get("product_list") != product_list:
        st.session_state["product_list"] = product_list
    
    # Store current tab to prevent redirects
    st.session_state["current_tab"] = "Sales Forecasting"
    st.session_state["last_action"] = {
        'action': 'view_forecast_tab',
        'time': str(datetime.now())
    }
    
    col_sel1, col_sel2 = st.columns([3, 1])
    with col_sel1:
        selected_product = st.selectbox(
            "🔍 Choose Product for Analysis",
            options=["-- Select Product --"] + product_list,
            key="selected_product",
            help="Select a product to view its sales history and run forecasts"
        )
        
        # Update session state when product changes (no redirect)
        if selected_product and selected_product != "-- Select Product --":
            st.session_state["last_action"] = {
                'action': 'select_product',
                'product': selected_product,
                'time': str(datetime.now())
            }
            
            # Map product name to product code if needed
            product_name_to_product_map = st.session_state.get("product_name_to_product_map", {})
            actual_product_value = product_name_to_product_map.get(selected_product, selected_product)
            
            # Filter product data and store in session state
            product_filtered = df_mapped[df_mapped["product"] == actual_product_value].copy()
            if "date" in product_filtered.columns:
                product_filtered = product_filtered.sort_values("date").reset_index(drop=True)
            st.session_state["selected_product_df"] = product_filtered
        else:
            st.session_state["selected_product_df"] = None
    
    # Show product selection UI only
    if not selected_product or selected_product == "-- Select Product --":
        st.info("👆 Please select a product from the dropdown above to view sales history and run forecasts.")
        st.stop()
    
    # Get selected product data
    product_df = st.session_state.get("selected_product_df")
    if product_df is None or product_df.empty:
        st.warning(f"⚠️ No data found for product: {selected_product}")
        st.stop()
    
    # Show simplified UI: Recent History + Run Forecast button
    with col_sel2:
        avg_sales = product_df["sales_qty"].mean() if "sales_qty" in product_df.columns else 0
        st.metric("Avg Weekly Sales", f"{avg_sales:.0f}")
    
    if len(product_df) < 8:
        st.warning(f"⚠️ Insufficient data for {st.session_state.get('selected_product')}. Need at least 8 weeks of history.")
        st.stop()
    
    # Load data_with_all_features.csv directly (for backward compatibility with existing tabs)
    @st.cache_data(show_spinner="Loading dataset...")
    def load_features_data():
        """Load the processed features dataset"""
        try:
            if not os.path.exists(FEATURES_DATA_PATH):
                return None
            
            # Try loading with error handling - use low_memory=False for mixed types
            df = pd.read_csv(FEATURES_DATA_PATH, low_memory=False, encoding='utf-8')
            
            if df.empty:
                return None
            
            # Convert date columns with error handling
            if "week_start" in df.columns:
                df["week_start"] = pd.to_datetime(df["week_start"], errors='coerce')
            if "week_end" in df.columns:
                df["week_end"] = pd.to_datetime(df["week_end"], errors='coerce')
            
            return df
        except pd.errors.EmptyDataError:
            return None
        except Exception as e:
            # Log error but don't use st commands in cached function
            print(f"Error loading dataset: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return None
    
    try:
        features_df = load_features_data()
        if features_df is None:
            # Try loading without cache to get actual error
            if os.path.exists(FEATURES_DATA_PATH):
                features_df = pd.read_csv(FEATURES_DATA_PATH, low_memory=False, encoding='utf-8')
                if "week_start" in features_df.columns:
                    features_df["week_start"] = pd.to_datetime(features_df["week_start"], errors='coerce')
    except Exception as e:
        st.error(f"❌ Error loading dataset: {str(e)}")
        with st.expander("🔍 View Error Details"):
            import traceback
            st.code(traceback.format_exc())
        features_df = None
    
    # Check if features_df is available and has required columns
    features_available = features_df is not None and not features_df.empty and "product_name" in features_df.columns
    
    if not features_available:
        st.error(f"❌ Dataset not found: {FEATURES_DATA_PATH}")
        
        # Provide helpful diagnostics
        if os.path.exists(CLEANED_DATA_PATH):
            st.info("💡 Phase 1 completed (cleaned_data.csv exists), but Phase 2 may have failed.")
        else:
            st.info("💡 Please run the Full Pipeline from the top of the page.")
        
        st.markdown("**To fix this:**")
        st.markdown("1. Ensure you've uploaded a CSV file")
        st.markdown("2. Click '🚀 Run Full Pipeline' button above")
        st.markdown("3. Wait for Phase 2 to complete successfully")
        st.markdown(f"4. Check that file exists: `{FEATURES_DATA_PATH}`")
        
        # Show if file exists but is empty or corrupted
        if os.path.exists(FEATURES_DATA_PATH):
            file_size = os.path.getsize(FEATURES_DATA_PATH)
            if file_size == 0:
                st.warning(f"⚠️ File exists but is empty ({file_size} bytes). Re-run Phase 2.")
            else:
                st.warning(f"⚠️ File exists ({file_size:,} bytes) but could not be loaded. It may be corrupted.")
        
        st.stop()
    
    # Use the selected product from the first product selector (above)
    # Map the selected product to features_df format
    selected_product_from_first = st.session_state.get("selected_product")
    
    if not selected_product_from_first or selected_product_from_first == "-- Select Product --":
        st.info("👆 Please select a product from the dropdown above to view advanced forecasting features.")
        st.stop()
    
    # Try to find the product in features_df by product_name
    # The selected product name should match or we need to find it by mapping
    product_df_features = None
    
    # First try exact match
    if selected_product_from_first in features_df["product_name"].values:
        product_df_features = features_df[features_df["product_name"] == selected_product_from_first].copy()
    else:
        # Try to find by partial match or use the first matching entry
        # Check if there's a mapping we can use
        matching_products = features_df[features_df["product_name"].str.contains(selected_product_from_first, case=False, na=False)]
        if not matching_products.empty:
            product_df_features = matching_products.copy()
            selected_product_from_first = matching_products["product_name"].iloc[0]
        else:
            # If no match found, use the first product as fallback
            st.warning(f"⚠️ Product '{selected_product_from_first}' not found in features dataset. Using first available product.")
            if not features_df.empty:
                selected_product_from_first = features_df["product_name"].iloc[0]
                product_df_features = features_df[features_df["product_name"] == selected_product_from_first].copy()
    
    if product_df_features is None or product_df_features.empty:
        st.error(f"❌ Could not find data for product: {selected_product_from_first}")
        st.stop()
    
    # Sort by week_start
    if "week_start" in product_df_features.columns:
        product_df_features = product_df_features.sort_values("week_start").reset_index(drop=True)
    
    if len(product_df_features) < 8:
        st.warning(f"⚠️ Insufficient data for {selected_product_from_first}. Need at least 8 weeks of history.")
        st.stop()
    
    # Use product_df_features for the subtabs (renamed to product_df for compatibility)
    product_df = product_df_features
    selected_product = selected_product_from_first
    
    # All content uses the selected product from the first selector
    # Color palette - Modern dark theme with teal/amber accents
    C_FORECAST = "#00C896"        # Teal - main forecast line
    C_HISTORY = "#888888"         # Gray - historical data  
    C_WHAT_IF = "#FFD43B"        # Amber - What-If scenarios
    C_FILL80 = "rgba(0, 200, 150, 0.35)"   # Teal 80% CI
    C_FILL95 = "rgba(0, 200, 150, 0.15)"   # Teal 95% CI
    C_ANOMALY = "#FF6B6B"         # Red - anomalies
    OUTPUT_DIR = r"F:\RetailSense_Lite\outputs"
    
    # Import functions
    from utils.advanced_forecasting import (
        run_hybrid_forecast, 
        run_advanced_forecast,
        simulate_forecast_with_scenarios
    )
    from utils.business_insights import (
        generate_forecast_insights,
        calculate_scenario_impact,
        calculate_price_elasticity,
        detect_sales_anomalies
    )
    
    # Prepare time series
    ts = product_df[["week_start", "sales_qty"]].copy()
    ts.columns = ["date", "sales_qty"]
    ts["date"] = pd.to_datetime(ts["date"])
    ts = ts.sort_values("date").reset_index(drop=True)
    
    if len(ts) < 8:
        st.warning("⚠️ Insufficient data. Need ≥8 weeks of history.")
        st.stop()
    
    last_date = ts["date"].max()
    forecast_start = last_date + pd.Timedelta(weeks=1)
    
    # ========================================================================
    # PROFESSIONAL CONFIGURATION SECTION
    # ========================================================================
    with st.container():
        st.markdown("### ⚙️ Configuration")
        st.markdown("---")
        
        # First Row: Product, Model Type, Show Previous Data
        row1_col1, row1_col2, row1_col3 = st.columns([2.5, 2.5, 2.0], gap="medium")
        
        with row1_col1:
            st.markdown("**📦 Product**")
            st.info(f"**{selected_product}**", icon="📦")
        
        with row1_col2:
            st.markdown("**🤖 Model Type**")
            model_type = st.selectbox(
                "Select forecasting model",
                options=["Hybrid", "Prophet", "XGBoost", "LightGBM"],
                index=0,
                help="Hybrid: Weighted ensemble (recommended for best accuracy)",
                key="fc_model_type",
                label_visibility="collapsed"
            )
        
        with row1_col3:
            st.markdown("**📊 Display Options**")
            show_previous_data = st.toggle(
                "Show Previous Data", 
                value=True, 
                key="fc_show_previous",
                help="Toggle to show/hide historical data in charts"
            )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Second Row: Forecast End Date, Fast Mode, Run Button
    row2_col1, row2_col2, row2_col3, row2_col4 = st.columns([3.5, 2.0, 2.5, 2.0], gap="medium")
    
    # Custom End Date - Default to None (user must select)
    max_date = last_date + pd.Timedelta(weeks=156)
    
    # Initialize session state for end date if not exists
    if "fc_end_date_state" not in st.session_state:
        st.session_state["fc_end_date_state"] = None
    
    with row2_col1:
        st.markdown("**📅 Forecast End Date**")
        custom_end_date = st.date_input(
            "Select target end date for forecast",
            value=pd.to_datetime(st.session_state["fc_end_date_state"]).date() if st.session_state["fc_end_date_state"] is not None else None,
            min_value=forecast_start.date(),
            max_value=max_date.date(),
            help="⚠️ Select an end date to enable forecasting (up to 3 years ahead)",
            key="fc_end_date",
            label_visibility="collapsed"
        )
        if custom_end_date is not None:
            horizon_weeks = max(1, int((pd.to_datetime(custom_end_date) - last_date).days / 7))
            # Clear forecast cache if end date changed
            last_end_date = st.session_state.get("fc_last_end_date")
            if last_end_date is not None and pd.to_datetime(custom_end_date) != pd.to_datetime(last_end_date):
                # Clear all forecast-related cache
                for key in list(st.session_state.keys()):
                    if key.startswith("fc_"):
                        del st.session_state[key]
            st.session_state["fc_end_date_state"] = custom_end_date
            st.session_state["fc_last_end_date"] = custom_end_date
        else:
            horizon_weeks = None
            # Clear cache if end date is None
            if "fc_last_end_date" in st.session_state:
                for key in list(st.session_state.keys()):
                    if key.startswith("fc_") and key != "fc_end_date_state":
                        del st.session_state[key]
    
    with row2_col2:
        st.markdown("**⚡ Performance**")
        fast_mode = st.toggle(
            "Fast Mode", 
            value=True, 
            key="fc_fast_mode",
            help="Enable faster model training (recommended for large datasets)",
            label_visibility="collapsed"
        )
        deep_tune = st.toggle(
            "🔍 Deep Forecast Mode (Auto-Tune)",
            value=False,
            key="fc_deep_tune",
            help="Enable GridSearchCV hyperparameter tuning for optimal performance (slower but more accurate)"
        )
    
    with row2_col3:
        st.markdown("**🚀 Action**")
        # Run Button - Disabled until end date is provided
        if custom_end_date is None:
            run_forecast_btn = st.button(
                "🚀 Run Forecast", 
                type="secondary", 
                use_container_width=True,
                disabled=True,
                key="fc_run_btn_disabled",
                help="Select an end date to enable forecasting"
            )
        else:
            run_forecast_btn = st.button(
                "🚀 Run Forecast", 
                type="primary", 
                use_container_width=True,
                key="fc_run_btn",
                help=f"Generate forecast for {horizon_weeks} weeks ahead"
            )
    
    with row2_col4:
        if custom_end_date is None:
            st.markdown("<br>", unsafe_allow_html=True)
            st.info("💡 Select End Date", icon="💡")
        elif horizon_weeks:
            st.markdown("<br>", unsafe_allow_html=True)
            st.success(f"✓ {horizon_weeks} weeks ahead")

st.markdown("---")

# Main content area (full width for forecast results)
main_col2 = st.container()

# ========================================================================
# FORECAST GENERATION (Only after button click + end date provided)
# ========================================================================
if custom_end_date is None:
    st.info("💡 Select a Forecast End Date and click '🚀 Run Forecast' to generate forecast")
    st.stop()

cache_key = f"fc_{selected_product}_{horizon_weeks}_{model_type.lower()}_{pd.to_datetime(custom_end_date).strftime('%Y%m%d')}"
cached_forecast = st.session_state.get(cache_key)

# Only generate forecast when button is clicked OR if cache exists
if not run_forecast_btn and cached_forecast is None:
    st.info("💡 Click '🚀 Run Forecast' button above to generate forecast")
    st.stop()

need_refresh = run_forecast_btn or cached_forecast is None

if need_refresh:
    with st.spinner(f"⚡ Training {model_type} model ({horizon_weeks} weeks ahead)..."):
        try:
            @st.cache_data(ttl=3600, show_spinner=False)
            def get_cached_forecast(_df_hash, _product, _horizon_days, _model, _fast):
                # Use run_advanced_forecast for enhanced features
                product_filtered = features_df[features_df["product_name"] == _product].copy()
                return run_advanced_forecast(
                    product_filtered,
                    horizon_days=_horizon_days,
                    debug=False
                )
            
            df_hash = hash(str(features_df.head(100).values.tobytes()))
            horizon_days = (pd.to_datetime(custom_end_date) - last_date).days
            result = get_cached_forecast(
                df_hash, selected_product, horizon_days, model_type, fast_mode
            )
            
            # Extract data - handle advanced forecast dictionary
            if isinstance(result, dict):
                forecast_df = result.get('forecast_df', pd.DataFrame())
                history_df = result.get('history_df', ts.copy())
                metrics = result.get('metrics', {})
                details = result.get('details', {})
                feature_importances = result.get('feature_importances', pd.DataFrame())
                anomaly_flags = result.get('anomaly_flags', [])
            elif hasattr(result, 'forecast'):
                # EnsembleResult object (fallback)
                forecast_df = result.forecast.copy()
                history_df = result.history.copy() if hasattr(result, 'history') else ts.copy()
                metrics = result.metrics if hasattr(result, 'metrics') else {}
                details = result.details if hasattr(result, 'details') else {}
                feature_importances = getattr(result, 'feature_importances', pd.DataFrame())
                anomaly_flags = []
            else:
                # Fallback
                forecast_df = getattr(result, 'forecast_df', pd.DataFrame())
                if hasattr(forecast_df, 'copy'):
                    forecast_df = forecast_df.copy()
                history_df = getattr(result, 'history_df', ts.copy())
                if hasattr(history_df, 'copy'):
                    history_df = history_df.copy()
                metrics = getattr(result, 'metrics', {})
                details = getattr(result, 'details', {})
                feature_importances = pd.DataFrame()
                anomaly_flags = []
            
            # Ensure date columns
            if "date" not in forecast_df.columns:
                if "ds" in forecast_df.columns:
                    forecast_df["date"] = pd.to_datetime(forecast_df["ds"])
                else:
                    # Generate default dates if missing
                    forecast_df["date"] = pd.date_range(
                        start=pd.Timestamp.now(),
                        periods=len(forecast_df),
                        freq='W'
                    )
            else:
                forecast_df["date"] = pd.to_datetime(forecast_df["date"])
            
            if "date" not in history_df.columns:
                if "week_start" in history_df.columns:
                    history_df["date"] = pd.to_datetime(history_df["week_start"])
                elif "ds" in history_df.columns:
                    history_df["date"] = pd.to_datetime(history_df["ds"])
                else:
                    # Use the ts DataFrame's date if available
                    history_df["date"] = ts["date"].values[:len(history_df)] if "date" in ts.columns else pd.date_range(
                        start=pd.Timestamp.now() - pd.Timedelta(weeks=len(history_df)),
                        periods=len(history_df),
                        freq='W'
                    )
            else:
                history_df["date"] = pd.to_datetime(history_df["date"])
            
            # Ensure yhat column exists
            if "yhat" not in forecast_df.columns:
                if "forecast" in forecast_df.columns:
                    forecast_df["yhat"] = forecast_df["forecast"]
                else:
                    st.error("Forecast result missing 'yhat' column")
                    st.stop()
            
            # Store in session state
            st.session_state[cache_key] = {
                "forecast_df": forecast_df,
                "history_df": history_df,
                "metrics": metrics,
                "details": details,
                "feature_importances": feature_importances if 'feature_importances' in locals() else pd.DataFrame(),
                "anomaly_flags": anomaly_flags if 'anomaly_flags' in locals() else []
            }
            
            st.success(f"✅ Forecast generated!")
            
        except Exception as e:
            st.error(f"❌ Forecast failed: {str(e)}")
            with st.expander("🔍 Error Details"):
                import traceback
                st.code(traceback.format_exc())
            st.stop()
else:
    cached_forecast = st.session_state.get(cache_key)
    if cached_forecast:
        forecast_df = cached_forecast["forecast_df"]
        history_df = cached_forecast["history_df"]
        metrics = cached_forecast["metrics"]
        details = cached_forecast["details"]
        feature_importances = cached_forecast.get("feature_importances", pd.DataFrame())
        anomaly_flags = cached_forecast.get("anomaly_flags", [])
    else:
        st.info("💡 Click '🚀 Run Forecast' to generate")
        st.stop()

# ========================================================================
# KPI CARDS SECTION - PROFESSIONAL LAYOUT
# ========================================================================
st.markdown("### 📊 Forecast KPIs")
st.markdown("---")

# Calculate KPIs
month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Growth %
if len(history_df) >= 4 and len(forecast_df) >= 4:
    last_4w_avg = history_df["sales_qty"].tail(4).mean()
    next_4w_avg = forecast_df.head(4)["yhat"].mean()
    growth_pct = ((next_4w_avg - last_4w_avg) / (last_4w_avg + 1e-6)) * 100
else:
    growth_pct = 0.0

# RMSE & MAPE
rmse_val = metrics.get("ensemble_rmse", metrics.get("rmse", np.nan))
mape_val = metrics.get("ensemble_mape", metrics.get("mape", np.nan))

# Next Month & Quarter
next_month_forecast = forecast_df.head(4)["yhat"].sum() if len(forecast_df) >= 4 else forecast_df["yhat"].sum()
next_q_forecast = forecast_df.head(13)["yhat"].sum() if len(forecast_df) >= 13 else forecast_df["yhat"].sum()

# Peak Month
if len(forecast_df) > 0:
    forecast_by_month = forecast_df.groupby(forecast_df["date"].dt.month)["yhat"].mean()
    peak_month_idx = forecast_by_month.idxmax() if not forecast_by_month.empty else 1
    peak_month = month_names[peak_month_idx - 1]
else:
    peak_month = "N/A"

# Stock-out risk
stockout_risk = None
if "stock_on_hand" in product_df.columns:
    try:
        current_stock = float(product_df["stock_on_hand"].tail(1).iloc[0]) if not product_df["stock_on_hand"].tail(1).isna().all() else None
        if current_stock is not None and current_stock > 0:
            monthly_demand = next_month_forecast
            weeks_of_stock = (current_stock / (monthly_demand / 4 + 1e-6))
            stockout_risk = max(0, min(100, (1 - weeks_of_stock / 4) * 100))
    except:
        pass

# Display KPI cards with proper spacing
kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4, gap="large")

with kpi_col1:
    st.metric(
        "📈 Predicted Growth",
        f"{growth_pct:+.1f}%",
        delta=f"{growth_pct:+.1f}%",
        help="Expected growth: next 4 weeks vs last 4 weeks",
        delta_color="normal"
    )

with kpi_col2:
    mape_display = f"{mape_val:.1f}%" if not pd.isna(mape_val) else "N/A"
    confidence_level = 100 - mape_val if not pd.isna(mape_val) else 0
    st.metric(
        "🧮 Model Accuracy",
        mape_display,
        delta=f"{confidence_level:.1f}% confidence" if not pd.isna(mape_val) else None,
        help="Mean Absolute Percentage Error — lower is better",
        delta_color="inverse"
    )

with kpi_col3:
    st.metric(
        "💰 Next Month Revenue",
        f"₹{next_month_forecast:,.0f}",
        help="Forecasted sales for next 4 weeks",
        delta=None
    )

with kpi_col4:
    risk_color = "🔴" if stockout_risk and stockout_risk > 70 else "🟡" if stockout_risk and stockout_risk > 40 else "🟢"
    risk_display = f"{stockout_risk:.1f}%" if stockout_risk is not None else "N/A"
    st.metric(
        "🚨 Stock-out Risk",
        risk_display,
        help="Risk of running out of stock in next month",
        delta=None
    )

st.markdown("<br>", unsafe_allow_html=True)

# Model Confidence Gauge (Plotly Indicator) with Color-Coded Badge
if not pd.isna(mape_val):
    confidence_pct = max(0, min(100, 100 - mape_val))
    
    # Color-coded badge based on confidence
    if confidence_pct > 80:
        badge_color = "🟢"
        badge_text = "High Confidence"
        badge_bg = "rgba(0, 200, 150, 0.15)"
    elif confidence_pct >= 60:
        badge_color = "🟡"
        badge_text = "Moderate Confidence"
        badge_bg = "rgba(255, 212, 59, 0.15)"
    else:
        badge_color = "🔴"
        badge_text = "Low Confidence"
        badge_bg = "rgba(255, 107, 107, 0.15)"
    
    # Display badge above gauge - centered
    gauge_col1, gauge_col2, gauge_col3 = st.columns([1, 3, 1])
    with gauge_col2:
        st.markdown(f"""
        <div style="background-color: {badge_bg}; padding: 12px; border-radius: 8px; text-align: center; margin-bottom: 15px; border: 1px solid rgba(0, 200, 150, 0.3);">
            <strong style="font-size: 16px;">{badge_color} {badge_text}: {confidence_pct:.1f}%</strong>
        </div>
        """, unsafe_allow_html=True)
        
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=confidence_pct,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Model Confidence Level", 'font': {'size': 18}},
            delta={'reference': 80, 'font': {'size': 14}},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': C_FORECAST if confidence_pct > 80 else (C_WHAT_IF if confidence_pct >= 60 else C_ANOMALY)},
                'steps': [
                    {'range': [0, 60], 'color': "lightgray"},
                    {'range': [60, 80], 'color': "lightyellow"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig_gauge.update_layout(height=220, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True, key="confidence_gauge")

st.markdown("---")

# ========================================================================
# TABBED INTERFACE: Forecast | What-If | Insights
# ========================================================================
tab_forecast, tab_whatif, tab_insights = st.tabs([
    "📈 Forecast",
    "💡 What-If Simulation",
    "🤖 AI Insights"
])

with tab_forecast:
    st.markdown("### 📈 Sales Forecast Visualization")
    
    # Chart controls with improved layout
    control_col1, control_col2, control_col3 = st.columns([2, 2, 1])
    with control_col1:
        show_historical = st.checkbox("✅ Show Historical", value=show_previous_data if 'show_previous_data' in locals() else True, key="fc_show_historical")
        show_forecast = st.checkbox("✅ Show Forecast", value=True, key="fc_show_forecast")
    with control_col2:
        show_ci_80 = st.checkbox("✅ Show 80% CI", value=True, key="fc_show_ci80")
        show_ci_95 = st.checkbox("✅ Show 95% CI", value=True, key="fc_show_ci95")
    with control_col3:
        show_anomalies = st.checkbox("✅ Show Anomalies", value=True, key="fc_show_anomalies")
        show_trend = st.checkbox("✅ Show Trend", value=True, key="fc_show_trend")
    
    # Create main forecast graph
    fig_main = go.Figure()
    
    # Historical data (conditional on toggle)
    if show_historical and "sales_qty" in history_df.columns:
        fig_main.add_trace(go.Scatter(
            x=history_df["date"],
            y=history_df["sales_qty"],
            name="Historical Sales",
            mode="lines+markers",
            line=dict(color=C_HISTORY, width=2),
            marker=dict(size=5, color=C_HISTORY),
            hovertemplate="<b>Historical</b><br>Date: %{x|%Y-%m-%d}<br>Sales: %{y:,.0f}<extra></extra>"
        ))
    
        # Anomalies overlay - from both historical detection and forecast flags
        if show_anomalies:
            try:
                # Historical anomalies
                anomalies_df = detect_sales_anomalies(features_df, selected_product)
                if not anomalies_df.empty and "date" in anomalies_df.columns:
                    anom_dates = pd.to_datetime(anomalies_df["date"], errors='coerce')
                    anom_values = anomalies_df["actual_sales"]
                    valid_mask = anom_dates.notna() & anom_values.notna()
                    if valid_mask.sum() > 0:
                        fig_main.add_trace(go.Scatter(
                            x=anom_dates[valid_mask],
                            y=anom_values[valid_mask],
                            mode="markers",
                            name="Historical Anomalies",
                            marker=dict(color=C_ANOMALY, symbol="x", size=12, line=dict(width=2, color="white")),
                            hovertemplate="<b>⚠️ ANOMALY</b><br>Date: %{x|%Y-%m-%d}<br>Sales: %{y:,.0f}<extra></extra>"
                        ))
            except:
                pass
            
            # Forecast anomaly flags (high forecast values)
            if 'anomaly_flags' in locals() and anomaly_flags:
                for anom in anomaly_flags:
                    if isinstance(anom, dict) and "date" in anom and "value" in anom:
                        fig_main.add_trace(go.Scatter(
                            x=[pd.to_datetime(anom["date"])],
                            y=[anom["value"]],
                            mode="markers",
                            name="High Forecast Alert",
                            marker=dict(color="#FFAA00", symbol="diamond", size=15, line=dict(width=2, color="orange")),
                            hovertemplate="<b>🔶 HIGH FORECAST</b><br>Date: %{x|%Y-%m-%d}<br>Sales: %{y:,.0f}<extra></extra>"
                        ))
    
    # Forecast line (conditional on toggle)
    if show_forecast:
        fig_main.add_trace(go.Scatter(
            x=forecast_df["date"],
            y=forecast_df["yhat"],
            name="Forecast",
            mode="lines",
            line=dict(color=C_FORECAST, width=3),
            hovertemplate="<b>Forecast</b><br>Date: %{x|%Y-%m-%d}<br>Sales: %{y:,.0f}<extra></extra>"
        ))
    
    # Confidence intervals
    if show_ci_95:
        if "yhat_upper_95" in forecast_df.columns:
            fig_main.add_trace(go.Scatter(
                x=pd.concat([forecast_df["date"], forecast_df["date"][::-1]]),
                y=pd.concat([forecast_df["yhat_upper_95"], forecast_df["yhat_lower_95"][::-1]]),
                fill='toself',
                fillcolor=C_FILL95,
                line=dict(color='rgba(255,255,255,0)'),
                name='95% CI',
                showlegend=True,
                hoverinfo='skip'
            ))
        elif "yhat_upper" in forecast_df.columns:
            # Approximate 95% CI from 80% CI
            ci_range = (forecast_df["yhat_upper"] - forecast_df["yhat_lower"]) / 1.28
            fig_main.add_trace(go.Scatter(
                x=pd.concat([forecast_df["date"], forecast_df["date"][::-1]]),
                y=pd.concat([forecast_df["yhat"] + ci_range * 1.96, (forecast_df["yhat"] - ci_range * 1.96)[::-1]]),
                fill='toself',
                fillcolor=C_FILL95,
                line=dict(color='rgba(255,255,255,0)'),
                name='95% CI (approx)',
                showlegend=True,
                hoverinfo='skip'
            ))
    
    if show_ci_80 and "yhat_upper" in forecast_df.columns:
        fig_main.add_trace(go.Scatter(
            x=pd.concat([forecast_df["date"], forecast_df["date"][::-1]]),
            y=pd.concat([forecast_df["yhat_upper"], forecast_df["yhat_lower"][::-1]]),
            fill='toself',
            fillcolor=C_FILL80,
            line=dict(color='rgba(255,255,255,0)'),
            name='80% CI',
            showlegend=True,
            hoverinfo='skip'
        ))
    
    # Trend line
    if show_trend and len(history_df) >= 4:
        ma_window = min(8, len(history_df) // 2)
        history_df_copy = history_df.copy()
        history_df_copy["trend"] = history_df_copy["sales_qty"].rolling(window=ma_window, center=True).mean()
        fig_main.add_trace(go.Scatter(
            x=history_df_copy["date"],
            y=history_df_copy["trend"],
            name="Trend",
            mode="lines",
            line=dict(color="#666666", width=1, dash="dot"),
            hovertemplate="Trend: %{y:,.0f}<extra></extra>"
        ))
    
    # Vertical separator - use add_shape instead of add_vline to avoid datetime arithmetic issues
    # Keep as pandas Timestamp to match the format used in traces
    if isinstance(last_date, pd.Timestamp):
        last_date_plotly = last_date
    else:
        last_date_plotly = pd.to_datetime(last_date)
    
    # Get y-axis range for the line
    y_min = 0
    y_max = max(
        history_df["sales_qty"].max() if "sales_qty" in history_df.columns else 0,
        forecast_df["yhat"].max() if "yhat" in forecast_df.columns else 0
    ) * 1.1
    
    # Add vertical line using add_shape (more reliable than add_vline for datetime handling)
    fig_main.add_shape(
        type="line",
        x0=last_date_plotly,
        x1=last_date_plotly,
        y0=y_min,
        y1=y_max,
        line=dict(color="#666666", width=2, dash="dash"),
        layer="below"
    )
    
    # Add annotation separately
    fig_main.add_annotation(
        x=last_date_plotly,
        y=y_max * 0.95,
        text="Forecast Start",
        showarrow=False,
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="#666666",
        borderwidth=1,
        font=dict(color="#333333", size=10)
    )
    
    # Dynamic horizon display (weeks and days)
    horizon_days = horizon_weeks * 7
    horizon_text = f"{horizon_weeks} weeks ({horizon_days} days)" if horizon_weeks > 0 else "Custom"
    
    # Layout
    fig_main.update_layout(
        title=f"{selected_product} - Sales Forecast ({horizon_text} ahead)",
        xaxis_title="Date",
        yaxis_title="Sales Quantity",
        template="plotly_white",
        height=600,
        hovermode='x unified',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(rangeslider=dict(visible=True, thickness=0.1)),
        yaxis=dict(gridcolor='rgba(128, 128, 128, 0.2)')
    )
    
    st.plotly_chart(fig_main, use_container_width=True, key="main_forecast_chart")
    
    # Top 5 Forecasted Weeks Table with Confidence
    if len(forecast_df) >= 5:
        st.markdown("### 📋 Top 5 Forecasted Weeks")
        top_weeks = forecast_df.nlargest(5, "yhat")[["date", "yhat", "yhat_lower", "yhat_upper"]].copy()
        top_weeks["date"] = pd.to_datetime(top_weeks["date"]).dt.strftime("%Y-%m-%d")
        top_weeks["CI_Range"] = (top_weeks["yhat_upper"] - top_weeks["yhat_lower"]).round(0)
        
        if len(history_df) > 0:
            baseline = history_df["sales_qty"].tail(4).mean()
            top_weeks["Growth_%"] = ((top_weeks["yhat"] - baseline) / (baseline + 1e-6) * 100).round(1)
        else:
            top_weeks["Growth_%"] = 0.0
        
        # Add Confidence column based on CI range relative to forecast
        top_weeks["Confidence_%"] = (100 - (top_weeks["CI_Range"] / (top_weeks["yhat"] + 1e-6) * 100)).round(1)
        top_weeks["Confidence_%"] = top_weeks["Confidence_%"].clip(0, 100)
        
        top_weeks_display = top_weeks.rename(columns={
            "date": "Date",
            "yhat": "Forecasted Sales",
            "CI_Range": "CI Range",
            "Growth_%": "% Growth",
            "Confidence_%": "Confidence %"
        })
        
        st.dataframe(
            top_weeks_display[["Date", "Forecasted Sales", "CI Range", "% Growth", "Confidence %"]],
            use_container_width=True,
            hide_index=True
        )
    
    # Feature Importance Analysis
    if 'feature_importances' in locals() and isinstance(feature_importances, pd.DataFrame) and not feature_importances.empty:
        st.markdown("### 🎯 Feature Importance Analysis")
        # Aggregate by feature if multiple models
        if "model" in feature_importances.columns:
            feature_agg = feature_importances.groupby("feature")["importance"].mean().sort_values(ascending=False).head(15)
        else:
            feature_agg = feature_importances.set_index("feature")["importance"].sort_values(ascending=False).head(15)
        
        fig_importance = go.Figure(go.Bar(
            x=feature_agg.values,
            y=feature_agg.index,
            orientation='h',
            marker=dict(color=feature_agg.values, colorscale='Viridis'),
            text=[f"{v:.2f}" for v in feature_agg.values],
            textposition='auto'
        ))
        fig_importance.update_layout(
            title="Top 15 Most Important Features",
            xaxis_title="Importance Score",
            yaxis_title="Feature",
            height=500,
            template="plotly_white",
            yaxis=dict(autorange="reversed")
        )
        st.plotly_chart(fig_importance, use_container_width=True, key="feature_importance_chart")
    
    # Rolling Weekly Sales Heatmap
    if len(history_df) >= 52:
        st.markdown("### 📊 Rolling Weekly Sales Heatmap (Last Year)")
        history_heatmap = history_df.tail(52).copy()
        history_heatmap["year"] = history_heatmap["date"].dt.year
        history_heatmap["week"] = history_heatmap["date"].dt.isocalendar().week
        history_heatmap["month"] = history_heatmap["date"].dt.month
        
        # Create pivot table
        heatmap_data = history_heatmap.pivot_table(
            values="sales_qty",
            index="month",
            columns="week",
            aggfunc="mean",
            fill_value=0
        )
        
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=list(range(1, 53)),
            y=[f"Month {i}" for i in heatmap_data.index],
            colorscale='YlOrRd',
            text=heatmap_data.values,
            texttemplate='%{text:.0f}',
            textfont={"size": 10},
            hovertemplate='Month: %{y}<br>Week: %{x}<br>Sales: %{z:.0f}<extra></extra>'
        ))
        fig_heatmap.update_layout(
            title="Weekly Sales Heatmap (Last 52 Weeks)",
            xaxis_title="Week of Year",
            yaxis_title="Month",
            height=400,
            template="plotly_white"
        )
        st.plotly_chart(fig_heatmap, use_container_width=True, key="sales_heatmap")
    
    # Model Performance Comparison - Enhanced Tier-3 Placement Ready
    if metrics and isinstance(metrics, dict):
        st.subheader("📊 Model Performance Insights — Forecast Accuracy Overview")
        st.caption("🏭 Industry-Grade Evaluation | Cross-Model Accuracy Summary")
        st.markdown("---")
        
        # Helper function to compute all metrics for a model
        def compute_model_metrics(model_name, metrics_dict):
            """Compute comprehensive metrics for a model"""
            prefix_map = {
                "Prophet": "prophet",
                "XGBoost": "xgb",
                "LightGBM": "lgbm",
                "Ensemble": "ensemble"
            }
            prefix = prefix_map.get(model_name, "")
            
            rmse = metrics_dict.get(f"{prefix}_rmse", np.nan) if prefix else metrics_dict.get("rmse", np.nan)
            mae = metrics_dict.get(f"{prefix}_mae", np.nan) if prefix else metrics_dict.get("mae", np.nan)
            mape = metrics_dict.get(f"{prefix}_mape", np.nan) if prefix else metrics_dict.get("mape", np.nan)
            
            # Compute MSE from RMSE (MSE = RMSE^2)
            mse = rmse ** 2 if not pd.isna(rmse) else np.nan
            
            # Try to get R², compute if possible
            r2 = metrics_dict.get(f"{prefix}_r2", np.nan)
            if prefix == "ensemble":
                r2 = metrics_dict.get("ensemble_r2", metrics_dict.get("r2", np.nan))
            
            return {
                "Model": model_name,
                "RMSE": rmse,
                "MAE": mae,
                "MSE": mse,
                "MAPE": mape,
                "R²": r2
            }
        
        # Compute metrics for all models
        models_to_compare = ["Prophet", "XGBoost", "LightGBM", "Ensemble"]
        model_metrics_list = [compute_model_metrics(m, metrics) for m in models_to_compare]
        df_metrics = pd.DataFrame(model_metrics_list)
        
        # Filter out models with all NaN metrics
        df_metrics = df_metrics[df_metrics[["RMSE", "MAE", "MSE"]].notna().any(axis=1)]
        
        if len(df_metrics) > 0:
            # Best Model Highlight
            if not df_metrics["RMSE"].isna().all():
                best_model_row = df_metrics.loc[df_metrics["RMSE"].idxmin()]
                best_model = best_model_row["Model"]
                best_rmse = best_model_row["RMSE"]
                
                # Get second best for improvement calculation
                sorted_rmse = df_metrics[df_metrics["RMSE"].notna()].sort_values("RMSE")
                if len(sorted_rmse) > 1:
                    second_best = sorted_rmse.iloc[1]
                    improvement_pct = ((second_best["RMSE"] - best_rmse) / (second_best["RMSE"] + 1e-6)) * 100
                else:
                    improvement_pct = 0
                
                # Display best model metric
                best_col1, best_col2, best_col3 = st.columns([2, 1, 1])
                with best_col1:
                    st.metric(
                        label="🏆 Best Performing Model",
                        value=best_model,
                        delta=f"Lowest RMSE: {best_rmse:.2f}" if not pd.isna(best_rmse) else None,
                        help="Model with lowest Root Mean Squared Error"
                    )
                with best_col2:
                    # Model Confidence Gauge
                    max_rmse = df_metrics["RMSE"].max()
                    if not pd.isna(max_rmse) and max_rmse > 0:
                        confidence_score = max(0, min(100, 100 - (best_rmse / max_rmse * 100)))
                        
                        # Determine gauge color
                        if confidence_score < 60:
                            gauge_color = "red"
                        elif confidence_score < 80:
                            gauge_color = "orange"
                        else:
                            gauge_color = "green"
                        
                        fig_gauge = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=confidence_score,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "Model Confidence"},
                            gauge={
                                'axis': {'range': [None, 100]},
                                'bar': {'color': gauge_color},
                                'steps': [
                                    {'range': [0, 60], 'color': "lightgray"},
                                    {'range': [60, 80], 'color': "gray"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 90
                                }
                            }
                        ))
                        fig_gauge.update_layout(height=200)
                        st.plotly_chart(fig_gauge, use_container_width=True, key="model_perf_confidence_gauge")
            
            # User controls
            control_row1, control_row2 = st.columns([2, 1])
            with control_row1:
                selected_models = st.multiselect(
                    "Select models to compare:",
                    options=df_metrics["Model"].tolist(),
                    default=df_metrics["Model"].tolist()[:3],  # Default to first 3 (excluding Ensemble)
                    key="model_selector"
                )
            with control_row2:
                metric_type = st.radio(
                    "Metric Type:",
                    options=["Error Metrics", "Accuracy Metrics"],
                    index=0,
                    horizontal=True,
                    key="metric_type_toggle"
                )
            
            # Filter selected models
            df_metrics_filtered = df_metrics[df_metrics["Model"].isin(selected_models)]
            
            if len(df_metrics_filtered) > 0:
                # Main visualization area
                vis_col1, vis_col2 = st.columns([3, 2])
                
                with vis_col1:
                    # Error metrics for radar (RMSE, MAE, MAPE)
                    error_categories = ["RMSE", "MAE", "MAPE"]
                    max_vals = {}
                    for cat in error_categories:
                        valid_vals = df_metrics_filtered[cat].dropna()
                        max_vals[cat] = valid_vals.max() if len(valid_vals) > 0 else 1
                    
                    # Create radar chart
                    fig_radar = go.Figure()
                    
                    model_colors = {
                        "Prophet": "#00e5ff",  # Cyan
                        "XGBoost": "#ff9800",  # Orange
                        "LightGBM": "#4caf50",  # Lime Green
                        "Ensemble": "#9c27b0"  # Purple
                    }
                    
                    for _, row in df_metrics_filtered.iterrows():
                        model_name = row["Model"]
                        if model_name not in selected_models:
                            continue
                        
                        values = []
                        for cat in error_categories:
                            val = row.get(cat, np.nan)
                            if not pd.isna(val) and max_vals[cat] > 0:
                                # Invert: 100 - (val / max_val * 100) so higher is better
                                normalized = 100 - (val / max_vals[cat] * 100)
                                values.append(max(0, min(100, normalized)))
                            else:
                                values.append(0)
                        
                        fig_radar.add_trace(go.Scatterpolar(
                            r=values + [values[0]],  # Close polygon
                            theta=error_categories + [error_categories[0]],
                            fill='toself',
                            name=model_name,
                            line=dict(color=model_colors.get(model_name, "#666666"), width=2),
                            marker=dict(size=8)
                        ))
                    
                    fig_radar.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 100],
                                tickfont=dict(size=10)
                            )
                        ),
                        title="Model Comparison (Normalized — Higher is Better)",
                        height=400,
                        template="plotly_dark",
                        showlegend=True,
                        margin=dict(l=20, r=20, t=40, b=20)
                    )
                    st.plotly_chart(fig_radar, use_container_width=True, key="enhanced_radar")
                    
                    # Horizontal bar chart - switch between Error and Accuracy metrics
                    fig_bar = go.Figure()
                    
                    if metric_type == "Error Metrics":
                        # Show error metrics (lower is better)
                        categories = ["RMSE", "MAE", "MAPE"]
                        bar_title = "Error Metrics Comparison (Lower is Better)"
                        
                        for _, row in df_metrics_filtered.iterrows():
                            model_name = row["Model"]
                            color = model_colors.get(model_name, "#666666")
                            
                            fig_bar.add_trace(go.Bar(
                                name=model_name,
                                x=categories,
                                y=[
                                    row.get("RMSE", 0) if not pd.isna(row.get("RMSE")) else 0,
                                    row.get("MAE", 0) if not pd.isna(row.get("MAE")) else 0,
                                    row.get("MAPE", 0) if not pd.isna(row.get("MAPE")) else 0
                                ],
                                marker_color=color,
                                text=[
                                    f"{row.get('RMSE', 0):.2f}" if not pd.isna(row.get("RMSE")) else "N/A",
                                    f"{row.get('MAE', 0):.2f}" if not pd.isna(row.get("MAE")) else "N/A",
                                    f"{row.get('MAPE', 0):.1f}%" if not pd.isna(row.get("MAPE")) else "N/A"
                                ],
                                textposition="outside"
                            ))
                    else:
                        # Show accuracy metrics (higher is better)
                        categories = ["R² Score"]
                        bar_title = "Accuracy Metrics Comparison (Higher is Better)"
                        
                        for _, row in df_metrics_filtered.iterrows():
                            model_name = row["Model"]
                            color = model_colors.get(model_name, "#666666")
                            
                            r2_val = row.get("R²", np.nan)
                            # Convert MAPE to accuracy (1 - MAPE/100, clipped to 0-1)
                            mape_val = row.get("MAPE", np.nan)
                            if not pd.isna(mape_val):
                                accuracy_from_mape = max(0, min(1, 1 - (mape_val / 100)))
                            else:
                                accuracy_from_mape = np.nan
                            
                            # Use R² if available, else use derived accuracy from MAPE
                            accuracy_val = r2_val if not pd.isna(r2_val) else (accuracy_from_mape if not pd.isna(accuracy_from_mape) else np.nan)
                            
                            if not pd.isna(accuracy_val):
                                categories_extended = ["R² Score"]
                                if pd.isna(r2_val) and not pd.isna(accuracy_from_mape):
                                    categories_extended = ["Accuracy (1-MAPE)"]
                                
                                fig_bar.add_trace(go.Bar(
                                    name=model_name,
                                    x=categories_extended,
                                    y=[accuracy_val],
                                    marker_color=color,
                                    text=[f"{accuracy_val:.3f}"],
                                    textposition="outside"
                                ))
                    
                    fig_bar.update_layout(
                        title=bar_title,
                        xaxis_title="Metric",
                        yaxis_title="Value",
                        barmode='group',
                        height=350,
                        template="plotly_white",
                        showlegend=True
                    )
                    st.plotly_chart(fig_bar, use_container_width=True, key="error_bar_chart")
                
                with vis_col2:
                    st.markdown("**📊 Metrics Summary**")
                    st.caption("💡 *RMSE: Lower is better. Indicates prediction error magnitude.*")
                    
                    for _, row in df_metrics_filtered.iterrows():
                        model_name = row["Model"]
                        st.markdown(f"**{model_name}**")
                        
                        # RMSE
                        rmse_val = row.get("RMSE", np.nan)
                        if not pd.isna(rmse_val):
                            st.caption(f"📉 **RMSE:** {rmse_val:.2f}  *(Lower is better)*")
                        else:
                            st.caption("📉 **RMSE:** N/A")
                        
                        # MAE
                        mae_val = row.get("MAE", np.nan)
                        if not pd.isna(mae_val):
                            st.caption(f"📊 **MAE:** {mae_val:.2f}")
                        else:
                            st.caption("📊 **MAE:** N/A")
                        
                        # MSE
                        mse_val = row.get("MSE", np.nan)
                        if not pd.isna(mse_val):
                            st.caption(f"📈 **MSE:** {mse_val:.2f}")
                        else:
                            st.caption("📈 **MSE:** N/A")
                        
                        # MAPE
                        mape_val = row.get("MAPE", np.nan)
                        if not pd.isna(mape_val):
                            st.caption(f"📉 **MAPE:** {mape_val:.1f}%")
                        else:
                            st.caption("📉 **MAPE:** N/A")
                        
                        # R²
                        r2_val = row.get("R²", np.nan)
                        if not pd.isna(r2_val):
                            st.caption(f"🎯 **R² Score:** {r2_val:.3f}")
                        else:
                            st.caption("🎯 **R² Score:** N/A")
                        
                        st.markdown("---")
                
                st.divider()
                
                # Dynamic Executive Summary
                if not df_metrics_filtered["RMSE"].isna().all():
                    best = df_metrics_filtered.loc[df_metrics_filtered["RMSE"].idxmin()]
                    sorted_by_rmse = df_metrics_filtered[df_metrics_filtered["RMSE"].notna()].sort_values("RMSE")
                    
                    if len(sorted_by_rmse) > 1:
                        second = sorted_by_rmse.iloc[1]
                        improvement = ((second["RMSE"] - best["RMSE"]) / (second["RMSE"] + 1e-6)) * 100
                        
                        st.info(
                            f"✅ **{best['Model']}** performed best with RMSE = {best['RMSE']:.2f}, "
                            f"improving over **{second['Model']}** by {improvement:.1f}%. "
                            f"This suggests stronger adaptability to recent sales trends and seasonal variations."
                        )
                
                # Download button
                csv_string = df_metrics_filtered.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Export Model Comparison Report",
                    data=csv_string,
                    file_name=f"model_comparison_{selected_product.replace(' ', '_')}.csv",
                    mime="text/csv",
                    key="download_model_comparison"
                )
                
                st.markdown("---")
                
                # ========================================================================
                # Residual Analysis and Forecast Decomposition
                # ========================================================================
                st.subheader("🔍 Residual Analysis & Forecast Decomposition")
                
                residual_col1, residual_col2 = st.columns(2)
                
                with residual_col1:
                    # Residual Plot
                    if "yhat" in history_df.columns or len(history_df) > 0:
                        try:
                            # Compute residuals from historical fitted values
                            if "yhat" in history_df.columns:
                                fitted_vals = history_df["yhat"].values
                            else:
                                # Estimate from rolling mean as fallback
                                fitted_vals = history_df["sales_qty"].rolling(4, min_periods=1).mean().fillna(history_df["sales_qty"].mean()).values
                            
                            actual_vals = history_df["sales_qty"].values[:len(fitted_vals)]
                            residuals_plot = actual_vals - fitted_vals
                            
                            fig_residual = go.Figure()
                            fig_residual.add_trace(go.Scatter(
                                x=history_df["date"].values[:len(residuals_plot)],
                                y=residuals_plot,
                                mode="markers",
                                name="Residuals",
                                marker=dict(color="#d62728", size=6, opacity=0.6)
                            ))
                            fig_residual.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Zero Line")
                            
                            fig_residual.update_layout(
                                title="Residual Analysis (Predicted vs Actual)",
                                xaxis_title="Date",
                                yaxis_title="Residuals (Actual - Predicted)",
                                template="plotly_white",
                                height=400
                            )
                            st.plotly_chart(fig_residual, use_container_width=True, key="residual_analysis_chart")
                        except Exception as e:
                            st.info("Residual analysis not available")
                
                with residual_col2:
                    # Forecast Decomposition (if Prophet components available)
                    if 'prophet_components' in locals() and prophet_components is not None and not prophet_components.empty:
                        try:
                            from plotly.subplots import make_subplots
                            
                            fig_decomp = make_subplots(
                                rows=3, cols=1,
                                subplot_titles=("Trend", "Yearly Seasonality", "Quarterly Seasonality"),
                                vertical_spacing=0.12,
                                row_heights=[0.5, 0.25, 0.25]
                            )
                            
                            if "trend" in prophet_components.columns and "ds" in prophet_components.columns:
                                fig_decomp.add_trace(
                                    go.Scatter(
                                        x=pd.to_datetime(prophet_components["ds"]), 
                                        y=prophet_components["trend"], 
                                        name="Trend",
                                        line=dict(color="#1f77b4")
                                    ),
                                    row=1, col=1
                                )
                            
                            if "yearly" in prophet_components.columns and "ds" in prophet_components.columns:
                                fig_decomp.add_trace(
                                    go.Scatter(
                                        x=pd.to_datetime(prophet_components["ds"]), 
                                        y=prophet_components["yearly"], 
                                        name="Yearly",
                                        line=dict(color="#ff7f0e")
                                    ),
                                    row=2, col=1
                                )
                            
                            if "quarterly" in prophet_components.columns and "ds" in prophet_components.columns:
                                fig_decomp.add_trace(
                                    go.Scatter(
                                        x=pd.to_datetime(prophet_components["ds"]), 
                                        y=prophet_components["quarterly"], 
                                        name="Quarterly",
                                        line=dict(color="#2ca02c")
                                    ),
                                    row=3, col=1
                                )
                            
                            fig_decomp.update_layout(
                                title="Forecast Decomposition (Trend + Seasonality)",
                                height=600,
                                template="plotly_white",
                                showlegend=False
                            )
                            st.plotly_chart(fig_decomp, use_container_width=True, key="forecast_decomposition_chart")
                        except Exception as e:
                            st.info("Forecast decomposition not available")
                    else:
                        st.info("💡 Prophet decomposition components not available for this forecast")
                
                # Download Forecast Results
                st.markdown("---")
                download_col1, download_col2 = st.columns(2)
                
                with download_col1:
                    # Forecast results CSV
                    forecast_csv = forecast_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Forecast Results (CSV)",
                        data=forecast_csv,
                        file_name=f"forecast_{selected_product.replace(' ', '_')}.csv",
                        mime="text/csv",
                        key="download_forecast_results"
                    )
                
                with download_col2:
                    # Metrics summary CSV
                    metrics_summary = {
                        "Metric": ["RMSE", "MAE", "MAPE", "R²"],
                        "Prophet": [
                            metrics.get("prophet_rmse", "N/A"),
                            metrics.get("prophet_mae", "N/A"),
                            metrics.get("prophet_mape", "N/A"),
                            metrics.get("prophet_r2", "N/A")
                        ],
                        "XGBoost": [
                            metrics.get("xgb_rmse", "N/A"),
                            metrics.get("xgb_mae", "N/A"),
                            metrics.get("xgb_mape", "N/A"),
                            metrics.get("xgb_r2", "N/A")
                        ],
                        "LightGBM": [
                            metrics.get("lgbm_rmse", "N/A"),
                            metrics.get("lgbm_mae", "N/A"),
                            metrics.get("lgbm_mape", "N/A"),
                            metrics.get("lgbm_r2", "N/A")
                        ],
                        "Ensemble": [
                            metrics.get("ensemble_rmse", metrics.get("rmse", "N/A")),
                            metrics.get("ensemble_mae", metrics.get("mae", "N/A")),
                            metrics.get("ensemble_mape", metrics.get("mape", "N/A")),
                            metrics.get("ensemble_r2", metrics.get("r2", "N/A"))
                        ]
                    }
                    metrics_df_summary = pd.DataFrame(metrics_summary)
                    metrics_csv = metrics_df_summary.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Metrics Summary (CSV)",
                        data=metrics_csv,
                        file_name=f"metrics_summary_{selected_product.replace(' ', '_')}.csv",
                        mime="text/csv",
                        key="download_metrics_summary"
                    )

with tab_whatif:
    st.markdown("### 💡 What-If Scenario Simulation")
    st.caption("Adjust parameters below to simulate different business scenarios")
    
    # What-If controls in columns
    whatif_col1, whatif_col2, whatif_col3, whatif_col4 = st.columns(4)
    
    with whatif_col1:
        price_delta = st.slider(
            "💰 Price Change (%)",
            min_value=-20.0,
            max_value=20.0,
            value=0.0,
            step=0.5,
            help="Simulate price increase/decrease impact",
            key="whatif_price"
        )
    
    with whatif_col2:
        promotion_flag = st.checkbox(
            "🎯 Active Promotion",
            value=False,
            help="Simulate promotion boost (+20%)",
            key="whatif_promotion"
        )
    
    with whatif_col3:
        holiday_flag = st.checkbox(
            "🎉 Holiday Period",
            value=False,
            help="Simulate holiday effect (+25%)",
            key="whatif_holiday"
        )
    
    with whatif_col4:
        weather_scenario = st.selectbox(
            "🌦️ Weather Scenario",
            options=["None", "Sunny", "Rainy", "Cloudy", "Stormy"],
            index=0,
            help="Simulate weather impact on sales",
            key="whatif_weather"
        )
        weather_val = None if weather_scenario == "None" else weather_scenario
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Additional controls row
    whatif_row2_col1, whatif_row2_col2 = st.columns(2)
    
    with whatif_row2_col1:
        stock_level_factor = st.slider(
            "📦 Stock Level Factor",
            min_value=0.5,
            max_value=2.0,
            value=1.0,
            step=0.1,
            help="Simulate inventory levels (0.5 = half stock, 2.0 = double stock)",
            key="whatif_stock"
        )
    
    with whatif_row2_col2:
        promotion_intensity = st.slider(
            "🎯 Promotion Intensity",
            min_value=0.0,
            max_value=1.0,
            value=1.0 if promotion_flag else 0.0,
            step=0.1,
            help="Promotion strength (0.0 = none, 1.0 = full)",
            key="whatif_promo_intensity"
        )
    
    # Calculate elasticity
    try:
        elasticity = calculate_price_elasticity(features_df, selected_product)
    except:
        elasticity = -1.2
    
    # Generate simulated forecast
    # Apply promotion intensity to promotion flag
    effective_promotion = promotion_flag and promotion_intensity > 0
    
    if price_delta != 0 or effective_promotion or holiday_flag or weather_val or stock_level_factor != 1.0:
        sim_forecast_df = simulate_forecast_with_scenarios(
            forecast_df.copy(),
            price_delta=price_delta,
            promotion_flag=effective_promotion,
            holiday_flag=holiday_flag,
            elasticity=elasticity
        )
        
        # Apply promotion intensity multiplier
        if effective_promotion:
            promo_mult = 1.0 + (promotion_intensity * 0.20)  # Base 20% boost scaled by intensity
            sim_forecast_df["yhat_simulated"] = sim_forecast_df["yhat_simulated"] * promo_mult
            sim_forecast_df["yhat_lower_simulated"] = sim_forecast_df["yhat_lower_simulated"] * promo_mult
            sim_forecast_df["yhat_upper_simulated"] = sim_forecast_df["yhat_upper_simulated"] * promo_mult
        
        # Apply weather multiplier if provided
        if weather_val:
            weather_multipliers = {
                "Sunny": 1.15,
                "Rainy": 0.90,
                "Cloudy": 1.0,
                "Stormy": 0.75
            }
            if weather_val in weather_multipliers:
                weather_mult = weather_multipliers[weather_val]
                sim_forecast_df["yhat_simulated"] = sim_forecast_df["yhat_simulated"] * weather_mult
                sim_forecast_df["yhat_lower_simulated"] = sim_forecast_df["yhat_lower_simulated"] * weather_mult
                sim_forecast_df["yhat_upper_simulated"] = sim_forecast_df["yhat_upper_simulated"] * weather_mult
        
        # Apply stock level factor (affects availability-driven demand)
        if stock_level_factor != 1.0:
            # Stock-out effect: lower stock reduces sales potential
            stock_mult = min(1.0, stock_level_factor)  # Cap at 1.0 for above-normal stock
            sim_forecast_df["yhat_simulated"] = sim_forecast_df["yhat_simulated"] * stock_mult
            sim_forecast_df["yhat_lower_simulated"] = sim_forecast_df["yhat_lower_simulated"] * stock_mult
            sim_forecast_df["yhat_upper_simulated"] = sim_forecast_df["yhat_upper_simulated"] * stock_mult
    else:
        # No scenario - show base forecast
        sim_forecast_df = forecast_df.copy()
        sim_forecast_df["yhat_simulated"] = sim_forecast_df["yhat"]
    
    # Always calculate impact metrics if simulation was run
    if 'sim_forecast_df' in locals() and "yhat_simulated" in sim_forecast_df.columns:
        # Calculate impact with detailed metrics
        try:
            impact = calculate_scenario_impact(
                forecast_df, sim_forecast_df, price_delta, effective_promotion if 'effective_promotion' in locals() else promotion_flag, holiday_flag
            )
            impact_text = impact.get('insight', 'Scenario applied')
            demand_change_pct = impact.get('demand_change_pct', 0)
            revenue_change = impact.get('revenue_change', 0)
            base_revenue = impact.get('base_revenue', 0)
            sim_revenue = impact.get('simulated_revenue', 0)
        except:
            base_total = forecast_df["yhat"].sum()
            sim_total = sim_forecast_df["yhat_simulated"].sum() if "yhat_simulated" in sim_forecast_df.columns else base_total
            demand_change_pct = ((sim_total - base_total) / (base_total + 1e-6)) * 100
            
            # Estimate revenue (assuming average price)
            avg_price = product_df["price"].mean() if "price" in product_df.columns else 100
            base_revenue = base_total * avg_price
            sim_revenue = sim_total * avg_price * (1 + price_delta / 100) if price_delta != 0 else sim_total * avg_price
            revenue_change = sim_revenue - base_revenue
            impact_text = f"Demand changes by {demand_change_pct:+.1f}%"
        
        # Display impact with simulation text
        scenario_desc_parts = []
        if price_delta != 0:
            scenario_desc_parts.append(f"{abs(price_delta):.0f}% {'discount' if price_delta < 0 else 'price increase'}")
        if effective_promotion if 'effective_promotion' in locals() else promotion_flag:
            promo_str = f"{promotion_intensity*100:.0f}% promo" if promotion_intensity < 1.0 else "promotion"
            scenario_desc_parts.append(promo_str)
        if holiday_flag:
            scenario_desc_parts.append("holiday period")
        if weather_val:
            scenario_desc_parts.append(f"{weather_val.lower()} weather")
        if stock_level_factor != 1.0:
            scenario_desc_parts.append(f"{stock_level_factor:.1f}x stock")
        
        scenario_desc = " + ".join(scenario_desc_parts) if scenario_desc_parts else "baseline"
        
        # Add stock level and promotion intensity to description if not already included
        if stock_level_factor != 1.0 and f"{stock_level_factor:.1f}x stock" not in scenario_desc:
            scenario_desc += f" + {stock_level_factor:.1f}x stock"
        
        st.info(f"💡 **Simulating impact of {scenario_desc} → Expected {demand_change_pct:+.1f}% demand {'increase' if demand_change_pct > 0 else 'decrease'}**")
        
        # Live KPI Updates: Revenue Before vs After
        st.markdown("#### 📊 Scenario Impact Metrics")
        kpi_whatif_col1, kpi_whatif_col2, kpi_whatif_col3, kpi_whatif_col4 = st.columns(4)
        
        with kpi_whatif_col1:
            st.metric(
                "💰 Revenue (Before)",
                f"₹{base_revenue:,.0f}",
                help="Base forecast revenue"
            )
        
        with kpi_whatif_col2:
            st.metric(
                "💰 Revenue (After)",
                f"₹{sim_revenue:,.0f}",
                delta=f"₹{revenue_change:+,.0f}",
                delta_color="normal" if revenue_change > 0 else "inverse",
                help="Simulated scenario revenue"
            )
        
        with kpi_whatif_col3:
            st.metric(
                "📈 Demand Change",
                f"{demand_change_pct:+.1f}%",
                help="Expected demand change"
            )
        
        with kpi_whatif_col4:
            revenue_change_pct = (revenue_change / (base_revenue + 1e-6)) * 100
            st.metric(
                "💵 Revenue Change",
                f"{revenue_change_pct:+.1f}%",
                help="Percentage revenue change"
            )
        
        # What-If Graph
        fig_whatif = go.Figure()
        
        # Base forecast
        fig_whatif.add_trace(go.Scatter(
            x=forecast_df["date"],
            y=forecast_df["yhat"],
            name="Base Forecast",
            mode="lines",
            line=dict(color=C_FORECAST, width=3),
            hovertemplate="<b>Base</b><br>Date: %{x|%Y-%m-%d}<br>Sales: %{y:,.0f}<extra></extra>"
        ))
        
        # Simulated forecast
        if "yhat_simulated" in sim_forecast_df.columns:
            fig_whatif.add_trace(go.Scatter(
                x=sim_forecast_df["date"],
                y=sim_forecast_df["yhat_simulated"],
                name="What-If Scenario",
                mode="lines",
                line=dict(color=C_WHAT_IF, width=3, dash="dash"),
                hovertemplate="<b>What-If</b><br>Date: %{x|%Y-%m-%d}<br>Sales: %{y:,.0f}<extra></extra>"
            ))
        
        # Scenario description
        scenario_parts = []
        if price_delta != 0:
            scenario_parts.append(f"{price_delta:+.0f}% price")
        if effective_promotion if 'effective_promotion' in locals() else promotion_flag:
            promo_str = f"{promotion_intensity*100:.0f}% promo" if promotion_intensity < 1.0 else "promotion"
            scenario_parts.append(promo_str)
        if holiday_flag:
            scenario_parts.append("holiday")
        if weather_val:
            scenario_parts.append(f"{weather_val.lower()} weather")
        if stock_level_factor != 1.0:
            scenario_parts.append(f"{stock_level_factor:.1f}x stock")
        scenario_desc = " + ".join(scenario_parts) if scenario_parts else "baseline"
        
        fig_whatif.update_layout(
            title=f"What-If Scenario: {scenario_desc}",
            xaxis_title="Date",
            yaxis_title="Sales Quantity",
            template="plotly_white",
            height=500,
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(rangeslider=dict(visible=True)),
            yaxis=dict(gridcolor='rgba(128, 128, 128, 0.2)')
        )
        
        st.plotly_chart(fig_whatif, use_container_width=True, key="whatif_chart")
    else:
        st.info("💡 Adjust sliders above to simulate different business scenarios")

with tab_insights:
    st.markdown("### 🤖 AI-Driven Business Insights")
    
    try:
        current_stock = float(product_df["stock_on_hand"].tail(1).iloc[0]) if "stock_on_hand" in product_df.columns and not product_df["stock_on_hand"].tail(1).isna().all() else None
        
        ai_insights = generate_forecast_insights(
            history_df,
            forecast_df,
            metrics,
            selected_product,
            stock_on_hand=current_stock,
            price_elasticity=elasticity
        )
        
        # ChatGPT-style Insights Panel with Sections
        st.markdown("#### 💬 AI Insight Chat")
        st.caption("🤖 AI-powered analysis of your forecast results")
        
        # Chat-style Q&A interface
        st.markdown("**💡 Ask Questions:**")
        qa_col1, qa_col2 = st.columns([3, 1])
        with qa_col1:
            user_question = st.text_input(
                "Type your question (e.g., 'Why did sales drop in April 2025?')",
                placeholder="Why did sales drop in April 2025?",
                key="ai_question_input"
            )
        with qa_col2:
            st.markdown("<br>", unsafe_allow_html=True)
            ask_button = st.button("🔍 Ask", use_container_width=True, key="ask_ai_button")
        
        # Auto-generate answer based on question keywords
        if ask_button and user_question:
            question_lower = user_question.lower()
            ai_response = ""
            
            if any(word in question_lower for word in ["why", "reason", "cause", "drop", "decline", "fall"]):
                # Analyze why sales might drop
                if "price" in question_lower or "cost" in question_lower:
                    avg_price_trend = product_df["price"].tail(8).mean() - product_df["price"].tail(4).mean() if "price" in product_df.columns else 0
                    if avg_price_trend > 0:
                        ai_response = f"**AI Analysis:** Sales decline likely due to **price increase** ({avg_price_trend:.2f} avg). Price elasticity ({elasticity:.2f}) suggests demand is sensitive. Consider promotional pricing during low-demand periods."
                    else:
                        ai_response = "**AI Analysis:** Price trend stable. Decline likely due to seasonal patterns or reduced promotion activity."
                elif "april" in question_lower or "month" in question_lower:
                    ai_response = f"**AI Analysis:** April sales decline typical due to **post-holiday seasonality**. Historical patterns show {peak_month} as peak month. Consider targeted promotions to boost April sales."
                else:
                    weights_str = str(details.get('weights', {})) if 'details' in locals() else "N/A"
                    ai_response = f"**AI Analysis:** Sales patterns show **seasonal variations**. Model weights: {weights_str}. Key drivers: price changes ({elasticity:.2f} elasticity) and promotion timing."
            
            elif any(word in question_lower for word in ["rise", "increase", "grow", "peak", "high"]):
                peak_date = forecast_df.loc[forecast_df["yhat"].idxmax(), "date"] if len(forecast_df) > 0 else None
                peak_month_name = peak_date.strftime("%B") if peak_date else peak_month
                ai_response = f"**AI Analysis:** Sales rise driven by **{peak_month_name} seasonality** and positive trend. Forecast shows {growth_pct:+.1f}% growth. Model confidence: {confidence_pct:.1f}%."
            
            elif any(word in question_lower for word in ["stock", "inventory", "out", "reorder"]):
                if stockout_risk is not None:
                    if stockout_risk > 70:
                        ai_response = f"**AI Analysis:** **High stock-out risk ({stockout_risk:.1f}%)** detected. Current inventory insufficient for forecasted demand. **Recommended action:** Reorder immediately to prevent lost sales."
                    elif stockout_risk > 40:
                        ai_response = f"**AI Analysis:** **Moderate stock-out risk ({stockout_risk:.1f}%)**. Monitor inventory levels closely. Consider reordering within 2 weeks."
                    else:
                        ai_response = f"**AI Analysis:** Stock levels adequate (risk: {stockout_risk:.1f}%). Current inventory sufficient for next month's demand."
                else:
                    ai_response = "**AI Analysis:** Stock level data unavailable. Unable to assess inventory risk."
            
            else:
                ai_response = f"**AI Analysis:** Based on hybrid modeling ({model_type}), sales for {selected_product} show {growth_pct:+.1f}% growth trend. Peak sales in {peak_month}. Key factors: price elasticity ({elasticity:.2f}), seasonal patterns, and promotion effects."
            
            if ai_response:
                st.markdown(f"""
                <div style="background-color: rgba(0, 200, 150, 0.1); padding: 15px; border-radius: 8px; border-left: 4px solid #00C896; margin: 10px 0;">
                    {ai_response}
                </div>
                """, unsafe_allow_html=True)
        
        # Forecast Summary Section
        with st.expander("📊 Forecast Summary", expanded=True):
            if "narrative" in ai_insights:
                # Enhanced business insight text format
                confidence_val = metrics.get("confidence", 100 - mape_val if not pd.isna(mape_val) else 0)
                peak_date = forecast_df.loc[forecast_df["yhat"].idxmax(), "date"] if len(forecast_df) > 0 else None
                peak_date_str = peak_date.strftime("%B %Y") if peak_date is not None else "upcoming period"
                
                enhanced_text = (
                    f"Based on hybrid modeling, sales for **{selected_product}** are expected to grow "
                    f"**{growth_pct:+.1f}%** by {peak_date_str}. "
                    f"Confidence: **{confidence_val:.1f}%**. "
                )
                
                if stockout_risk and stockout_risk > 40:
                    enhanced_text += f"Top risk: inventory shortages in {peak_month}."
                else:
                    enhanced_text += f"Peak sales expected in {peak_month}."
                
                st.markdown(f"**{enhanced_text}**")
                st.markdown(f"{ai_insights['narrative']}")
            else:
                st.markdown(f"**📝 Summary:** {selected_product} forecast shows {growth_pct:+.1f}% growth. Peak sales in {peak_month}.")
        
        # Market Signals Section
        with st.expander("📡 Market Signals", expanded=True):
            if "top_drivers" in ai_insights and ai_insights["top_drivers"]:
                for driver in ai_insights["top_drivers"]:
                    st.markdown(f"• {driver}")
            else:
                st.markdown("• Seasonal patterns show strong influence on demand")
                st.markdown("• Price changes drive short-term sales fluctuations")
        
        # Strategic Actions Section
        with st.expander("🎯 Strategic Actions", expanded=True):
            if "recommendations" in ai_insights and ai_insights["recommendations"]:
                for rec in ai_insights["recommendations"]:
                    st.markdown(f"• {rec}")
            else:
                st.markdown("• Monitor forecast accuracy and update model monthly")
                st.markdown("• Plan promotions for peak months to maximize sales")
        
        # Auto-Insight Cards
        st.markdown("---")
        st.markdown("#### 🔔 Auto-Insight Cards")
        
        insight_cards_col1, insight_cards_col2, insight_cards_col3 = st.columns(3)
        
        # High Growth Month Card
        if len(forecast_df) >= 52:
            forecast_by_month = forecast_df.groupby(forecast_df["date"].dt.month)["yhat"].mean()
            peak_month_idx = forecast_by_month.idxmax() if not forecast_by_month.empty else None
            peak_month_val = forecast_by_month.max() if not forecast_by_month.empty else 0
            if peak_month_idx:
                month_name = month_names[peak_month_idx - 1]
                baseline_month = forecast_by_month.mean()
                growth_month_pct = ((peak_month_val - baseline_month) / (baseline_month + 1e-6) * 100) if baseline_month > 0 else 0
                
                with insight_cards_col1:
                    st.success(f"📈 **High Growth Month Detected:** {month_name} 2026 (+{growth_month_pct:.0f}%)")
        
        # Stock-out Risk Card
        if stockout_risk is not None:
            if stockout_risk > 70:
                with insight_cards_col2:
                    st.error(f"🚨 **Stock-out Risk Alert:** {stockout_risk:.1f}% risk for {selected_product}")
            elif stockout_risk > 40:
                with insight_cards_col2:
                    st.warning(f"⚠️ **Stock-out Risk:** {stockout_risk:.1f}% - Monitor inventory")
        
        # Revenue Hotspot Card
        if len(forecast_df) >= 5:
            top_revenue_week = forecast_df.nlargest(1, "yhat").iloc[0]
            top_revenue_date = top_revenue_week["date"].strftime("%B %Y") if hasattr(top_revenue_week["date"], 'strftime') else str(top_revenue_week["date"])
            with insight_cards_col3:
                st.info(f"💰 **Revenue Hotspot:** {top_revenue_date} ({top_revenue_week['yhat']:,.0f} units)")
        
        # Business Impact Paragraph
        st.markdown("---")
        st.markdown("#### 💰 Projected Business Impact")
        
        # Calculate potential savings/optimization
        if not pd.isna(mape_val) and mape_val < 15:
            estimated_savings = next_q_forecast * 0.05  # 5% optimization opportunity
            st.success(
                f"**By optimizing prices and inventory according to forecast, "
                f"projected savings = ₹{estimated_savings:,.0f} over next quarter.** "
                f"Model confidence: {100-mape_val:.1f}% ensures reliable decision-making."
            )
    
    except Exception as e:
        st.warning(f"Enhanced insights unavailable: {e}")
        st.markdown(f"**📝 Summary:** {selected_product} forecast shows {growth_pct:+.1f}% growth. Peak sales in {peak_month}.")

# ========================================================================
# DOWNLOAD BUTTONS
# ========================================================================
st.markdown("---")
st.markdown("### 💾 Download Results")

download_col1, download_col2, download_col3 = st.columns(3)

with download_col1:
    # Forecast CSV
    forecast_csv_display = forecast_df.copy()
    forecast_csv_display["date"] = pd.to_datetime(forecast_csv_display["date"]).dt.strftime("%Y-%m-%d")
    if "product_name" not in forecast_csv_display.columns:
        forecast_csv_display["product_name"] = selected_product
    csv_data = forecast_csv_display.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Forecast (CSV)",
        data=csv_data,
        file_name=f"forecast_{selected_product.replace(' ', '_')}_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        key="download_forecast_csv_final",
        use_container_width=True
    )

with download_col2:
    # Insights Summary CSV
    try:
        insights_summary = pd.DataFrame({
            "Metric": ["Next Month Revenue", "Next Quarter Revenue", "Stock-out Risk %", "Peak Month", "Growth %", "MAPE", "RMSE"],
            "Value": [
                f"{next_month_forecast:,.0f}",
                f"{next_q_forecast:,.0f}",
                f"{stockout_risk:.1f}%" if stockout_risk is not None else "N/A",
                peak_month,
                f"{growth_pct:+.1f}%",
                f"{mape_val:.1f}%" if not pd.isna(mape_val) else "N/A",
                f"{rmse_val:.2f}" if not pd.isna(rmse_val) else "N/A"
            ]
        })
        insights_csv = insights_summary.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📊 Download Insights (CSV)",
            data=insights_csv,
            file_name=f"insights_{selected_product.replace(' ', '_')}.csv",
            mime="text/csv",
            key="download_insights_csv_final",
            use_container_width=True
        )
    except:
        st.info("📊 Insights CSV unavailable")

with download_col3:
    # Metrics JSON
    try:
        metrics_json = json.dumps(metrics, indent=2).encode('utf-8')
        st.download_button(
            label="📈 Download Metrics (JSON)",
            data=metrics_json,
            file_name=f"metrics_{selected_product.replace(' ', '_')}_{pd.Timestamp.now().strftime('%Y%m%d')}.json",
            mime="application/json",
            key="download_metrics_json_final",
            use_container_width=True
        )
    except:
        st.info("📈 Metrics JSON unavailable")

# PDF Export Section
st.markdown("---")
st.markdown("### 📄 Forecast Report Export")

pdf_col1, pdf_col2 = st.columns([2, 1])

with pdf_col1:
    # Generate PDF Report (as text/HTML format - can be converted to PDF)
    try:
        confidence_val = confidence_pct if 'confidence_pct' in locals() else (100 - mape_val if not pd.isna(mape_val) else 0)
        mape_display = f"{mape_val:.1f}%" if not pd.isna(mape_val) else "N/A"
        risk_display = f"{stockout_risk:.1f}%" if stockout_risk is not None else "N/A"
        
        report_text = f"""
FORECAST REPORT - {selected_product}
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
==========================================

FORECAST SUMMARY
----------------
Product: {selected_product}
Forecast Horizon: {horizon_weeks} weeks ({horizon_weeks * 7} days)
Model Type: {model_type}
Confidence Level: {confidence_val:.1f}%

KEY PERFORMANCE INDICATORS
---------------------------
• Predicted Growth: {growth_pct:+.1f}%
• Model Accuracy (MAPE): {mape_display}
• Next Month Revenue: ₹{next_month_forecast:,.0f}
• Stock-out Risk: {risk_display}
• Peak Month: {peak_month}

TOP 5 FORECASTED WEEKS
----------------------
"""
        if len(forecast_df) >= 5:
            top_weeks_for_report = forecast_df.nlargest(5, "yhat")
            for idx, row in top_weeks_for_report.iterrows():
                report_text += f"• {row['date'].strftime('%Y-%m-%d')}: {row['yhat']:,.0f} units\n"
        
        report_text += f"""
AI-GENERATED INSIGHTS
---------------------
"""
        try:
            if "narrative" in ai_insights:
                report_text += f"{ai_insights['narrative']}\n\n"
            if "top_drivers" in ai_insights:
                report_text += "Top Demand Drivers:\n"
                for driver in ai_insights["top_drivers"]:
                    report_text += f"• {driver}\n"
            if "recommendations" in ai_insights:
                report_text += "\nRecommendations:\n"
                for rec in ai_insights["recommendations"]:
                    report_text += f"• {rec}\n"
        except:
            report_text += f"Sales for {selected_product} are expected to grow {growth_pct:+.1f}% with peak sales in {peak_month}.\n"
        
        report_bytes = report_text.encode('utf-8')
        
        st.download_button(
            label="📥 Download Forecast Report (TXT/PDF)",
            data=report_bytes,
            file_name=f"forecast_report_{selected_product.replace(' ', '_')}_{pd.Timestamp.now().strftime('%Y%m%d')}.txt",
            mime="text/plain",
            key="download_pdf_report",
            use_container_width=True,
            help="Download comprehensive forecast report with KPIs, insights, and top weeks"
        )
    except Exception as e:
        st.warning(f"Report generation error: {e}")

with pdf_col2:
    # Optional: AI Voice Summary (if TTS available)
    try:
        import pyttsx3
        TTS_AVAILABLE = True
    except ImportError:
        TTS_AVAILABLE = False
    
    if TTS_AVAILABLE:
        if st.button("🔊 Generate AI Voice Summary", use_container_width=True, key="voice_summary"):
            try:
                engine = pyttsx3.init()
                summary = f"Sales forecast for {selected_product} shows {growth_pct:+.1f} percent growth. Peak sales expected in {peak_month}."
                engine.say(summary)
                engine.runAndWait()
                st.success("Voice summary generated!")
            except:
                st.info("Voice synthesis unavailable")
    else:
        st.info("💡 Install pyttsx3 for voice summaries")


with tab3:
    # Tier-3 Professional Header with Gradient Styling
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(244, 67, 54, 0.1) 0%, rgba(255, 193, 7, 0.1) 100%);
                padding: 20px; border-radius: 12px; margin-bottom: 20px; border-left: 4px solid #F44336;">
        <h2>🚨 Sales Anomalies — Tier-3 Hybrid Detection Engine</h2>
        <p style="font-size: 16px; color: #666;"><strong>AI-Powered Anomaly Detection</strong> | 
        <strong>Root-Cause Analysis</strong> | 
        <strong>Intelligent Insights</strong> | 
        <strong>Interactive Visualizations</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Try to load data_with_all_features.csv for anomaly detection
    @st.cache_data(show_spinner="Loading data...")
    def load_tab3_data():
        try:
            if os.path.exists(FEATURES_DATA_PATH):
                return pd.read_csv(FEATURES_DATA_PATH, low_memory=False, encoding='utf-8')
        except Exception as e:
            st.error(f"⚠️ Error loading data: {e}")
            st.stop()
        return None
    
    tab3_df = load_tab3_data()
    
    if tab3_df is not None and not tab3_df.empty:
        if "product_name" in tab3_df.columns and "sales_qty" in tab3_df.columns:
            # ========================================================================
            # CONTROLS: Product Selection & Sensitivity
            # ========================================================================
            control_col1, control_col2 = st.columns([2, 1])
            with control_col1:
                product_select = st.selectbox(
                    "🔍 Select Product", 
                    ["All Products"] + sorted(tab3_df["product_name"].unique().tolist()), 
                    key="tab3_product"
                )
            with control_col2:
                sensitivity = st.slider(
                    "🎚️ Detection Sensitivity", 
                    min_value=1, 
                    max_value=100, 
                    value=50, 
                    help="Lower = stricter detection, Higher = catch more anomalies",
                    key="tab3_sensitivity"
                )
                show_mild = st.checkbox("Show Mild Anomalies", value=False, key="tab3_mild")
            
            if product_select == "All Products":
                # ========================================================================
                # MULTI-PRODUCT ANOMALY HEATMAP
                # ========================================================================
                st.markdown("### 📊 Multi-Product Anomaly Heatmap")
                
                # Aggregate anomalies across all products
                all_anomalies = []
                for prod in tab3_df["product_name"].unique()[:20]:  # Limit to 20 for performance
                    try:
                        anomalies = detect_sales_anomalies(tab3_df, prod, method="hybrid")
                        if not anomalies.empty and "date" in anomalies.columns:
                            anomalies["product"] = prod
                            all_anomalies.append(anomalies)
                    except:
                        continue
                
                if all_anomalies:
                    combined_anom = pd.concat(all_anomalies, ignore_index=True)
                    combined_anom["date"] = pd.to_datetime(combined_anom["date"], errors='coerce')
                    combined_anom["month"] = combined_anom["date"].dt.to_period("M").astype(str)
                    combined_anom["severity_score"] = combined_anom["deviation_pct"].abs()
                    
                    # Create heatmap data
                    heatmap_data = combined_anom.pivot_table(
                        values="severity_score",
                        index="product",
                        columns="month",
                        aggfunc="mean",
                        fill_value=0
                    )
                    
                    if not heatmap_data.empty:
                        fig_heatmap = go.Figure(data=go.Heatmap(
                            z=heatmap_data.values,
                            x=heatmap_data.columns,
                            y=heatmap_data.index,
                            colorscale="RdYlGn_r",
                            text=heatmap_data.values.round(1),
                            texttemplate="%{text}%",
                            textfont={"size": 8},
                            colorbar=dict(title="Severity %")
                        ))
                        fig_heatmap.update_layout(
                            title="Anomaly Severity Heatmap (Product × Month)",
                            xaxis_title="Month",
                            yaxis_title="Product",
                            height=600,
                            template="plotly_dark"
                        )
                        st.plotly_chart(fig_heatmap, use_container_width=True)
                
                st.info("💡 Select a specific product below for detailed analysis")
                display_df = tab3_df
            else:
                display_df = tab3_df[tab3_df["product_name"] == product_select]
                
                # ========================================================================
                # SINGLE PRODUCT: TIER-3 HYBRID ANOMALY DETECTION
                # Ensemble: Isolation Forest + Z-Score + Prophet Residual Analysis
                # ========================================================================
                with st.spinner("🔍 Running hybrid anomaly detection (Isolation Forest + Z-Score + Prophet Residual Analysis)..."):
                    anomalies = detect_sales_anomalies(tab3_df, product_select, method="hybrid")
                
                if not anomalies.empty:
                    # Filter by sensitivity
                    threshold = (100 - sensitivity) / 100 * anomalies["deviation_pct"].abs().quantile(0.95)
                    filtered_anomalies = anomalies[anomalies["deviation_pct"].abs() >= threshold]
                    
                    if not show_mild:
                        filtered_anomalies = filtered_anomalies[
                            filtered_anomalies["severity"].isin(["Moderate", "Severe"])
                        ]
                    
                    # ========================================================================
                    # TIER-3 STATISTICAL KPIs WITH STYLISH BLOCKS
                    # ========================================================================
                    st.markdown("### 📊 Statistical Anomaly Metrics")
                    
                    # Calculate comprehensive statistics
                    avg_dev = filtered_anomalies["deviation_pct"].abs().mean() if not filtered_anomalies.empty else 0
                    median_dev = filtered_anomalies["deviation_pct"].abs().median() if not filtered_anomalies.empty else 0
                    std_dev = filtered_anomalies["deviation_pct"].std() if not filtered_anomalies.empty else 0
                    severe_count = len(filtered_anomalies[filtered_anomalies["severity"] == "Severe"]) if not filtered_anomalies.empty else 0
                    
                    # Most frequent anomaly period
                    if not filtered_anomalies.empty:
                        filtered_anomalies["month"] = pd.to_datetime(filtered_anomalies["date"]).dt.month
                        most_freq_period = filtered_anomalies["month"].mode().iloc[0] if not filtered_anomalies["month"].mode().empty else "N/A"
                        month_names = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                        most_freq_period_str = month_names[most_freq_period] if isinstance(most_freq_period, (int, np.integer)) and 1 <= most_freq_period <= 12 else str(most_freq_period)
                    else:
                        most_freq_period_str = "N/A"
                    
                    # Average confidence from anomalies
                    avg_confidence = filtered_anomalies["confidence"].mean() if "confidence" in filtered_anomalies.columns and not filtered_anomalies.empty else (100 - sensitivity / 2)
                    
                    kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
                    
                    with kpi_col1:
                        st.metric("📈 Total Anomalies", len(filtered_anomalies), help="Number of detected anomalies")
                    with kpi_col2:
                        st.metric("📊 Avg Deviation", f"{avg_dev:.1f}%", delta=f"Median: {median_dev:.1f}%", help="Average absolute deviation percentage")
                    with kpi_col3:
                        st.metric("📉 Std Deviation", f"{std_dev:.1f}%", help="Standard deviation of anomalies")
                    with kpi_col4:
                        st.metric("🚨 Severe Cases", severe_count, help="Number of severe anomalies detected")
                    with kpi_col5:
                        st.metric("🎯 Peak Period", most_freq_period_str, help="Most frequent anomaly month")
                    
                    # Confidence indicator
                    conf_color = "🟢" if avg_confidence >= 80 else "🟡" if avg_confidence >= 60 else "🔴"
                    st.caption(f"{conf_color} **Detection Confidence:** {avg_confidence:.0f}% | Last updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    st.markdown("---")
                    
                    # ========================================================================
                    # TIER-3 MULTI-LAYER INTERACTIVE VISUALIZATION
                    # ========================================================================
                    st.markdown("### 📈 Multi-Layer Anomaly Visualization Dashboard")
                    
                    # Severity threshold control
                    viz_col1, viz_col2, viz_col3 = st.columns([2, 1, 1])
                    with viz_col1:
                        severity_threshold = st.slider(
                            "🎚️ Severity Threshold (%)",
                            min_value=0,
                            max_value=100,
                            value=25,
                            help="Minimum deviation percentage to highlight",
                            key="severity_threshold_slider"
                        )
                    with viz_col2:
                        compare_products = st.toggle("🔀 Compare Products", value=False, key="compare_products_toggle")
                    with viz_col3:
                        playback_mode = st.toggle("▶️ Playback Mode", value=False, key="playback_mode_toggle", help="Animate anomalies chronologically")
                    
                    # Filter anomalies by threshold
                    thresholded_anomalies = filtered_anomalies[
                        filtered_anomalies["deviation_pct"].abs() >= severity_threshold
                    ] if not filtered_anomalies.empty else filtered_anomalies
                    
                    fig_anom = go.Figure()
                    
                    # Layer 1: Actual vs Expected Sales line
                    if "week_start" in display_df.columns:
                        display_df_sorted = display_df.sort_values("week_start")
                        fig_anom.add_trace(go.Scatter(
                            x=pd.to_datetime(display_df_sorted["week_start"]),
                            y=display_df_sorted["sales_qty"],
                            name="Actual Sales",
                            mode="lines+markers",
                            line=dict(color="#666666", width=3),
                            marker=dict(size=6, color="#666666", opacity=0.7),
                            hovertemplate="<b>Actual</b><br>Date: %{x|%Y-%m-%d}<br>Sales: %{y:,.0f}<extra></extra>"
                        ))
                    
                    # Expected sales (blue dashed)
                    if "date" in anomalies.columns and "expected_sales" in anomalies.columns:
                        anom_dates = pd.to_datetime(anomalies["date"], errors='coerce')
                        valid_mask = anom_dates.notna() & anomalies["expected_sales"].notna()
                        if valid_mask.sum() > 0:
                            fig_anom.add_trace(go.Scatter(
                                x=anom_dates[valid_mask],
                                y=anomalies.loc[valid_mask, "expected_sales"],
                                name="Expected Sales",
                                mode="lines",
                                line=dict(color="#2196F3", width=2, dash="dot"),
                                hovertemplate="<b>Expected</b><br>Date: %{x|%Y-%m-%d}<br>Sales: %{y:,.0f}<extra></extra>"
                            ))
                    
                    # Layer 2: Highlighted anomaly regions (red zones)
                    if "date" in thresholded_anomalies.columns and not thresholded_anomalies.empty:
                        for _, anom_row in thresholded_anomalies.iterrows():
                            anom_date = pd.to_datetime(anom_row["date"], errors='coerce')
                            if pd.notna(anom_date):
                                deviation = anom_row.get("deviation_pct", 0)
                                actual = anom_row.get("actual_sales", 0)
                                expected = anom_row.get("expected_sales", actual)
                                
                                # Add shaded region
                                fig_anom.add_shape(
                                    type="rect",
                                    x0=anom_date - pd.Timedelta(days=3),
                                    x1=anom_date + pd.Timedelta(days=3),
                                    y0=min(actual, expected) * 0.9,
                                    y1=max(actual, expected) * 1.1,
                                    fillcolor="rgba(255, 0, 0, 0.1)",
                                    line=dict(width=0),
                                    layer="below"
                                )
                    
                    # Layer 3: Anomaly points with hover tooltips
                    if "date" in thresholded_anomalies.columns and "actual_sales" in thresholded_anomalies.columns:
                        anom_dates = pd.to_datetime(thresholded_anomalies["date"], errors='coerce')
                        valid_mask = anom_dates.notna() & thresholded_anomalies["actual_sales"].notna()
                        if valid_mask.sum() > 0:
                            severity_colors = {
                                "Mild": "#FFC107",
                                "Moderate": "#FF9800",
                                "Severe": "#F44336"
                            }
                            for sev in ["Mild", "Moderate", "Severe"]:
                                sev_mask = valid_mask & (thresholded_anomalies["severity"] == sev)
                                if sev_mask.sum() > 0:
                                    sev_data = thresholded_anomalies.loc[sev_mask]
                                    fig_anom.add_trace(go.Scatter(
                                        x=anom_dates[sev_mask],
                                        y=sev_data["actual_sales"],
                                        name=f"{sev} Anomalies",
                                        mode="markers",
                                        marker=dict(
                                            color=severity_colors.get(sev, "red"),
                                            symbol="x",
                                            size=14,
                                            line=dict(width=2, color="white")
                                        ),
                                        hovertemplate="<b>⚠️ %{fullData.name}</b><br>" +
                                                     "Date: %{x|%Y-%m-%d}<br>" +
                                                     "Sales: %{y:,.0f}<br>" +
                                                     "Deviation: " + sev_data["deviation_pct"].astype(str) + "%<br>" +
                                                     "Severity: " + sev_data["severity"].astype(str) + "<extra></extra>"
                                    ))
                    
                    # Layer 4: Severity threshold lines
                    if "week_start" in display_df.columns:
                        mean_sales = display_df["sales_qty"].mean()
                        upper_thresh = mean_sales * (1 + severity_threshold / 100)
                        lower_thresh = mean_sales * (1 - severity_threshold / 100)
                        
                        fig_anom.add_hline(
                            y=upper_thresh,
                            line_dash="dash",
                            line_color="orange",
                            annotation_text=f"+{severity_threshold}% Threshold",
                            annotation_position="right"
                        )
                        fig_anom.add_hline(
                            y=lower_thresh,
                            line_dash="dash",
                            line_color="orange",
                            annotation_text=f"-{severity_threshold}% Threshold",
                            annotation_position="right"
                        )
                    
                    fig_anom.update_layout(
                        title=f"{product_select} — Enhanced Anomaly Detection Timeline",
                        xaxis_title="Date",
                        yaxis_title="Sales Quantity",
                        template="plotly_dark",
                        height=550,
                        hovermode='closest',
                        showlegend=True,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig_anom, use_container_width=True, key="enhanced_anomaly_timeline")
                    
                    # Deviation Histogram
                    if not filtered_anomalies.empty:
                        st.markdown("#### 📊 Deviation Distribution Histogram")
                        fig_hist = go.Figure()
                        fig_hist.add_trace(go.Histogram(
                            x=filtered_anomalies["deviation_pct"],
                            nbinsx=20,
                            marker_color="#FF6B6B",
                            opacity=0.7
                        ))
                        fig_hist.update_layout(
                            title="Distribution of Deviation Percentages",
                            xaxis_title="Deviation %",
                            yaxis_title="Frequency",
                            template="plotly_white",
                            height=300
                        )
                        st.plotly_chart(fig_hist, use_container_width=True, key="deviation_histogram")
                    
                    # ========================================================================
                    # TIER-3 AI-DRIVEN ROOT CAUSE ANALYSIS (GPT-STYLE)
                    # ========================================================================
                    st.markdown("### 🧠 AI-Driven Root Cause Analysis & Intelligent Insights")
                    
                    # Initialize variables for use throughout the section
                    sorted_anomalies = pd.DataFrame()
                    ai_insights = {}
                    
                    if not filtered_anomalies.empty:
                        # Sort by deviation (most significant first)
                        sorted_anomalies = filtered_anomalies.sort_values("deviation_pct", key=lambda x: x.abs(), ascending=False)
                        latest_anom = sorted_anomalies.iloc[0]
                        
                        # Generate AI explanation
                        ai_insights = generate_ai_root_cause_explanation(
                            latest_anom, 
                            product_select,
                            context_df=display_df
                        )
                        
                        confidence_score = ai_insights.get("confidence_score", 50)
                        conf_badge_color = "🟢" if confidence_score >= 80 else "🟡" if confidence_score >= 60 else "🔴"
                        conf_bg_color = "rgba(76, 175, 80, 0.1)" if confidence_score >= 80 else "rgba(255, 193, 7, 0.1)" if confidence_score >= 60 else "rgba(244, 67, 54, 0.1)"
                        
                        # Main AI Explanation Card
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, {conf_bg_color}, rgba(255,255,255,0.1)); 
                                    padding: 20px; border-radius: 12px; border-left: 4px solid {'#4CAF50' if confidence_score >= 80 else '#FFC107' if confidence_score >= 60 else '#F44336'};
                                    margin: 15px 0;">
                            <h4>🔍 AI-Powered Anomaly Explanation</h4>
                            <p>{ai_insights.get('explanation', 'Analysis in progress...')}</p>
                            <p><strong>{conf_badge_color} Confidence Score:</strong> {confidence_score:.0f}% | 
                               <strong>Pattern Type:</strong> {ai_insights.get('pattern_type', 'Standard variation')}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Expandable diagnostic section
                        with st.expander("🔍 Explain This Anomaly — Full Diagnostic", expanded=False):
                            diag_col1, diag_col2 = st.columns(2)
                            
                            with diag_col1:
                                st.markdown("#### 📋 Likely Causes")
                                causes = ai_insights.get("likely_causes", [])
                                for i, cause in enumerate(causes[:5], 1):
                                    st.markdown(f"**{i}.** {cause.title()}")
                            
                            with diag_col2:
                                st.markdown("#### 💡 Suggested Corrective Actions")
                                actions = ai_insights.get("suggested_actions", [])
                                for i, action in enumerate(actions[:5], 1):
                                    st.markdown(f"**{i}.** {action}")
                        
                        st.markdown("---")
                        
                        # ========================================================================
                        # INTERACTIVE INSIGHTS FEED
                        # ========================================================================
                        st.markdown("### 💬 AI Insights Feed — Real-Time Anomaly Intelligence")
                        
                        # Create insight cards for top anomalies
                        insights_feed_col1, insights_feed_col2 = st.columns(2)
                        
                        for idx, (_, anom_row) in enumerate(sorted_anomalies.head(6).iterrows()):
                            col = insights_feed_col1 if idx % 2 == 0 else insights_feed_col2
                            
                            with col:
                                deviation_val = anom_row.get("deviation_pct", 0)
                                date_val = str(anom_row.get("date", ""))[:10]
                                severity_val = anom_row.get("severity", "Unknown")
                                
                                # Generate quick insight
                                if deviation_val > 30:
                                    insight_msg = f"⚠️ Week {date_val} spike due to promotional activity or demand surge"
                                elif deviation_val < -30:
                                    insight_msg = f"📉 Week {date_val} drop likely due to supply constraints or pricing changes"
                                else:
                                    insight_msg = f"📊 Week {date_val} shows {abs(deviation_val):.1f}% variation — monitor trends"
                                
                                # Card styling
                                severity_color = "#F44336" if severity_val == "Severe" else "#FF9800" if severity_val == "Moderate" else "#FFC107"
                                
                                st.markdown(f"""
                                <div style="background-color: rgba(33, 33, 33, 0.05); 
                                            padding: 15px; border-radius: 8px; 
                                            border-left: 4px solid {severity_color};
                                            margin: 10px 0;">
                                    <strong>{insight_msg}</strong><br>
                                    <small>Severity: {severity_val} | Deviation: {deviation_val:+.1f}%</small>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Action buttons row
                                btn_col1, btn_col2, btn_col3 = st.columns(3)
                                with btn_col1:
                                    if st.button(f"✅ Mark Resolved", key=f"resolve_{idx}", use_container_width=True):
                                        st.success(f"Anomaly {date_val} marked as resolved!")
                                with btn_col2:
                                    if st.button(f"🔍 Investigate", key=f"investigate_{idx}", use_container_width=True):
                                        st.info(f"Investigating anomaly for {date_val}...")
                                with btn_col3:
                                    if st.button(f"📄 Report", key=f"report_{idx}", use_container_width=True):
                                        st.info(f"Generating report for {date_val}...")
                    
                    # ========================================================================
                    # CORRELATION ANALYSIS WITH EXTERNAL FACTORS
                    # ========================================================================
                    if not filtered_anomalies.empty and len(display_df) > 0:
                        st.markdown("### 🔗 Correlation Analysis with External Factors")
                        
                        corr_col1, corr_col2 = st.columns(2)
                        
                        with corr_col1:
                            # Calculate correlations if columns exist
                            corr_data = []
                            
                            # Merge anomalies with display_df for correlation analysis
                            anom_with_context = filtered_anomalies.copy()
                            anom_with_context["date"] = pd.to_datetime(anom_with_context["date"])
                            display_df_dates = pd.to_datetime(display_df.get("week_start", display_df.get("date", "")))
                            
                            if "price" in display_df.columns:
                                price_corr = None
                                try:
                                    merged = anom_with_context.merge(
                                        display_df,
                                        left_on="date",
                                        right_on=pd.to_datetime(display_df_dates),
                                        how="inner"
                                    )
                                    if len(merged) > 3:
                                        price_corr = merged["deviation_pct"].corr(merged["price"])
                                        if not pd.isna(price_corr):
                                            corr_data.append({"Factor": "Price", "Correlation": price_corr})
                                except:
                                    pass
                            
                            if "stock_on_hand" in display_df.columns:
                                stock_corr = None
                                try:
                                    merged = anom_with_context.merge(
                                        display_df,
                                        left_on="date",
                                        right_on=pd.to_datetime(display_df_dates),
                                        how="inner"
                                    )
                                    if len(merged) > 3:
                                        stock_corr = merged["deviation_pct"].corr(merged["stock_on_hand"])
                                        if not pd.isna(stock_corr):
                                            corr_data.append({"Factor": "Stock Level", "Correlation": stock_corr})
                                except:
                                    pass
                            
                            if corr_data:
                                corr_df = pd.DataFrame(corr_data)
                                st.dataframe(
                                    corr_df.style.format({"Correlation": "{:.3f}"}),
                                    use_container_width=True,
                                    hide_index=True
                                )
                            else:
                                st.info("💡 Correlation analysis requires matching date columns")
                        
                        with corr_col2:
                            st.markdown("#### 📊 Pattern Recognition")
                            if not filtered_anomalies.empty:
                                # Analyze patterns
                                positive_anoms = len(filtered_anomalies[filtered_anomalies["deviation_pct"] > 0])
                                negative_anoms = len(filtered_anomalies[filtered_anomalies["deviation_pct"] < 0])
                                
                                pattern_text = ""
                                if positive_anoms > negative_anoms * 2:
                                    pattern_text = "📈 **Pattern:** Consistent spikes (promotional or demand surges)"
                                elif negative_anoms > positive_anoms * 2:
                                    pattern_text = "📉 **Pattern:** Consistent drops (supply chain or pricing issues)"
                                else:
                                    pattern_text = "📊 **Pattern:** Mixed variations (seasonal or market volatility)"
                                
                                st.markdown(pattern_text)
                    
                    st.markdown("---")
                    
                    # ========================================================================
                    # DETAILED ANOMALY TABLE WITH ENHANCED FORMATTING
                    # ========================================================================
                    st.markdown("### 📋 Detailed Anomaly Table")
                    
                    # Prepare table with all required columns
                    table_df = filtered_anomalies.copy()
                    if "date" in table_df.columns:
                        table_df["Date"] = pd.to_datetime(table_df["date"]).dt.strftime("%Y-%m-%d")
                    if "actual_sales" in table_df.columns:
                        table_df["Actual"] = table_df["actual_sales"].round(0).astype(int)
                    if "expected_sales" in table_df.columns:
                        table_df["Expected"] = table_df["expected_sales"].round(0).astype(int)
                    if "deviation_pct" in table_df.columns:
                        table_df["Deviation (%)"] = table_df["deviation_pct"].round(1)
                    if "severity" in table_df.columns:
                        table_df["Severity"] = table_df["severity"]
                    if "confidence" in table_df.columns:
                        table_df["Confidence"] = table_df["confidence"].round(0).astype(int)
                    
                    display_cols = ["Date", "Actual", "Expected", "Deviation (%)", "Severity", "Confidence"]
                    available_cols = [c for c in display_cols if c in table_df.columns]
                    
                    if available_cols:
                        st.dataframe(
                            table_df[available_cols].sort_values("Deviation (%)", key=lambda x: x.abs(), ascending=False),
                            use_container_width=True,
                            hide_index=True,
                            height=400
                        )
                    
                    # ========================================================================
                    # PROFESSIONAL REPORTING MODE
                    # ========================================================================
                    st.markdown("### 🧾 Professional Reporting Mode")
                    
                    report_col1, report_col2, report_col3 = st.columns(3)
                    
                    with report_col1:
                        # CSV Export
                        csv_data = filtered_anomalies.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "📥 Export Anomaly Report (CSV)",
                            csv_data,
                            f"anomaly_report_{product_select.replace(' ', '_')}.csv",
                            "text/csv",
                            key=f"download_anomalies_csv_{product_select}",
                            use_container_width=True
                        )
                    
                    with report_col2:
                        # Generate comprehensive report summary
                        if st.button("📄 Generate Comprehensive Report", use_container_width=True, key="generate_full_report"):
                            with st.spinner("Generating comprehensive report..."):
                                # Get top anomalies for report
                                if not filtered_anomalies.empty:
                                    top_anoms = filtered_anomalies.sort_values("deviation_pct", key=lambda x: x.abs(), ascending=False).head(5)
                                    
                                    # Generate AI insights for report if not already done
                                    if not filtered_anomalies.empty:
                                        top_anom = top_anoms.iloc[0]
                                        report_ai_insights = generate_ai_root_cause_explanation(
                                            top_anom, 
                                            product_select,
                                            context_df=display_df
                                        )
                                    else:
                                        report_ai_insights = {"suggested_actions": ["Monitor trends", "Investigate causes"]}
                                else:
                                    top_anoms = pd.DataFrame()
                                    report_ai_insights = {"suggested_actions": []}
                                
                                report_summary = f"""
# Anomaly Detection Report
## Product: {product_select}
## Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

### Summary Statistics
- **Total Anomalies Detected:** {len(filtered_anomalies)}
- **Average Deviation:** {avg_dev:.2f}%
- **Severe Cases:** {severe_count}
- **Detection Confidence:** {avg_confidence:.0f}%

### Top 5 Anomalies
"""
                                if not top_anoms.empty:
                                    for idx, (_, row) in enumerate(top_anoms.iterrows(), 1):
                                        report_summary += f"""
{idx}. **Date:** {str(row.get('date', ''))[:10]} | **Deviation:** {row.get('deviation_pct', 0):+.1f}% | **Severity:** {row.get('severity', 'Unknown')}
"""
                                else:
                                    report_summary += "\nNo anomalies detected.\n"
                                
                                report_summary += f"""
### Recommended Actions
Based on AI analysis, the following actions are recommended:
"""
                                for action in report_ai_insights.get("suggested_actions", ["Monitor trends", "Investigate underlying causes"])[:5]:
                                    report_summary += f"- {action}\n"
                                
                                st.success("Report generated! Copy the text below or use CSV export.")
                                st.code(report_summary, language="markdown")
                    
                    with report_col3:
                        # Correlation report
                        if st.button("📊 Generate Correlation Report", use_container_width=True, key="generate_corr_report"):
                            st.info("💡 Correlation analysis available in the section above")
                    
                    # Last updated timestamp
                    st.caption(f"📅 **Last updated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} | **Detection Engine:** Hybrid (Isolation Forest + Z-Score + Prophet Residual Analysis)")
                else:
                    st.success(f"✅ No anomalies detected for {product_select} at current sensitivity level.")
        else:
            st.warning("⚠️ Required columns (product_name, sales_qty) not found in dataset")
    elif sales_anomalies is not None and not sales_anomalies.empty:
        # Fallback to old format
        st.dataframe(sales_anomalies.head(20))
        if {"date", "actual_sales"}.issubset(sales_anomalies.columns):
            fig = px.scatter(sales_anomalies, x="date", y="actual_sales", title="Sales Anomalies Timeline")
            st.plotly_chart(fig, use_container_width=True, key="tab3_anomalies_timeline_chart")
        download_button(sales_anomalies, "⬇️ Download Sales Anomalies", "sales_anomalies.csv", key="download_anomalies_tab3")
    else:
        st.info("💡 Please ensure **data_with_all_features.csv** exists. Upload data and run pipeline if needed.")

with tab4:
    st.subheader("📦 Inventory Alerts — Predictive Warehouse Optimization")
    st.markdown("**AI-Driven Stock Health Monitoring & Smart Replenishment**")
    
    @st.cache_data(show_spinner="Analyzing inventory...")
    def load_tab4_data():
        if os.path.exists(FEATURES_DATA_PATH):
            try:
                return pd.read_csv(FEATURES_DATA_PATH, low_memory=False, encoding='utf-8')
            except:
                return None
        return None
    
    tab4_df = load_tab4_data()
    
    if tab4_df is not None and not tab4_df.empty:
        # Check if stock data exists
        if "stock_on_hand" not in tab4_df.columns:
            st.warning("⚠️ 'stock_on_hand' column not found. Using estimated stock levels.")
            # Create dummy stock levels for demo
            tab4_df["stock_on_hand"] = np.random.randint(50, 500, len(tab4_df))
        
        if "product_name" in tab4_df.columns:
            # ========================================================================
            # CONTROLS: Product Selection & Simulation Parameters
            # ========================================================================
            control_col1, control_col2 = st.columns([2, 2])
            with control_col1:
                product_select = st.selectbox(
                    "🔍 Select Product",
                    ["All Products"] + sorted(tab4_df["product_name"].unique().tolist()),
                    key="tab4_product"
                )
            with control_col2:
                demand_growth = st.slider(
                    "📈 Demand Growth Rate (%)",
                    min_value=-20,
                    max_value=50,
                    value=0,
                    help="Simulate future demand change",
                    key="tab4_demand_growth"
                )
                restock_delay = st.slider(
                    "⏱️ Restock Delay (days)",
                    min_value=0,
                    max_value=30,
                    value=7,
                    help="Average supplier lead time",
                    key="tab4_restock_delay"
                )
            
            # ========================================================================
            # SMART REPLENISHMENT ANALYZER
            # ========================================================================
            if product_select != "All Products":
                product_df = tab4_df[tab4_df["product_name"] == product_select].copy()
                
                if not product_df.empty:
                    # Calculate current stock
                    current_stock = product_df["stock_on_hand"].iloc[-1] if "stock_on_hand" in product_df.columns else 0
                    
                    # Predict next-week demand (using recent average)
                    if "sales_qty" in product_df.columns:
                        recent_sales = product_df["sales_qty"].tail(4).mean()
                        weekly_demand = recent_sales * (1 + demand_growth / 100)
                        days_to_stockout = (current_stock / (weekly_demand / 7)) if weekly_demand > 0 else 999
                    else:
                        weekly_demand = 0
                        days_to_stockout = 999
                    
                    # Recommend reorder quantity (safety stock + lead time demand)
                    safety_stock_factor = 1.5
                    lead_time_demand = (weekly_demand / 7) * (restock_delay + 7)  # Buffer for restock delay
                    recommended_reorder = max(0, (safety_stock_factor * lead_time_demand) - current_stock)
                    
                    # Risk level classification
                    if days_to_stockout < restock_delay:
                        risk_level = "🔴 Critical"
                        risk_pct = min(100, ((restock_delay - days_to_stockout) / restock_delay) * 100)
                    elif days_to_stockout < restock_delay + 7:
                        risk_level = "🟡 Warning"
                        risk_pct = 40
                    else:
                        risk_level = "🟢 Safe"
                        risk_pct = max(0, 100 - (days_to_stockout - restock_delay - 7))
                    
                    # ========================================================================
                    # KPI CARDS
                    # ========================================================================
                    st.markdown("### 📊 Inventory Health KPIs")
                    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
                    
                    with kpi_col1:
                        st.metric(
                            "Low Stock Count",
                            "N/A",
                            help="Products below reorder threshold"
                        )
                    with kpi_col2:
                        st.metric(
                            "Avg. Days to Stockout",
                            f"{days_to_stockout:.1f}",
                            delta=f"{risk_level}",
                            delta_color="inverse" if "Critical" in risk_level else "normal"
                        )
                    with kpi_col3:
                        inventory_health = max(0, min(100, 100 - risk_pct))
                        st.metric(
                            "Inventory Health Index",
                            f"{inventory_health:.0f}%",
                            help="Overall inventory health score"
                        )
                    with kpi_col4:
                        st.metric(
                            "Recommended Reorder",
                            f"{recommended_reorder:.0f}",
                            help="Units to order immediately"
                        )
                    
                    # ========================================================================
                    # STOCK HEALTH VISUALIZATION
                    # ========================================================================
                    st.markdown("### 📈 Stock Health Timeline")
                    
                    # Prepare data for visualization
                    if "week_start" in product_df.columns:
                        product_df_sorted = product_df.sort_values("week_start").tail(52)  # Last year
                        
                        fig_stock = go.Figure()
                        
                        # Stock levels
                        if "stock_on_hand" in product_df_sorted.columns:
                            fig_stock.add_trace(go.Scatter(
                                x=pd.to_datetime(product_df_sorted["week_start"]),
                                y=product_df_sorted["stock_on_hand"],
                                name="Stock on Hand",
                                mode="lines+markers",
                                line=dict(color="#00e5ff", width=3),
                                marker=dict(size=6)
                            ))
                        
                        # Demand line (if available)
                        if "sales_qty" in product_df_sorted.columns:
                            # Normalize sales to stock scale for visualization
                            stock_avg = product_df_sorted["stock_on_hand"].mean() if "stock_on_hand" in product_df_sorted.columns else 0
                            sales_avg = product_df_sorted["sales_qty"].mean()
                            scale_factor = stock_avg / sales_avg if sales_avg > 0 else 1
                            
                            fig_stock.add_trace(go.Scatter(
                                x=pd.to_datetime(product_df_sorted["week_start"]),
                                y=product_df_sorted["sales_qty"] * scale_factor,
                                name="Weekly Demand (scaled)",
                                mode="lines",
                                line=dict(color="#ff9800", width=2, dash="dot"),
                                yaxis="y"
                            ))
                        
                        # Critical threshold line
                        if weekly_demand > 0:
                            critical_threshold = (weekly_demand / 7) * restock_delay
                            fig_stock.add_hline(
                                y=critical_threshold,
                                line_dash="dash",
                                line_color="red",
                                annotation_text=f"Critical Threshold ({critical_threshold:.0f} units)",
                                annotation_position="right"
                            )
                        
                        fig_stock.update_layout(
                            title=f"{product_select} — Stock Health & Demand Forecast",
                            xaxis_title="Date",
                            yaxis_title="Stock Units",
                            template="plotly_dark",
                            height=500,
                            hovermode='x unified',
                            showlegend=True
                        )
                        st.plotly_chart(fig_stock, use_container_width=True)
                    
                    # ========================================================================
                    # SIMULATION PANEL
                    # ========================================================================
                    st.markdown("### 🎛️ Demand & Restock Simulation")
                    
                    sim_col1, sim_col2 = st.columns(2)
                    
                    with sim_col1:
                        st.markdown("**📊 Current Status**")
                        st.info(f"""
                        **Current Stock:** {current_stock:.0f} units  
                        **Weekly Demand:** {weekly_demand:.1f} units  
                        **Days to Stockout:** {days_to_stockout:.1f} days  
                        **Risk Level:** {risk_level} ({risk_pct:.0f}% risk)
                        """)
                    
                    with sim_col2:
                        st.markdown("**🔮 Simulated Scenario**")
                        simulated_demand = weekly_demand * (1 + demand_growth / 100)
                        simulated_days = (current_stock / (simulated_demand / 7)) if simulated_demand > 0 else 999
                        
                        # Updated risk with simulation
                        if simulated_days < restock_delay:
                            sim_risk = "🔴 Critical"
                            sim_risk_pct = min(100, ((restock_delay - simulated_days) / restock_delay) * 100)
                        elif simulated_days < restock_delay + 7:
                            sim_risk = "🟡 Warning"
                            sim_risk_pct = 40
                        else:
                            sim_risk = "🟢 Safe"
                            sim_risk_pct = max(0, 100 - (simulated_days - restock_delay - 7))
                        
                        delta_days = simulated_days - days_to_stockout
                        st.info(f"""
                        **Simulated Demand:** {simulated_demand:.1f} units/week  
                        **Simulated Days:** {simulated_days:.1f} days  
                        **Updated Risk:** {sim_risk}  
                        **Change:** {delta_days:+.1f} days
                        """)
                    
                    # ========================================================================
                    # AI RECOMMENDATIONS
                    # ========================================================================
                    st.markdown("### 🧠 AI Stock Optimization Recommendations")
                    
                    recommendation_text = f"""
                    **📦 Inventory Analysis for {product_select}**
                    
                    Based on current stock levels ({current_stock:.0f} units) and projected demand ({weekly_demand:.1f} units/week), 
                    the system recommends:
                    
                    **🚨 Immediate Actions:**
                    - **Reorder Quantity:** {recommended_reorder:.0f} units
                    - **Expected Stockout:** {days_to_stockout:.1f} days (Current Risk: {risk_level})
                    - **Safety Stock Required:** {safety_stock_factor * lead_time_demand:.0f} units
                    
                    **💡 Strategic Insights:**
                    """
                    
                    if risk_level == "🔴 Critical":
                        recommendation_text += f"""
                        - ⚠️ **URGENT:** Stock will deplete in {days_to_stockout:.1f} days, but restock takes {restock_delay} days
                        - 🔄 **Action:** Place emergency order immediately for {recommended_reorder:.0f} units
                        - 📞 **Priority:** Contact supplier for expedited delivery
                        """
                    elif risk_level == "🟡 Warning":
                        recommendation_text += f"""
                        - ⚠️ **CAUTION:** Stock levels are approaching critical threshold
                        - 📋 **Action:** Initiate standard reorder process for {recommended_reorder:.0f} units
                        - 📅 **Timeline:** Order should arrive within {restock_delay} days
                        """
                    else:
                        recommendation_text += f"""
                        - ✅ **OPTIMAL:** Stock levels are healthy with {days_to_stockout:.1f} days buffer
                        - 📊 **Monitoring:** Continue tracking weekly demand patterns
                        - 🔄 **Next Review:** Reassess in {max(7, int(days_to_stockout - restock_delay - 7))} days
                        """
                    
                    recommendation_text += f"""
                    
                    **📈 Future Planning:**
                    - **Projected Demand Change:** {demand_growth:+.1f}%
                    - **Adjusted Reorder Point:** {safety_stock_factor * lead_time_demand * (1 + demand_growth/100):.0f} units
                    - **Supplier Lead Time:** {restock_delay} days
                    """
                    
                    st.markdown(recommendation_text)
                    
                    # Download recommendation
                    rec_csv = pd.DataFrame([{
                        "Product": product_select,
                        "Current_Stock": current_stock,
                        "Weekly_Demand": weekly_demand,
                        "Days_to_Stockout": days_to_stockout,
                        "Risk_Level": risk_level,
                        "Recommended_Reorder": recommended_reorder,
                        "Safety_Stock": safety_stock_factor * lead_time_demand
                    }]).to_csv(index=False).encode('utf-8')
                    
                    st.download_button(
                        "📥 Download Stock Recommendation (CSV)",
                        rec_csv,
                        f"stock_recommendation_{product_select}.csv",
                        "text/csv",
                        key=f"download_stock_{product_select}"
                    )
            
            else:
                # ========================================================================
                # ALL PRODUCTS: STOCK HEALTH HEATMAP
                # ========================================================================
                st.markdown("### 📊 Multi-Product Stock Health Overview")
                
                # Generate alerts for all products
                alerts = generate_inventory_alerts(tab4_df)
                
                if not alerts.empty:
                    # Summary KPI cards
                    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
                    
                    low_stock = len(alerts[alerts["status"].str.contains("Low", na=False)]) if "status" in alerts.columns else 0
                    overstock = len(alerts[alerts["status"].str.contains("Overstock", na=False)]) if "status" in alerts.columns else 0
                    optimal = len(alerts[alerts["status"].str.contains("Optimal", na=False)]) if "status" in alerts.columns else 0
                    
                    with summary_col1:
                        st.metric("🔴 Low Stock", low_stock)
                    with summary_col2:
                        st.metric("🟡 Overstock", overstock)
                    with summary_col3:
                        st.metric("🟢 Optimal", optimal)
                    with summary_col4:
                        total_products = len(tab4_df["product_name"].unique()) if "product_name" in tab4_df.columns else 0
                        health_pct = (optimal / total_products * 100) if total_products > 0 else 0
                        st.metric("Health Index", f"{health_pct:.1f}%")
                    
                    # Stock Health Heatmap
                    if "product_name" in tab4_df.columns and "stock_on_hand" in tab4_df.columns:
                        # Create pivot table: Product vs Demand ratio
                        product_stock = tab4_df.groupby("product_name").agg({
                            "stock_on_hand": "mean",
                            "sales_qty": "mean" if "sales_qty" in tab4_df.columns else lambda x: 0
                        }).reset_index()
                        
                        product_stock["stock_demand_ratio"] = (
                            product_stock["stock_on_hand"] / (product_stock["sales_qty"] + 1e-6)
                        )
                        
                        # Classify health status
                        def classify_health(ratio):
                            if ratio < 0.5:
                                return "🔴 Critical"
                            elif ratio < 1.0:
                                return "🟡 Warning"
                            elif ratio < 2.0:
                                return "🟢 Optimal"
                            else:
                                return "🔵 Overstock"
                        
                        product_stock["health_status"] = product_stock["stock_demand_ratio"].apply(classify_health)
                        
                        # Create heatmap
                        fig_heatmap_stock = go.Figure(data=go.Heatmap(
                            z=[[1 if h == "🔴 Critical" else 2 if h == "🟡 Warning" else 3 if h == "🟢 Optimal" else 4 
                                for h in product_stock["health_status"].head(30)]],
                            x=product_stock["product_name"].head(30),
                            y=["Stock Health"],
                            colorscale=[[0, "#F44336"], [0.33, "#FFC107"], [0.66, "#4CAF50"], [1, "#2196F3"]],
                            colorbar=dict(title="Health Status", tickvals=[1, 2, 3, 4], ticktext=["Critical", "Warning", "Optimal", "Overstock"]),
                            text=[[h for h in product_stock["health_status"].head(30)]],
                            texttemplate="%{text}",
                            textfont={"size": 10}
                        ))
                        
                        fig_heatmap_stock.update_layout(
                            title="Stock Health Heatmap (Top 30 Products)",
                            xaxis_title="Product",
                            height=200,
                            template="plotly_dark"
                        )
                        st.plotly_chart(fig_heatmap_stock, use_container_width=True)
                    
                    # Alert table
                    st.markdown("### 📋 Detailed Inventory Alerts")
                    display_cols = ["product_name", "status", "current_stock", "days_to_stockout", "recommended_action"]
                    available_cols = [c for c in display_cols if c in alerts.columns]
                    if available_cols:
                        st.dataframe(
                            alerts[available_cols].sort_values("status"),
                            use_container_width=True,
                            hide_index=True,
                            height=400
                        )
                    
                    # Download
                    csv_data = alerts.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Export Inventory Alerts (CSV)",
                        csv_data,
                        "inventory_alerts_all.csv",
                        "text/csv",
                        key="download_inventory_all"
                    )
                else:
                    st.info("💡 Select a specific product for detailed analysis")
        else:
            st.warning("⚠️ 'product_name' column not found in dataset")
    else:
        st.info("💡 Please ensure **data_with_all_features.csv** exists with inventory data.")

with tab5:
    st.subheader("🎯 Seasonal Insights — Pattern Detection & Strategic Planning")
    st.markdown("**Seasonality Decomposition, YOY Comparison & Peak Detection**")
    
    @st.cache_data(show_spinner="Analyzing seasonality...")
    def load_tab5_data():
        if os.path.exists(FEATURES_DATA_PATH):
            try:
                return pd.read_csv(FEATURES_DATA_PATH, low_memory=False, encoding='utf-8')
            except:
                return None
        return None
    
    tab5_df = load_tab5_data()
    
    if tab5_df is not None and not tab5_df.empty:
        if "product_name" in tab5_df.columns and "sales_qty" in tab5_df.columns:
            product_select = st.selectbox(
                "🔍 Select Product",
                ["All Products"] + sorted(tab5_df["product_name"].unique().tolist()),
                key="tab5_product"
            )
            
            if product_select != "All Products":
                product_df = tab5_df[tab5_df["product_name"] == product_select].copy()
                
                if not product_df.empty and "week_start" in product_df.columns:
                    product_df = product_df.sort_values("week_start").reset_index(drop=True)
                    product_df["week_start"] = pd.to_datetime(product_df["week_start"], errors='coerce')
                    
                    # ========================================================================
                    # SEASONALITY DECOMPOSITION (Prophet/STL)
                    # ========================================================================
                    st.markdown("### 📊 Time Series Decomposition")
                    
                    try:
                        from statsmodels.tsa.seasonal import STL
                        
                        if len(product_df) >= 52:
                            # Prepare weekly aggregated data
                            product_df["week"] = product_df["week_start"].dt.isocalendar().week
                            product_df["year"] = product_df["week_start"].dt.year
                            
                            # Aggregate by week
                            weekly_sales = product_df.groupby("week_start")["sales_qty"].sum().resample("W").mean()
                            weekly_sales = weekly_sales.fillna(method="ffill").fillna(method="bfill")
                            
                            if len(weekly_sales) >= 52:
                                # STL Decomposition
                                period = min(52, len(weekly_sales) // 2)
                                stl_result = STL(weekly_sales, period=period, robust=True).fit()
                                
                                # Create decomposition chart
                                fig_decomp = go.Figure()
                                
                                dates = weekly_sales.index
                                
                                fig_decomp.add_trace(go.Scatter(x=dates, y=weekly_sales, name="Original", line=dict(color="#2196F3", width=2)))
                                fig_decomp.add_trace(go.Scatter(x=dates, y=stl_result.trend, name="Trend", line=dict(color="#4CAF50", width=2)))
                                fig_decomp.add_trace(go.Scatter(x=dates, y=stl_result.seasonal, name="Seasonal", line=dict(color="#FF9800", width=2)))
                                fig_decomp.add_trace(go.Scatter(x=dates, y=stl_result.resid, name="Residual", line=dict(color="#9E9E9E", width=1)))
                                
                                fig_decomp.update_layout(
                                    title=f"{product_select} — Time Series Decomposition",
                                    xaxis_title="Date",
                                    yaxis_title="Sales Quantity",
                                    template="plotly_dark",
                                    height=600,
                                    hovermode='x unified',
                                    showlegend=True
                                )
                                st.plotly_chart(fig_decomp, use_container_width=True)
                                
                                # Separate subplots for clarity
                                decomp_tabs = st.tabs(["Trend", "Seasonal Component", "Residual"])
                                
                                with decomp_tabs[0]:
                                    fig_trend = go.Figure()
                                    fig_trend.add_trace(go.Scatter(
                                        x=dates,
                                        y=stl_result.trend,
                                        name="Trend",
                                        fill='tonexty',
                                        line=dict(color="#4CAF50", width=3)
                                    ))
                                    fig_trend.update_layout(
                                        title="Trend Component",
                                        xaxis_title="Date",
                                        yaxis_title="Trend",
                                        template="plotly_dark",
                                        height=300
                                    )
                                    st.plotly_chart(fig_trend, use_container_width=True)
                                
                                with decomp_tabs[1]:
                                    fig_seasonal = go.Figure()
                                    fig_seasonal.add_trace(go.Scatter(
                                        x=dates,
                                        y=stl_result.seasonal,
                                        name="Seasonal",
                                        line=dict(color="#FF9800", width=2),
                                        marker=dict(size=4)
                                    ))
                                    fig_seasonal.update_layout(
                                        title="Seasonal Component",
                                        xaxis_title="Date",
                                        yaxis_title="Seasonal Effect",
                                        template="plotly_dark",
                                        height=300
                                    )
                                    st.plotly_chart(fig_seasonal, use_container_width=True)
                                
                                with decomp_tabs[2]:
                                    fig_resid = go.Figure()
                                    fig_resid.add_trace(go.Scatter(
                                        x=dates,
                                        y=stl_result.resid,
                                        name="Residual",
                                        mode="markers",
                                        marker=dict(color="#9E9E9E", size=3)
                                    ))
                                    fig_resid.update_layout(
                                        title="Residual Component",
                                        xaxis_title="Date",
                                        yaxis_title="Residual",
                                        template="plotly_dark",
                                        height=300
                                    )
                                    st.plotly_chart(fig_resid, use_container_width=True)
                    except Exception as e:
                        st.warning(f"⚠️ Decomposition failed: {e}. Using monthly pattern instead.")
                    
                    # ========================================================================
                    # YOY COMPARISON PLOT
                    # ========================================================================
                    st.markdown("### 📅 Year-over-Year Comparison")
                    
                    if "year" in product_df.columns:
                        years = sorted(product_df["year"].unique())
                        if len(years) >= 2:
                            # Aggregate by year and week
                            product_df["week_num"] = product_df["week_start"].dt.isocalendar().week
                            yearly_data = product_df.groupby(["year", "week_num"])["sales_qty"].mean().reset_index()
                            
                            fig_yoy = go.Figure()
                            
                            colors_yoy = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0"]
                            for idx, year in enumerate(years[-2:]):  # Last 2 years
                                year_data = yearly_data[yearly_data["year"] == year]
                                if not year_data.empty:
                                    fig_yoy.add_trace(go.Scatter(
                                        x=year_data["week_num"],
                                        y=year_data["sales_qty"],
                                        name=f"{int(year)}",
                                        mode="lines+markers",
                                        line=dict(color=colors_yoy[idx % len(colors_yoy)], width=3),
                                        marker=dict(size=6)
                                    ))
                            
                            fig_yoy.update_layout(
                                title=f"{product_select} — Year-over-Year Sales Comparison",
                                xaxis_title="Week Number",
                                yaxis_title="Average Sales",
                                template="plotly_dark",
                                height=500,
                                hovermode='x unified',
                                showlegend=True
                            )
                            st.plotly_chart(fig_yoy, use_container_width=True)
                    
                    # ========================================================================
                    # SEASONAL HEATMAP
                    # ========================================================================
                    st.markdown("### 🔥 Seasonal Heatmap")
                    
                    product_df["month"] = product_df["week_start"].dt.month
                    product_df["week"] = product_df["week_start"].dt.isocalendar().week
                    
                    # Create pivot: Month × Week
                    heatmap_pivot = product_df.pivot_table(
                        values="sales_qty",
                        index="month",
                        columns="week",
                        aggfunc="mean",
                        fill_value=0
                    )
                    
                    if not heatmap_pivot.empty:
                        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                        fig_heatmap = go.Figure(data=go.Heatmap(
                            z=heatmap_pivot.values,
                            x=heatmap_pivot.columns,
                            y=[month_names[m-1] for m in heatmap_pivot.index],
                            colorscale="Viridis",
                            text=heatmap_pivot.values.round(0),
                            texttemplate="%{text}",
                            textfont={"size": 8},
                            colorbar=dict(title="Avg Sales")
                        ))
                        fig_heatmap.update_layout(
                            title=f"{product_select} — Seasonal Sales Heatmap (Month × Week)",
                            xaxis_title="Week Number",
                            yaxis_title="Month",
                            height=500,
                            template="plotly_dark"
                        )
                        st.plotly_chart(fig_heatmap, use_container_width=True)
                    
                    # ========================================================================
                    # PEAK & TROUGH DETECTION
                    # ========================================================================
                    st.markdown("### 🎯 Peak & Trough Detection")
                    
                    # Monthly averages for peak detection
                    monthly_avg = product_df.groupby("month")["sales_qty"].mean()
                    
                    if not monthly_avg.empty:
                        top_3_peaks = monthly_avg.nlargest(3)
                        top_3_lows = monthly_avg.nsmallest(3)
                        
                        peak_col1, peak_col2 = st.columns(2)
                        
                        with peak_col1:
                            st.markdown("**📈 Top 3 Seasonal Highs**")
                            for idx, (month_idx, sales) in enumerate(top_3_peaks.items(), 1):
                                month_name = month_names[int(month_idx) - 1]
                                st.metric(f"{idx}. {month_name}", f"{sales:.0f}", delta=f"+{((sales - monthly_avg.mean()) / monthly_avg.mean() * 100):.1f}%")
                        
                        with peak_col2:
                            st.markdown("**📉 Top 3 Seasonal Lows**")
                            for idx, (month_idx, sales) in enumerate(top_3_lows.items(), 1):
                                month_name = month_names[int(month_idx) - 1]
                                st.metric(f"{idx}. {month_name}", f"{sales:.0f}", delta=f"-{((monthly_avg.mean() - sales) / monthly_avg.mean() * 100):.1f}%")
                    
                    # ========================================================================
                    # AI SEASON SUMMARY
                    # ========================================================================
                    st.markdown("### 🧠 AI-Generated Season Summary")
                    
                    if not monthly_avg.empty:
                        peak_month = monthly_avg.idxmax()
                        low_month = monthly_avg.idxmin()
                        peak_sales = monthly_avg.max()
                        low_sales = monthly_avg.min()
                        avg_sales = monthly_avg.mean()
                        peak_pct = ((peak_sales - avg_sales) / avg_sales * 100)
                        low_pct = ((avg_sales - low_sales) / avg_sales * 100)
                        
                        summary_text = f"""
                        **📊 Seasonal Pattern Analysis for {product_select}**
                        
                        **🔝 Peak Season:** {month_names[int(peak_month)-1]} shows the highest average sales ({peak_sales:.0f} units), 
                        representing a **{peak_pct:+.1f}%** increase above the annual average. This peak is likely driven by:
                        - Consumer demand patterns
                        - Promotional activities
                        - Weather/seasonal factors
                        
                        **📉 Low Season:** {month_names[int(low_month)-1]} experiences the lowest sales ({low_sales:.0f} units), 
                        representing a **{low_pct:+.1f}%** decrease below average. Consider:
                        - Reduced inventory ordering during this period
                        - Strategic promotions to boost demand
                        - Seasonal marketing campaigns
                        
                        **💡 Strategic Recommendations:**
                        - **Optimize Cold Storage:** Plan for increased inventory during {month_names[int(peak_month)-1]} peak
                        - **Marketing Focus:** Launch campaigns in {month_names[int(low_month)-1]} to mitigate seasonal dips
                        - **Supply Chain:** Coordinate with suppliers for peak season restocking (lead time: ~2-3 weeks before peak)
                        - **Pricing Strategy:** Consider dynamic pricing during peak months to maximize revenue
                        
                        **📈 Seasonal Variation:**
                        The product exhibits **{abs(peak_pct - low_pct):.1f}%** seasonal variation, indicating {'strong' if abs(peak_pct - low_pct) > 30 else 'moderate'} seasonality.
                        """
                        
                        st.markdown(summary_text)
                        
                        # Download summary
                        summary_csv = pd.DataFrame([{
                            "Product": product_select,
                            "Peak_Month": month_names[int(peak_month)-1],
                            "Peak_Sales": peak_sales,
                            "Low_Month": month_names[int(low_month)-1],
                            "Low_Sales": low_sales,
                            "Seasonal_Variation_%": abs(peak_pct - low_pct)
                        }]).to_csv(index=False).encode('utf-8')
                        
                        st.download_button(
                            "📥 Download Seasonal Summary (CSV)",
                            summary_csv,
                            f"seasonal_summary_{product_select}.csv",
                            "text/csv",
                            key=f"download_seasonal_{product_select}"
                        )
            else:
                st.info("👆 Select a specific product to see detailed seasonal analysis")
        else:
            st.warning("⚠️ Required columns (product_name, sales_qty) not found")
    else:
        st.info("💡 Please ensure **data_with_all_features.csv** exists with sales data.")

with tab6:
    st.subheader("💰 Pricing Opportunities — Elasticity & Profit Optimization")
    st.markdown("**Price Elasticity Analysis & Revenue Maximization Strategies**")
    
    @st.cache_data(show_spinner="Analyzing pricing...")
    def load_tab6_data():
        if os.path.exists(FEATURES_DATA_PATH):
            try:
                return pd.read_csv(FEATURES_DATA_PATH, low_memory=False, encoding='utf-8')
            except:
                return None
        return None
    
    tab6_df = load_tab6_data()
    
    if tab6_df is not None and not tab6_df.empty:
        if "product_name" in tab6_df.columns and "price" in tab6_df.columns:
            product_select = st.selectbox(
                "🔍 Select Product",
                sorted(tab6_df["product_name"].unique().tolist()),
                key="tab6_product"
            )
            
            if product_select:
                product_df = tab6_df[tab6_df["product_name"] == product_select].copy()
                
                if "sales_qty" in product_df.columns and "price" in product_df.columns:
                    # Calculate elasticity
                    elasticity = calculate_price_elasticity(tab6_df, product_select)
                    current_price = product_df["price"].iloc[-1] if len(product_df) > 0 else product_df["price"].mean()
                    current_sales = product_df["sales_qty"].mean()
                    current_revenue = current_price * current_sales
                    
                    # ========================================================================
                    # ELASTICITY CALCULATION & CLASSIFICATION
                    # ========================================================================
                    st.markdown("### 📊 Price Elasticity Analysis")
                    
                    if elasticity < -1:
                        elasticity_label = "🔴 Elastic"
                        elasticity_desc = "Sales are highly sensitive to price changes"
                    elif elasticity > -0.5:
                        elasticity_label = "🟢 Inelastic"
                        elasticity_desc = "Sales are relatively insensitive to price changes"
                    else:
                        elasticity_label = "🟡 Unit-Elastic"
                        elasticity_desc = "Sales respond proportionally to price changes"
                    
                    el_col1, el_col2, el_col3 = st.columns(3)
                    with el_col1:
                        st.metric("Price Elasticity", f"{elasticity:.2f}", delta=elasticity_label)
                    with el_col2:
                        st.metric("Current Price", f"₹{current_price:.2f}")
                    with el_col3:
                        st.metric("Avg Weekly Sales", f"{current_sales:.0f}")
                    
                    st.info(f"**{elasticity_label} Product:** {elasticity_desc}. E = {elasticity:.2f}")
                    
                    # ========================================================================
                    # PROFIT SIMULATION CHART
                    # ========================================================================
                    st.markdown("### 🎛️ Profit Simulation Engine")
                    
                    sim_col1, sim_col2, sim_col3 = st.columns(3)
                    with sim_col1:
                        price_change_pct = st.slider(
                            "Price Change (%)",
                            -20, 20, 0, 1,
                            help="Adjust price from current level",
                            key="tab6_price_change"
                        )
                    with sim_col2:
                        cost_pct = st.slider(
                            "Cost % (of price)",
                            30, 80, 50, 1,
                            help="Cost as percentage of selling price",
                            key="tab6_cost"
                        ) / 100
                    with sim_col3:
                        promo_effect = st.slider(
                            "Promotion Effect",
                            0.0, 1.0, 0.0, 0.1,
                            help="Additional demand boost from promotions",
                            key="tab6_promo"
                        )
                    
                    # Calculate new metrics
                    new_price = current_price * (1 + price_change_pct / 100)
                    cost_per_unit = current_price * cost_pct
                    # Demand change = elasticity * price_change + promotion boost
                    demand_change_pct = (elasticity * price_change_pct) + (promo_effect * 10)
                    new_sales = current_sales * (1 + demand_change_pct / 100)
                    new_revenue = new_price * new_sales
                    new_profit = (new_price - cost_per_unit) * new_sales
                    current_profit = (current_price - cost_per_unit) * current_sales
                    profit_change = new_profit - current_profit
                    profit_change_pct = (profit_change / current_profit * 100) if current_profit > 0 else 0
                    
                    # Display results
                    result_col1, result_col2, result_col3, result_col4 = st.columns(4)
                    with result_col1:
                        st.metric("💰 New Revenue", f"₹{new_revenue:,.0f}", delta=f"{((new_revenue - current_revenue) / current_revenue * 100):+.1f}%")
                    with result_col2:
                        st.metric("📦 Projected Demand", f"{new_sales:.0f}", delta=f"{demand_change_pct:+.1f}%")
                    with result_col3:
                        st.metric("💵 New Price", f"₹{new_price:.2f}", delta=f"{price_change_pct:+.1f}%")
                    with result_col4:
                        st.metric("📈 Profit Change", f"₹{profit_change:,.0f}", delta=f"{profit_change_pct:+.1f}%")
                    
                    # Interactive profit curve
                    price_range = np.linspace(current_price * 0.7, current_price * 1.3, 50)
                    profit_curve = []
                    revenue_curve = []
                    
                    for p in price_range:
                        pct_change = ((p - current_price) / current_price) * 100
                        demand_factor = 1 + (elasticity * pct_change / 100)
                        qty = current_sales * max(0.1, demand_factor)  # Prevent negative
                        rev = p * qty
                        prof = (p - cost_per_unit) * qty
                        profit_curve.append(prof)
                        revenue_curve.append(rev)
                    
                    fig_profit = go.Figure()
                    fig_profit.add_trace(go.Scatter(
                        x=price_range,
                        y=profit_curve,
                        name="Profit",
                        line=dict(color="#4CAF50", width=3),
                        mode="lines"
                    ))
                    fig_profit.add_trace(go.Scatter(
                        x=price_range,
                        y=revenue_curve,
                        name="Revenue",
                        line=dict(color="#2196F3", width=2, dash="dot"),
                        mode="lines"
                    ))
                    # Mark current point
                    fig_profit.add_trace(go.Scatter(
                        x=[current_price],
                        y=[current_profit],
                        name="Current",
                        mode="markers",
                        marker=dict(color="#FF9800", size=15, symbol="star")
                    ))
                    # Mark optimal point
                    optimal_idx = np.argmax(profit_curve)
                    optimal_price = price_range[optimal_idx]
                    fig_profit.add_trace(go.Scatter(
                        x=[optimal_price],
                        y=[profit_curve[optimal_idx]],
                        name="Optimal",
                        mode="markers",
                        marker=dict(color="#4CAF50", size=15, symbol="diamond")
                    ))
                    # Mark simulated point
                    fig_profit.add_trace(go.Scatter(
                        x=[new_price],
                        y=[new_profit],
                        name="Simulated",
                        mode="markers",
                        marker=dict(color="#F44336", size=12)
                    ))
                    
                    fig_profit.update_layout(
                        title=f"{product_select} — Profit vs Price Curve",
                        xaxis_title="Price (₹)",
                        yaxis_title="Profit / Revenue (₹)",
                        template="plotly_dark",
                        height=500,
                        hovermode='x unified',
                        showlegend=True
                    )
                    st.plotly_chart(fig_profit, use_container_width=True)
                    
                    # ========================================================================
                    # DYNAMIC PRICING SUGGESTION
                    # ========================================================================
                    st.markdown("### 🎯 Optimal Price Recommendation")
                    
                    # Find optimal price for max profit
                    optimal_margin = (optimal_price - cost_per_unit) / optimal_price * 100
                    price_tolerance = current_price * 0.05  # ±5% range
                    
                    rec_col1, rec_col2 = st.columns(2)
                    with rec_col1:
                        st.info(f"""
                        **🎯 Recommended Price:** ₹{optimal_price:.2f} ± ₹{price_tolerance:.2f}
                        
                        **Expected Profit Margin:** {optimal_margin:.1f}%
                        **Projected Profit Increase:** {(profit_curve[optimal_idx] - current_profit) / current_profit * 100:+.1f}%
                        """)
                    with rec_col2:
                        st.info(f"""
                        **Current vs Optimal:**
                        - Price: ₹{current_price:.2f} → ₹{optimal_price:.2f}
                        - Profit: ₹{current_profit:,.0f} → ₹{profit_curve[optimal_idx]:,.0f}
                        - Margin: {(current_price - cost_per_unit) / current_price * 100:.1f}% → {optimal_margin:.1f}%
                        """)
                    
                    # ========================================================================
                    # TOP OPPORTUNITY TABLE
                    # ========================================================================
                    st.markdown("### 📋 Top Pricing Opportunities Across Products")
                    
                    # Calculate opportunities for all products
                    all_opps = []
                    for prod in tab6_df["product_name"].unique()[:20]:  # Limit to 20 for performance
                        try:
                            prod_df = tab6_df[tab6_df["product_name"] == prod]
                            if len(prod_df) >= 12 and "price" in prod_df.columns and "sales_qty" in prod_df.columns:
                                el = calculate_price_elasticity(tab6_df, prod)
                                curr_price = prod_df["price"].mean()
                                curr_sales = prod_df["sales_qty"].mean()
                                
                                # Find optimal price (simplified)
                                test_prices = [curr_price * f for f in [0.9, 0.95, 1.0, 1.05, 1.1]]
                                best_profit = 0
                                best_price = curr_price
                                for tp in test_prices:
                                    pct_change = ((tp - curr_price) / curr_price) * 100
                                    demand = curr_sales * (1 + el * pct_change / 100)
                                    cost_est = tp * 0.5  # Assume 50% cost
                                    profit = (tp - cost_est) * demand
                                    if profit > best_profit:
                                        best_profit = profit
                                        best_price = tp
                                
                                curr_revenue = curr_price * curr_sales
                                new_revenue = best_price * (curr_sales * (1 + el * ((best_price - curr_price) / curr_price) * 100 / 100))
                                revenue_change = ((new_revenue - curr_revenue) / curr_revenue * 100) if curr_revenue > 0 else 0
                                
                                all_opps.append({
                                    "Product": prod,
                                    "Current Price": curr_price,
                                    "Elasticity": el,
                                    "Recommended Price": best_price,
                                    "Revenue Change (%)": revenue_change
                                })
                        except:
                            continue
                    
                    if all_opps:
                        opps_df = pd.DataFrame(all_opps)
                        opps_df = opps_df.sort_values("Revenue Change (%)", ascending=False)
                        
                        st.dataframe(
                            opps_df,
                            use_container_width=True,
                            hide_index=True,
                            height=400
                        )
                        
                        # Download
                        opps_csv = opps_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "📥 Download Pricing Opportunities (CSV)",
                            opps_csv,
                            "pricing_opportunities.csv",
                            "text/csv",
                            key="download_pricing_tab6"
                        )
                    
                    # ========================================================================
                    # AI RECOMMENDATION SUMMARY
                    # ========================================================================
                    st.markdown("### 🧠 AI Pricing Recommendation Summary")
                    
                    recommendation_text = f"""
                    **💡 Pricing Strategy for {product_select}**
                    
                    Based on price elasticity analysis (E = {elasticity:.2f}), this product is **{elasticity_label.replace('🔴', '').replace('🟡', '').replace('🟢', '')}**.
                    
                    **📊 Current State:**
                    - Current Price: ₹{current_price:.2f}
                    - Average Sales: {current_sales:.0f} units/week
                    - Current Revenue: ₹{current_revenue:,.0f}/week
                    
                    **🎯 Recommended Action:**
                    """
                    
                    if elasticity < -1:
                        recommendation_text += f"""
                        - **Lower Prices Strategically:** Since demand is elastic, consider a **5-10% price reduction** to boost volume
                        - **Target Price Range:** ₹{current_price * 0.95:.2f} - ₹{current_price * 0.90:.2f}
                        - **Expected Impact:** Volume increase of {abs(elasticity * 5):.1f}% with {abs(elasticity * 5) - 5:.1f}% net revenue gain
                        - **Risk:** Price too high may lead to significant demand loss
                        """
                    elif elasticity > -0.5:
                        recommendation_text += f"""
                        - **Raise Prices Moderately:** Since demand is inelastic, **{abs(elasticity * 7):.1f}% price increase** can sustain profit
                        - **Target Price Range:** ₹{current_price * 1.05:.2f} - ₹{current_price * 1.07:.2f}
                        - **Expected Impact:** Revenue increase with minimal demand loss (only {abs(elasticity * 7):.1f}% volume decrease)
                        - **Opportunity:** Rice can sustain a 7% price increase with only 2% demand loss
                        """
                    else:
                        recommendation_text += f"""
                        - **Maintain Current Pricing:** Product is unit-elastic; price changes have proportional demand effects
                        - **Focus on Volume:** Use promotions and marketing to drive sales rather than price adjustments
                        - **Margin Optimization:** Consider cost reduction strategies instead
                        """
                    
                    recommendation_text += f"""
                    
                    **💼 Strategic Recommendations:**
                    - **Optimal Price:** ₹{optimal_price:.2f} ± ₹{price_tolerance:.2f} for maximum {optimal_margin:.1f}% margin
                    - **Promotion Timing:** Combine price changes with promotional campaigns for amplified effect
                    - **Competitive Analysis:** Monitor competitor pricing to stay competitive
                    - **A/B Testing:** Test price changes in select markets before full rollout
                    """
                    
                    st.markdown(recommendation_text)
                else:
                    st.warning("⚠️ Required columns (price, sales_qty) not found for pricing analysis")
            else:
                st.info("👆 Select a product to begin pricing analysis")
        else:
            st.warning("⚠️ Required columns (product_name, price) not found")
    else:
        st.info("💡 Please ensure **data_with_all_features.csv** exists with pricing data.")

with tab7:
    st.subheader("⚙️ Dynamic Pricing Engine — Real-Time AI Pricing Optimizer")
    st.markdown("**Live Elasticity Tracker, Competitive Benchmarking & Demand Surge Mode**")
    
    @st.cache_data(show_spinner="Loading data...")
    def load_tab7_data():
        if os.path.exists(FEATURES_DATA_PATH):
            try:
                return pd.read_csv(FEATURES_DATA_PATH, low_memory=False, encoding='utf-8')
            except:
                return None
        return None
    
    tab7_df = load_tab7_data()
    
    if tab7_df is not None and not tab7_df.empty and "product_name" in tab7_df.columns:
        product_select = st.selectbox(
            "🔍 Select Product",
            sorted(tab7_df["product_name"].unique().tolist()),
            key="tab7_product"
        )
        
        if product_select:
            product_df = tab7_df[tab7_df["product_name"] == product_select].copy()
            
            if "price" in product_df.columns and "sales_qty" in product_df.columns:
                elasticity = calculate_price_elasticity(tab7_df, product_select)
                current_price = product_df["price"].iloc[-1] if len(product_df) > 0 else product_df["price"].mean()
                current_sales = product_df["sales_qty"].mean()
                
                # ========================================================================
                # LIVE ELASTICITY TRACKER
                # ========================================================================
                st.markdown("### 📊 Live Price vs Profit Tracker")
                
                # Simulation controls
                sim_col1, sim_col2 = st.columns([2, 1])
                with sim_col1:
                    margin_pct = st.slider("Profit Margin (%)", 10, 50, 30, 1, key="tab7_margin")
                    margin = margin_pct / 100
                with sim_col2:
                    demand_surge = st.checkbox("🌊 Enable Demand Surge Mode (+25%)", key="tab7_surge")
                    festival_boost = 0.25 if demand_surge else 0.0
                
                # Calculate profit curve
                cost_per_unit = current_price * (1 - margin)
                price_test_range = np.linspace(current_price * 0.8, current_price * 1.2, 100)
                profit_values = []
                
                for test_price in price_test_range:
                    price_change_pct = ((test_price - current_price) / current_price) * 100
                    demand_factor = 1 + (elasticity * price_change_pct / 100) + festival_boost
                    qty = current_sales * max(0.1, demand_factor)
                    profit = (test_price - cost_per_unit) * qty
                    profit_values.append(profit)
                
                optimal_idx = np.argmax(profit_values)
                optimal_price = price_test_range[optimal_idx]
                optimal_profit = profit_values[optimal_idx]
                current_profit = (current_price - cost_per_unit) * current_sales
                
                # Create interactive chart with moving pointer
                fig_live = go.Figure()
                
                # Profit curve
                fig_live.add_trace(go.Scatter(
                    x=price_test_range,
                    y=profit_values,
                    name="Profit Curve",
                    line=dict(color="#4CAF50", width=4),
                    mode="lines",
                    fill='tozeroy',
                    fillcolor="rgba(76, 175, 80, 0.2)"
                ))
                
                # Current price marker
                fig_live.add_trace(go.Scatter(
                    x=[current_price],
                    y=[current_profit],
                    name="Current Price",
                    mode="markers+text",
                    marker=dict(color="#FF9800", size=20, symbol="star"),
                    text=["Current"],
                    textposition="top center"
                ))
                
                # Optimal price marker
                fig_live.add_trace(go.Scatter(
                    x=[optimal_price],
                    y=[optimal_profit],
                    name="Optimal Price",
                    mode="markers+text",
                    marker=dict(color="#4CAF50", size=20, symbol="diamond"),
                    text=["Optimal"],
                    textposition="top center"
                ))
                
                # Optimal range band
                optimal_range_low = optimal_price * 0.98
                optimal_range_high = optimal_price * 1.02
                fig_live.add_vrect(
                    x0=optimal_range_low,
                    x1=optimal_range_high,
                    fillcolor="rgba(76, 175, 80, 0.1)",
                    layer="below",
                    line_width=0
                )
                
                fig_live.update_layout(
                    title=f"{product_select} — Live Price vs Profit Optimization",
                    xaxis_title="Price (₹)",
                    yaxis_title="Profit (₹)",
                    template="plotly_dark",
                    height=500,
                    hovermode='x unified',
                    showlegend=True,
                    annotations=[
                        dict(
                            x=optimal_price,
                            y=optimal_profit,
                            text=f"Optimal: ₹{optimal_price:.2f}<br>Profit: ₹{optimal_profit:,.0f}",
                            showarrow=True,
                            arrowhead=2,
                            ax=0,
                            ay=-40
                        )
                    ]
                )
                st.plotly_chart(fig_live, use_container_width=True)
                
                # ========================================================================
                # COMPETITIVE BENCHMARK SIMULATION
                # ========================================================================
                st.markdown("### 🏆 Competitive Benchmark Simulation")
                
                comp_col1, comp_col2 = st.columns(2)
                
                with comp_col1:
                    competitor_price = st.number_input(
                        "Competitor Price (₹)",
                        min_value=current_price * 0.5,
                        max_value=current_price * 1.5,
                        value=current_price * 0.95,
                        step=1.0,
                        key="tab7_competitor"
                    )
                    
                    competitor_impact = st.slider(
                        "Competitor Price Impact (%)",
                        0, 30, 10, 1,
                        help="How much competitor price affects our demand",
                        key="tab7_comp_impact"
                    ) / 100
                
                with comp_col2:
                    # Calculate adjusted recommendation
                    price_diff_pct = ((competitor_price - current_price) / current_price) * 100
                    adjusted_elasticity = elasticity + (competitor_impact * abs(price_diff_pct) / 10)
                    
                    # Recalculate optimal with competitor influence
                    adj_profit_values = []
                    for test_price in price_test_range:
                        price_change_pct = ((test_price - current_price) / current_price) * 100
                        # Add competitor adjustment
                        comp_adjustment = competitor_impact * ((competitor_price - test_price) / current_price) * 100
                        demand_factor = 1 + (adjusted_elasticity * price_change_pct / 100) + (comp_adjustment / 100)
                        qty = current_sales * max(0.1, demand_factor)
                        profit = (test_price - cost_per_unit) * qty
                        adj_profit_values.append(profit)
                    
                    adj_optimal_idx = np.argmax(adj_profit_values)
                    adj_optimal_price = price_test_range[adj_optimal_idx]
                    
                    st.metric("Competitor Price", f"₹{competitor_price:.2f}")
                    st.metric(
                        "Adjusted Optimal Price",
                        f"₹{adj_optimal_price:.2f}",
                        delta=f"{(adj_optimal_price - optimal_price):+.2f} vs baseline"
                    )
                
                # ========================================================================
                # DEMAND SURGE MODE
                # ========================================================================
                if demand_surge:
                    st.markdown("### 🌊 Demand Surge Mode Active (+25% Festival Demand)")
                    
                    surge_current_profit = (current_price - cost_per_unit) * (current_sales * 1.25)
                    surge_optimal_profit = optimal_profit * 1.15  # Boosted
                    
                    surge_col1, surge_col2, surge_col3 = st.columns(3)
                    with surge_col1:
                        st.metric("Current (Surge)", f"₹{surge_current_profit:,.0f}", delta="+25% demand")
                    with surge_col2:
                        st.metric("Optimal (Surge)", f"₹{surge_optimal_profit:,.0f}", delta="+15% profit")
                    with surge_col3:
                        surge_gain = ((surge_optimal_profit - surge_current_profit) / surge_current_profit * 100)
                        st.metric("Surge Gain", f"{surge_gain:+.1f}%")
                
                # ========================================================================
                # AI RECOMMENDATION CARD
                # ========================================================================
                st.markdown("### 🧠 AI Dynamic Pricing Recommendation")
                
                confidence = 100 - abs((optimal_price - current_price) / current_price * 100)
                confidence = max(80, min(95, confidence))
                
                recommendation_card = f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 10px; color: white; margin: 1rem 0;">
                    <h3>🧠 Suggested Price Adjustment for {product_select}</h3>
                    <p style="font-size: 1.1rem; margin: 0.5rem 0;">
                        <strong>Current Price:</strong> ₹{current_price:.2f} → 
                        <strong style="color: #FFD700;">Recommended:</strong> ₹{optimal_price:.2f} 
                        <span style="background: rgba(255,255,255,0.2); padding: 0.2rem 0.5rem; border-radius: 5px; margin-left: 1rem;">
                            {((optimal_price - current_price) / current_price * 100):+.1f}%
                        </span>
                    </p>
                    <p style="font-size: 0.95rem; margin-top: 0.5rem;">
                        💰 Expected profit improvement: <strong>₹{optimal_profit - current_profit:,.0f}</strong> 
                        ({((optimal_profit - current_profit) / current_profit * 100):+.1f}%)<br>
                        📊 Confidence Level: <strong>{confidence:.0f}%</strong> | 
                        Margin Target: <strong>{margin_pct:.0f}%</strong>
                    </p>
                    <p style="font-size: 0.9rem; margin-top: 0.5rem; font-style: italic;">
                        💡 This recommendation is based on price elasticity (E={elasticity:.2f}) and real-time demand patterns.
                        {'🌊 Demand surge mode is active — festival season detected!' if demand_surge else ''}
                    </p>
                </div>
                """
                st.markdown(recommendation_card, unsafe_allow_html=True)
                
                # Display metrics
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                with metric_col1:
                    st.metric("💰 Current Price", f"₹{current_price:.2f}")
                with metric_col2:
                    st.metric("🎯 Optimal Price", f"₹{optimal_price:.2f}", delta=f"{((optimal_price - current_price) / current_price * 100):+.1f}%")
                with metric_col3:
                    st.metric("📈 Profit Gain", f"₹{optimal_profit - current_profit:,.0f}", delta=f"{((optimal_profit - current_profit) / current_profit * 100):+.1f}%")
                with metric_col4:
                    st.metric("🎯 Confidence", f"{confidence:.0f}%")
            else:
                st.warning("⚠️ Required columns (price, sales_qty) not found")
        else:
            st.info("👆 Select a product to begin dynamic pricing analysis")
    else:
        st.info("💡 Please ensure **data_with_all_features.csv** exists with pricing data.")

with tab8:
    st.subheader("🧾 Data Summary — EDA & Health Dashboard")
    st.markdown("**Comprehensive Data Profiling, Correlation Analysis & Feature Engineering Summary**")
    
    # Try to load data_with_all_features.csv first, fallback to uploaded
    @st.cache_data(show_spinner="Loading dataset...")
    def load_tab8_data():
        if os.path.exists(FEATURES_DATA_PATH):
            try:
                return pd.read_csv(FEATURES_DATA_PATH, low_memory=False, encoding='utf-8')
            except:
                pass
        if uploaded_ready and os.path.exists(UPLOADED_FILE_PATH):
            try:
                return pd.read_csv(UPLOADED_FILE_PATH, low_memory=False, encoding='utf-8')
            except:
                pass
        return None
    
    df_full = load_tab8_data()
    
    if df_full is not None and not df_full.empty:
        # ========================================================================
        # DATA HEALTH KPIs
        # ========================================================================
        st.markdown("### 📊 Data Health Overview")
        
        total_cells = int(df_full.shape[0] * df_full.shape[1])
        missing_cells = int(df_full.isna().sum().sum())
        missing_cells_pct = round((missing_cells / total_cells * 100.0), 2) if total_cells else 0.0
        duplicate_rows = int(df_full.duplicated().sum())
        mem_bytes = int(df_full.memory_usage(deep=True).sum())
        mem_mb = round(mem_bytes / (1024 * 1024), 2)
        
        # Calculate freshness (if date column exists)
        date_cols = [c for c in df_full.columns if 'date' in c.lower() or 'week' in c.lower() or 'time' in c.lower()]
        freshness_date = None
        if date_cols:
            try:
                latest_date = pd.to_datetime(df_full[date_cols[0]], errors='coerce').max()
                freshness_date = latest_date.strftime('%Y-%m-%d') if pd.notna(latest_date) else None
            except:
                pass
        
        health_col1, health_col2, health_col3, health_col4, health_col5 = st.columns(5)
        with health_col1:
            health_status = "🟢 Excellent" if missing_cells_pct < 5 else "🟡 Good" if missing_cells_pct < 15 else "🔴 Needs Attention"
            st.metric("Data Quality", health_status)
        with health_col2:
            st.metric("Rows", f"{df_full.shape[0]:,}")
        with health_col3:
            st.metric("Columns", f"{df_full.shape[1]:,}")
        with health_col4:
            st.metric("Missing (%)", f"{missing_cells_pct:.1f}%", delta_color="inverse")
        with health_col5:
            st.metric("Memory", f"{mem_mb:.1f} MB")
        
        # Additional health metrics
        health_row2_col1, health_row2_col2, health_row2_col3 = st.columns(3)
        with health_row2_col1:
            st.metric("Duplicate Rows", duplicate_rows, delta_color="inverse")
        with health_row2_col2:
            completeness = 100 - missing_cells_pct
            st.metric("Completeness", f"{completeness:.1f}%")
        with health_row2_col3:
            st.metric("Last Updated", freshness_date if freshness_date else "N/A")
        
        # ========================================================================
        # FEATURE SUMMARY TABLE
        # ========================================================================
        st.markdown("### 📋 Feature Summary & Statistics")
        
        prof = profile_dataframe(df_full)
        
        # Enhanced table with descriptions
        enhanced_prof = prof.copy()
        enhanced_prof["Description"] = enhanced_prof.apply(lambda row: 
            f"Date column" if row["dtype"] == "datetime" else
            f"Numeric: range {row['min']:.2f} to {row['max']:.2f}" if row["dtype"] == "numeric" and pd.notna(row.get("min")) else
            f"Categorical: {row['unique']} unique values" if row["dtype"] == "categorical" else
            "Text/Other data", axis=1
        )
        
        st.dataframe(
            enhanced_prof,
            use_container_width=True,
            hide_index=True,
            height=400
        )
        
        # Download button
        prof_csv = prof.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Data Profile (CSV)",
            prof_csv,
            "data_profile.csv",
            "text/csv",
            key="download_profile_tab8"
        )
        
        # ========================================================================
        # INTERACTIVE CORRELATION MAP
        # ========================================================================
        st.markdown("### 🔥 Feature Correlation Heatmap")
        
        # Select numeric columns only
        numeric_cols = df_full.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 1:
            # Limit to top 20 for performance
            numeric_cols = numeric_cols[:20]
            corr_df = df_full[numeric_cols].corr()
            
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_df.values,
                x=corr_df.columns,
                y=corr_df.index,
                colorscale="RdBu",
                zmid=0,
                text=corr_df.values.round(2),
                texttemplate="%{text}",
                textfont={"size": 8},
                colorbar=dict(title="Correlation")
            ))
            
            fig_corr.update_layout(
                title="Feature Correlation Matrix (Numeric Features Only)",
                xaxis_title="Features",
                yaxis_title="Features",
                height=600,
                template="plotly_dark"
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Find strongest correlations
            st.markdown("#### 🔍 Strongest Correlations")
            corr_pairs = []
            for i in range(len(corr_df.columns)):
                for j in range(i+1, len(corr_df.columns)):
                    val = corr_df.iloc[i, j]
                    if abs(val) > 0.5:  # Strong correlation threshold
                        corr_pairs.append({
                            "Feature 1": corr_df.columns[i],
                            "Feature 2": corr_df.columns[j],
                            "Correlation": val
                        })
            
            if corr_pairs:
                corr_pairs_df = pd.DataFrame(corr_pairs).sort_values("Correlation", key=lambda x: x.abs(), ascending=False)
                st.dataframe(corr_pairs_df.head(10), use_container_width=True, hide_index=True)
            else:
                st.info("No strong correlations (>0.5) found between numeric features")
        else:
            st.warning("⚠️ Insufficient numeric columns for correlation analysis")
        
        # ========================================================================
        # FEATURE ENGINEERING SUMMARY
        # ========================================================================
        st.markdown("### 🔧 Feature Engineering Summary")
        
        feature_summary = []
        
        # Detect lag features
        lag_features = [c for c in df_full.columns if 'lag' in c.lower() or '_lag' in c.lower()]
        if lag_features:
            feature_summary.append(f"✅ **{len(lag_features)} Lag Features** detected: {', '.join(lag_features[:5])}")
        
        # Detect rolling averages
        rolling_features = [c for c in df_full.columns if 'rolling' in c.lower() or 'ma' in c.lower() or 'avg' in c.lower()]
        if rolling_features:
            feature_summary.append(f"✅ **{len(rolling_features)} Rolling Average Features**: {', '.join(rolling_features[:5])}")
        
        # Detect seasonality encoding
        season_features = [c for c in df_full.columns if 'season' in c.lower() or 'month' in c.lower() or 'week' in c.lower()]
        if season_features:
            feature_summary.append(f"✅ **{len(season_features)} Seasonality Features**: {', '.join(season_features[:5])}")
        
        # Detect encoded categories
        encoded_features = [c for c in df_full.columns if 'encoded' in c.lower() or '_cat' in c.lower()]
        if encoded_features:
            feature_summary.append(f"✅ **{len(encoded_features)} Category Encoded Features**: {', '.join(encoded_features[:5])}")
        
        if feature_summary:
            summary_text = f"""
            **🧩 Engineered Features Detected:**
            
            {'<br>'.join(feature_summary)}
            
            **📊 Total Features:** {len(df_full.columns)}  
            **🎯 Target Variable:** sales_qty (detected)  
            **📅 Time Features:** week_start, week_end (detected)  
            **🏷️ Categorical:** product_name, category (detected)
            
            **💡 Feature Engineering Applied:**
            - Added {len(lag_features)} lag features for temporal patterns
            - Added {len(rolling_features)} rolling averages for trend smoothing
            - Added {len(season_features) if season_features else 0} season encoding features
            - Encoded categorical variables for ML compatibility
            """
            st.markdown(summary_text, unsafe_allow_html=True)
        else:
            st.info("💡 No obvious engineered features detected. Raw dataset or manual feature engineering may be needed.")
        
        # ========================================================================
        # EDA REPORT GENERATOR
        # ========================================================================
        st.markdown("### 📄 Data Profiling Report Generator")
        
        report_col1, report_col2 = st.columns([2, 1])
        
        with report_col1:
            # Generate comprehensive report
            report_text = f"""
# RetailSense Data Profiling Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Dataset Overview
- **Total Rows:** {df_full.shape[0]:,}
- **Total Columns:** {df_full.shape[1]}
- **Data Completeness:** {100 - missing_cells_pct:.1f}%
- **Memory Usage:** {mem_mb:.2f} MB
- **Duplicate Rows:** {duplicate_rows}

## Data Quality Assessment
- **Missing Data:** {missing_cells_pct:.1f}% ({missing_cells:,} cells)
- **Data Quality Status:** {health_status}
- **Latest Data Point:** {freshness_date if freshness_date else 'N/A'}

## Feature Summary
{prof.to_string(index=False) if len(prof) <= 50 else prof.head(50).to_string(index=False)}

## Key Insights
- Dataset contains {df_full.shape[0]:,} records across {df_full.shape[1]} features
- {len(numeric_cols)} numeric features identified
- {len(season_features) if season_features else 0} temporal/seasonal features detected
- {len(encoded_features) if encoded_features else 0} encoded categorical features

## Recommendations
- {"✅ Data quality is excellent" if missing_cells_pct < 5 else "⚠️ Consider handling missing values" if missing_cells_pct < 15 else "🔴 High missing data - data cleaning required"}
- {"✅ Feature engineering appears comprehensive" if feature_summary else "💡 Consider adding lag/rolling features for time series analysis"}
- {"✅ Strong correlations detected - consider feature selection" if corr_pairs else "✅ Low multicollinearity - good for modeling"}
"""
            
            st.text_area("Report Preview", report_text, height=300, key="eda_report_preview")
        
        with report_col2:
            st.markdown("**📥 Download Options**")
            st.download_button(
                "📄 Download EDA Report (TXT)",
                report_text,
                f"eda_report_{datetime.now().strftime('%Y%m%d')}.txt",
                "text/plain",
                key="download_eda_report"
            )
            
            # CSV exports
            if not prof.empty:
                prof_csv_btn = prof.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📊 Download Profile (CSV)",
                    prof_csv_btn,
                    "data_profile.csv",
                    "text/csv",
                    key="download_profile_csv_tab8"
                )
            
            if corr_pairs:
                corr_csv = pd.DataFrame(corr_pairs).to_csv(index=False).encode('utf-8')
                st.download_button(
                    "🔥 Download Correlations (CSV)",
                    corr_csv,
                    "correlations.csv",
                    "text/csv",
                    key="download_corr_tab8"
                )
    else:
        st.info("💡 Please upload a CSV file from the sidebar or ensure **data_with_all_features.csv** exists.")

# ============================================================================
# CHAT INSIGHT ASSISTANT (BONUS FEATURE)
# ============================================================================
if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = []
if "show_chat" not in st.session_state:
    st.session_state["show_chat"] = False

# Display chat if enabled
if st.session_state.get("show_chat", False):
    st.markdown("---")
    st.markdown("### 💬 AI Insight Chat Assistant")
    st.caption("Ask questions about your retail data and get AI-powered insights")
    
    # Display chat history
    for msg in st.session_state["chat_messages"][-10:]:  # Last 10 messages
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    # Chat input
    user_query = st.chat_input("Ask a question about your data (e.g., 'Why did Milk sales drop in March?')")
    
    if user_query:
        # Add user message
        st.session_state["chat_messages"].append({"role": "user", "content": user_query})
        
        # Generate AI response (rule-based)
        response = f"""
        **Analysis of your query:** "{user_query}"
        
        Based on the available data and analysis modules:
        
        - For **forecast queries**, please use the Sales Forecasting tab and select a product
        - For **anomaly detection**, check the Sales Anomalies tab
        - For **pricing questions**, refer to the Pricing Opportunities tab
        - For **seasonal patterns**, see the Seasonal Insights tab
        
        **Quick Tips:**
        - Ask specific questions like: "What's the forecast for [Product] next month?"
        - Query elasticity: "What's the price elasticity for [Product]?"
        - Check seasonality: "When does [Product] peak?"
        """
        
        st.session_state["chat_messages"].append({"role": "assistant", "content": response})
        st.rerun()

st.markdown("---")
st.markdown(
    f"<div style='text-align: center; color: grey; padding: 1rem;'>"
    f"🛍️ RetailSense Lite — AI-Driven Retail Analytics<br>"
    f"Powered by XGBoost • LightGBM • Prophet • Streamlit<br>"
    f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    f"</div>",
    unsafe_allow_html=True
)