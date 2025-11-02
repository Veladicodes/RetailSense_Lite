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
from sklearn.metrics import mean_absolute_error, mean_squared_error
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
    from utils.advanced_forecasting import train_ensemble, train_ensemble_for_app, run_hybrid_forecast, simulate_forecast_with_scenarios, run_advanced_forecast
    from utils.business_insights import (
        detect_sales_anomalies, generate_inventory_alerts, analyze_seasonality,
        calculate_price_elasticity, analyze_pricing_opportunities, optimize_price,
        generate_executive_summary, generate_forecast_insights, calculate_scenario_impact
    )
    from utils.data_loader import load_dataset, preprocess_data
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
st.set_page_config(page_title="RetailSense Dashboard", layout="wide")

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
st.sidebar.image("https://img.icons8.com/color/96/000000/shopify.png", width=120)
st.sidebar.title("⚙️ RetailSense Controls")

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
        st.session_state.pop("colmap", None)
        st.session_state.pop("selected_product_id", None)
        st.session_state["force_sales_tab"] = False
        
        # Clear cached data
        st.session_state.pop("df_full_cached", None)
        st.session_state.pop("df_profile_cached", None)
        st.session_state.pop("auto_map_cached", None)
        
        st.sidebar.success(f"✅ File uploaded successfully!")
    else:
        # Same file still uploaded - ensure flag is set
        st.session_state["uploaded_ready"] = True
    
    # Load dataset only once and cache in session state
    if "df_full_cached" not in st.session_state:
        try:
            with st.spinner("Loading data..."):
                df_full = pd.read_csv(UPLOADED_FILE_PATH)
                st.session_state["df_full_cached"] = df_full
        except Exception as e:
            st.sidebar.error(f"❌ Unable to read uploaded CSV: {e}")
            df_full = None
    else:
        df_full = st.session_state["df_full_cached"]

    if df_full is not None and not df_full.empty:
        # Quick summary (lightweight)
        st.sidebar.subheader("📋 Quick Summary")
        c1, c2 = st.sidebar.columns(2)
        c1.metric("Rows", f"{df_full.shape[0]:,}")
        c2.metric("Columns", f"{df_full.shape[1]:,}")
        
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

        # Column Mapping Controls
        st.sidebar.subheader("🗂️ Column Mapping")
        
        # Cache auto-mapping
        if "auto_map_cached" not in st.session_state:
            st.session_state["auto_map_cached"] = identify_columns(df_full)
        auto_map = st.session_state["auto_map_cached"]
        
        prev_map = st.session_state.get("colmap", {})
        colnames = df_full.columns.tolist()
        sel_date = st.sidebar.selectbox("Date column", options=colnames, index=(colnames.index(prev_map.get("date", auto_map.get("date"))) if auto_map.get("date") in colnames or prev_map.get("date") in colnames else 0), key="map_date")
        sel_product = st.sidebar.selectbox("Product column", options=colnames, index=(colnames.index(prev_map.get("product", auto_map.get("product"))) if auto_map.get("product") in colnames or prev_map.get("product") in colnames else 0), key="map_product")
        sel_sales = st.sidebar.selectbox("Sales column", options=colnames, index=(colnames.index(prev_map.get("sales", auto_map.get("sales"))) if auto_map.get("sales") in colnames or prev_map.get("sales") in colnames else 0), key="map_sales")
        sel_category = st.sidebar.selectbox("Category column (optional)", options=["<None>"] + colnames, index=(1 + colnames.index(prev_map.get("category", auto_map.get("category"))) if (prev_map.get("category") or auto_map.get("category")) in colnames else 0), key="map_category")
        sel_price = st.sidebar.selectbox("Price column (optional)", options=["<None>"] + colnames, index=(1 + colnames.index(prev_map.get("price", auto_map.get("price"))) if (prev_map.get("price") or auto_map.get("price")) in colnames else 0), key="map_price")
        sel_stock = st.sidebar.selectbox("Stock column (optional)", options=["<None>"] + colnames, index=(1 + colnames.index(prev_map.get("stock", auto_map.get("stock"))) if (prev_map.get("stock") or auto_map.get("stock")) in colnames else 0), key="map_stock")
        sel_store = st.sidebar.selectbox("Store/Location column (optional)", options=["<None>"] + colnames, index=(1 + colnames.index(prev_map.get("store", auto_map.get("store"))) if (prev_map.get("store") or auto_map.get("store")) in colnames else 0), key="map_store")
        if st.sidebar.button("✅ Apply Mapping"):
            colmap = {
                "date": sel_date,
                "product": sel_product,
                "sales": sel_sales,
                "category": (None if sel_category == "<None>" else sel_category),
                "price": (None if sel_price == "<None>" else sel_price),
                "stock": (None if sel_stock == "<None>" else sel_stock),
                "store": (None if sel_store == "<None>" else sel_store),
            }
            valid, missing = validate_colmap(colmap, df_full)
            if valid:
                st.session_state["colmap"] = colmap
                st.sidebar.success("Column mapping saved for this session.")
            else:
                st.sidebar.error(f"Missing required mappings: {', '.join(missing)}")

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
    
    # Load data_with_all_features.csv directly
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
    
    if features_df is None or features_df.empty:
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
    
    # Product Selection - Use features_df directly
    if "product_name" not in features_df.columns:
        st.error("❌ 'product_name' column not found in dataset")
        st.stop()
    
    product_list = sorted(features_df["product_name"].unique().tolist())
    
    col_sel1, col_sel2 = st.columns([3, 1])
    with col_sel1:
        selected_product = st.selectbox(
            "🔍 Choose Product for Analysis",
            options=["-- Select Product --"] + product_list,
            key="forecast_product_selector"
        )
        # Store the current tab in session state to prevent redirection
        st.session_state["current_tab"] = "Sales Forecasting"
    with col_sel2:
        if selected_product and selected_product != "-- Select Product --":
            product_stats = features_df[features_df["product_name"] == selected_product]
            avg_sales = product_stats["sales_qty"].mean() if "sales_qty" in product_stats.columns else 0
            st.metric("Avg Weekly Sales", f"{avg_sales:.0f}")
    
    if selected_product and selected_product != "-- Select Product --":
        # Filter data for selected product
        product_df = features_df[features_df["product_name"] == selected_product].copy()
        product_df = product_df.sort_values("week_start").reset_index(drop=True)
        
        if len(product_df) < 8:
            st.warning(f"⚠️ Insufficient data for {selected_product}. Need at least 8 weeks of history.")
            st.stop()
        
        # Create sub-tabs for all modules
        subtab1, subtab2, subtab3, subtab4, subtab5, subtab6, subtab7 = st.tabs([
            "📈 Forecast Explorer",
            "🚨 Sales Anomalies", 
            "📦 Inventory Alerts",
            "🌦️ Seasonal Insights",
            "💰 Pricing Opportunities",
            "⚙️ Dynamic Pricing Engine",
            "📋 Executive Summary"
        ])
        
        with subtab1:
            st.subheader("📈 Sales Forecasting / Forecast Explorer")
            st.markdown("**🚀 Industry-Grade Hybrid Forecasting Engine: Prophet + XGBoost + LightGBM**")
            
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
            # LAST 10 WEEKS SUMMARY TABLE
            # ========================================================================
            st.markdown("### 📊 Recent Sales History (Last 10 Weeks)")
            last_10_weeks = ts.tail(10).copy()
            if "price" in product_df.columns:
                price_data = product_df[product_df["week_start"].isin(last_10_weeks["date"])][["week_start", "price"]].copy()
                price_data.columns = ["date", "price"]
                last_10_weeks = last_10_weeks.merge(price_data, on="date", how="left")
            
            summary_table = last_10_weeks[["date", "sales_qty"]].copy()
            summary_table.columns = ["Week", "Sales Qty"]
            summary_table["Week"] = summary_table["Week"].dt.strftime("%Y-%m-%d")
            if "price" in last_10_weeks.columns:
                summary_table["Price"] = last_10_weeks["price"].round(2)
            summary_table["Sales Qty"] = summary_table["Sales Qty"].round(0).astype(int)
            
            st.dataframe(summary_table, use_container_width=True, hide_index=True)
            st.markdown("---")
            
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
                
                # Top Opportunity Week annotation
                if len(forecast_df) >= 5:
                    top_week = forecast_df.nlargest(1, "yhat").iloc[0]
                    # Keep as pandas Timestamp to match trace format
                    top_week_date = top_week["date"]
                    if not isinstance(top_week_date, pd.Timestamp):
                        top_week_date = pd.to_datetime(top_week_date)
                    
                    fig_main.add_annotation(
                        x=top_week_date,
                        y=top_week["yhat"],
                        text="🎯 Top Opportunity Week",
                        showarrow=True,
                        arrowhead=2,
                        arrowcolor=C_WHAT_IF,
                        bgcolor="rgba(255, 212, 59, 0.8)",
                        bordercolor=C_WHAT_IF,
                        font=dict(color="black", size=10)
                    )
                
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
                
                # Feature Importance Bar Chart
                if 'feature_importances' in locals() and isinstance(feature_importances, pd.DataFrame) and not feature_importances.empty:
                    st.markdown("### 🧮 Feature Importance Analysis")
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
                        template="plotly_white"
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
                    
                    # CSV export for Top Weeks
                    top_weeks_csv = top_weeks_display.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Top Weeks (CSV)",
                        data=top_weeks_csv,
                        file_name=f"top_weeks_{selected_product.replace(' ', '_')}.csv",
                        mime="text/csv",
                        key="download_top_weeks"
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

        with subtab2:
            st.markdown("### 🚨 Sales Anomalies")
            st.markdown("**Smart Problem Detection: Z-score + IQR + Isolation Forest**")
            st.caption("Hybrid anomaly detection with severity classification and actionable insights")
            
            @st.cache_data(show_spinner="Detecting anomalies...")
            def get_anomalies(_df, _product):
                return detect_sales_anomalies(_df, _product)
            
            anomalies_df = get_anomalies(features_df, selected_product)
            
            # === ANOMALY SUMMARY PANEL ===
            if not anomalies_df.empty:
                # Count by severity
                if "severity" in anomalies_df.columns:
                    severe_count = len(anomalies_df[anomalies_df["severity"].str.contains("severe", case=False, na=False)])
                    moderate_count = len(anomalies_df[anomalies_df["severity"].str.contains("moderate", case=False, na=False)])
                    mild_count = len(anomalies_df[anomalies_df["severity"].str.contains("mild", case=False, na=False)])
                else:
                    # Estimate severity from deviation
                    if "deviation_pct" in anomalies_df.columns:
                        severe_count = len(anomalies_df[anomalies_df["deviation_pct"].abs() > 50])
                        moderate_count = len(anomalies_df[(anomalies_df["deviation_pct"].abs() > 25) & (anomalies_df["deviation_pct"].abs() <= 50)])
                        mild_count = len(anomalies_df[anomalies_df["deviation_pct"].abs() <= 25])
                    else:
                        severe_count = moderate_count = mild_count = 0
                
                total_anomalies = len(anomalies_df)
                
                # Summary panel
                st.markdown("#### 📊 Anomaly Summary")
                summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
                with summary_col1:
                    st.metric("Total Anomalies", total_anomalies, help="Total number of anomalies detected")
                with summary_col2:
                    st.metric("🔴 Severe", severe_count, delta=f"{severe_count} urgent", delta_color="inverse")
                with summary_col3:
                    st.metric("🟡 Moderate", moderate_count)
                with summary_col4:
                    st.metric("🟢 Mild", mild_count)
                
                # Likely causes
                causes = []
                if severe_count > 0:
                    causes.append("supply chain disruption")
                if moderate_count > 0:
                    causes.append("promotion effect")
                if mild_count > 0:
                    causes.append("normal variation")
                
                if causes:
                    st.info(f"🔍 **Likely causes:** {', '.join(set(causes))}")
            else:
                st.success("✅ No anomalies detected — sales patterns are within expected ranges")
            
            # Save anomalies CSV
            try:
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                anom_csv_path = os.path.join(OUTPUT_DIR, "business_sales_anomalies.csv")
                if not anomalies_df.empty:
                    anom_output = anomalies_df.copy()
                    if "product_name" not in anom_output.columns:
                        anom_output["product_name"] = selected_product
                    anom_output.to_csv(anom_csv_path, index=False)
            except Exception:
                pass
            
            if not anomalies_df.empty:
                # Use existing severity counts if already computed
                if "severity" in anomalies_df.columns:
                    severe_count = len(anomalies_df[anomalies_df["severity"] == "severe"])
                    moderate_count = len(anomalies_df[anomalies_df["severity"] == "moderate"])
                    mild_count = len(anomalies_df[anomalies_df["severity"] == "mild"])
                else:
                    severe_count = moderate_count = mild_count = 0
                
                col_a1, col_a2, col_a3, col_a4 = st.columns(4)
                with col_a1:
                    st.metric("🔴 Severe", severe_count, help="Anomalies requiring immediate attention")
                with col_a2:
                    st.metric("🟡 Moderate", moderate_count, help="Anomalies needing monitoring")
                with col_a3:
                    st.metric("🟢 Mild", mild_count, help="Minor deviations from expected")
                with col_a4:
                    st.metric("📊 Total", len(anomalies_df), help="Total anomalies detected")
                
                # Dual Visualization: Timeline with Normal vs Anomalous
                fig_anom = go.Figure()
                
                # Prepare product data for timeline
                product_timeline = product_df[["week_start", "sales_qty"]].copy()
                product_timeline["week_start"] = pd.to_datetime(product_timeline["week_start"])
                product_timeline = product_timeline.sort_values("week_start")
                
                # Separate normal and anomalous points
                anom_dates = pd.to_datetime(anomalies_df["date"], errors='coerce')
                normal_mask = ~product_timeline["week_start"].isin(anom_dates)
                normal_data = product_timeline[normal_mask]
                
                # Normal sales (gray markers)
                fig_anom.add_trace(go.Scatter(
                    x=normal_data["week_start"], 
                    y=normal_data["sales_qty"],
                    name="Normal Sales", 
                    mode="markers",
                    marker=dict(color="#666666", size=5, opacity=0.7)
                ))
                
                # Anomalies by severity
                for severity in ["severe", "moderate", "mild"]:
                    sev_anoms = anomalies_df[anomalies_df["severity"] == severity]
                    if not sev_anoms.empty:
                        sev_dates = pd.to_datetime(sev_anoms["date"], errors='coerce')
                        sev_values = sev_anoms["actual_sales"]
                        valid_mask = sev_dates.notna() & sev_values.notna()
                        
                        color_map = {"severe": "#FF6B6B", "moderate": "#FFD43B", "mild": "#FFA500"}
                        symbol_map = {"severe": "x", "moderate": "star", "mild": "circle"}
                        size_map = {"severe": 14, "moderate": 12, "mild": 10}
                        
                        fig_anom.add_trace(go.Scatter(
                            x=sev_dates[valid_mask],
                            y=sev_values[valid_mask],
                            name=f"{severity.title()} Anomalies",
                            mode="markers",
                            marker=dict(
                                color=color_map[severity],
                                symbol=symbol_map[severity],
                                size=size_map[severity],
                                line=dict(width=2, color="white")
                            )
                        ))
                
                fig_anom.update_layout(
                    title=f"{selected_product} - Anomaly Detection Timeline",
                    xaxis_title="Date",
                    yaxis_title="Sales Quantity",
                    template="plotly_dark",
                    height=500,
                    hovermode='closest'
                )
                st.plotly_chart(fig_anom, use_container_width=True, key="anomaly_timeline_chart")
                
                # Detailed Anomalies Table
                st.markdown("#### 📋 Anomaly Details")
                display_cols = ["date", "actual_sales", "expected_sales", "deviation_pct", "severity", "suggested_action"]
                available_cols = [col for col in display_cols if col in anomalies_df.columns]
                st.dataframe(anomalies_df[available_cols], use_container_width=True, hide_index=True)
                
                # Auto-generated Suggested Business Action
                if not anomalies_df.empty:
                    worst_anom = anomalies_df.loc[anomalies_df["deviation_pct"].abs().idxmax()]
                    anom_date_str = worst_anom['date'].strftime('%Y-%m-%d') if hasattr(worst_anom['date'], 'strftime') else str(worst_anom['date'])
                    action_text = f"⚠️ **Alert:** Unusual **{worst_anom['deviation_pct']:.1f}%** {'drop' if worst_anom['deviation_pct'] < 0 else 'surge'} in {selected_product} sales during {anom_date_str} – {worst_anom.get('suggested_action', 'Investigate supply chain and demand factors')}"
                    
                    if worst_anom["severity"] == "severe":
                        st.error(action_text)
                    elif worst_anom["severity"] == "moderate":
                        st.warning(action_text)
                    else:
                        st.info(action_text)
                
                # Download button
                with st.expander("📥 Export Anomalies", expanded=False):
                    anom_csv = anomalies_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "⬇️ Download Anomalies CSV",
                        data=anom_csv,
                        file_name=f"anomalies_{selected_product.replace(' ', '_')}.csv",
                        mime="text/csv",
                        key="download_anomalies_subtab2"
                    )
            else:
                st.success(f"✅ No anomalies detected for {selected_product}. Sales patterns are normal.")
                st.caption("💡 All sales data points fall within expected statistical ranges.")
        
        with subtab3:
            st.markdown("### 📦 Inventory Alerts")
            st.markdown("**Predictive Stock Optimization**")
            st.caption("ML-powered demand forecasting with stockout risk assessment and auto-reorder suggestions")
            
            @st.cache_data(show_spinner="Analyzing inventory...")
            def get_inventory_alerts(_df):
                return generate_inventory_alerts(_df)
            
            inventory_df = get_inventory_alerts(features_df)
            
            # Save inventory CSV
            try:
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                inv_csv_path = os.path.join(OUTPUT_DIR, "business_inventory_alerts.csv")
                if not inventory_df.empty:
                    inventory_df.to_csv(inv_csv_path, index=False)
            except Exception:
                pass
            
            if not inventory_df.empty:
                # Overall KPIs
                low_stock_count = len(inventory_df[inventory_df["status"] == "🔴 Low Stock"])
                overstock_count = len(inventory_df[inventory_df["status"] == "🟡 Overstock"])
                optimal_count = len(inventory_df[inventory_df["status"] == "🟢 Optimal"])
                
                col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
                with col_kpi1:
                    st.metric("🔴 Low Stock", low_stock_count, help="Products needing immediate reorder")
                with col_kpi2:
                    st.metric("🟡 Overstock", overstock_count, help="Products with excess inventory")
                with col_kpi3:
                    st.metric("🟢 Optimal", optimal_count, help="Products with balanced stock")
                with col_kpi4:
                    st.metric("📊 Total Products", len(inventory_df))
                
                # Product-specific view
                product_inventory = inventory_df[inventory_df["product_name"] == selected_product] if "product_name" in inventory_df.columns else None
                
                if product_inventory is not None and not product_inventory.empty:
                    inv_row = product_inventory.iloc[0]
                    
                    st.markdown(f"#### 📦 {selected_product} Inventory Analysis")
                    col_i1, col_i2, col_i3, col_i4 = st.columns(4)
                    with col_i1:
                        st.metric("📊 Current Stock", f"{inv_row.get('stock', 0):.0f} units", help="Current stock on hand")
                    with col_i2:
                        st.metric("📈 Predicted Demand", f"{inv_row.get('predicted_demand', 0):.0f} units", help="Next week forecast")
                    with col_i3:
                        days_to_stockout = inv_row.get("days_to_stockout", "N/A")
                        st.metric("⏰ Days to Stockout", f"{days_to_stockout}", help="Estimated days until stockout", delta=None if isinstance(days_to_stockout, str) else None)
                    with col_i4:
                        status = inv_row.get("status", "Unknown")
                        st.metric("Status", status)
                    
                    # Auto-Reorder Suggestion
                    if status == "🔴 Low Stock":
                        days_left = inv_row.get("days_to_stockout", "N/A")
                        reorder_qty = inv_row.get("suggested_reorder_qty", 0)
                        if not isinstance(days_left, str):
                            st.error(f"🚨 **URGENT:** {selected_product} may run out in **{days_left} days** — reorder **{reorder_qty:.0f} units** to avoid stock-out.")
                        else:
                            st.error(f"🚨 **URGENT:** {selected_product} requires immediate reorder of **{reorder_qty:.0f} units** to prevent stockout.")
                    elif status == "🟡 Overstock":
                        excess_pct = ((inv_row.get('stock', 0) - inv_row.get('predicted_demand', 0) * 2) / inv_row.get('predicted_demand', 1) * 100) if inv_row.get('predicted_demand', 0) > 0 else 0
                        st.warning(f"⚠️ Overstock detected ({excess_pct:.0f}% above optimal). Consider promotional pricing or reducing future orders.")
                    else:
                        st.success(f"✅ Stock levels optimal for {selected_product}. Current inventory aligns with predicted demand.")
                
                # Top 10 Risk Chart
                st.markdown("#### 📊 Top 10 Stockout Risk Products")
                # Convert days_to_stockout to numeric if it exists
                if "days_to_stockout" in inventory_df.columns:
                    inventory_df["days_to_stockout"] = pd.to_numeric(inventory_df["days_to_stockout"], errors='coerce')
                    sort_col = "days_to_stockout"
                elif "stock" in inventory_df.columns:
                    sort_col = "stock"
                else:
                    sort_col = None
                
                if sort_col:
                    top_risk = inventory_df.nlargest(10, sort_col).copy()
                else:
                    top_risk = inventory_df.head(10).copy()
                
                if not top_risk.empty and "days_to_stockout" in top_risk.columns and "product_name" in top_risk.columns:
                    fig_risk = px.bar(
                        top_risk, 
                        x="product_name", 
                        y="days_to_stockout",
                        title="Days to Stockout (Top 10 Risky Products)",
                        color="days_to_stockout",
                        color_continuous_scale="Reds"
                    )
                    fig_risk.update_layout(template="plotly_dark", height=400, xaxis_tickangle=-45)
                    st.plotly_chart(fig_risk, use_container_width=True, key="stockout_risk_chart")
                
                # Full inventory table
                st.markdown("#### 📋 All Products Inventory Status")
                display_cols = ["product_name", "stock", "predicted_demand", "status", "days_to_stockout", "suggested_reorder_qty", "action_suggestion"]
                available_cols = [col for col in display_cols if col in inventory_df.columns]
                st.dataframe(inventory_df[available_cols], use_container_width=True, hide_index=True)
                
                # Download button
                with st.expander("📥 Export Inventory Alerts", expanded=False):
                    inv_csv = inventory_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "⬇️ Download Inventory Alerts CSV",
                        data=inv_csv,
                        file_name="inventory_alerts.csv",
                        mime="text/csv",
                        key="download_inventory_subtab3"
                    )
            else:
                st.info(f"Inventory analysis not available. Ensure stock_on_hand column exists in dataset.")
        
        with subtab4:
            st.markdown("### 🌦️ Seasonal Insights")
            st.markdown("**Forecast Beyond the Noise: Seasonal Decomposition**")
            st.caption("Trend analysis, monthly seasonality patterns, and correlation heatmaps")
            
            @st.cache_data(show_spinner="Analyzing seasonality...")
            def get_seasonality(_df, _product):
                return analyze_seasonality(_df, _product)
            
            seasonal_data = get_seasonality(features_df, selected_product)
            
            if seasonal_data and len(product_df) >= 52:
                # Yearly Trend Chart
                if "trend" in seasonal_data and seasonal_data["trend"] is not None:
                    st.markdown("#### 📈 Yearly Trend")
                    trend_series = seasonal_data["trend"]
                    if isinstance(trend_series, pd.Series):
                        trend_df = pd.DataFrame({
                            "Date": trend_series.index if hasattr(trend_series.index, 'tolist') else range(len(trend_series)),
                            "Trend": trend_series.values
                        })
                        fig_trend = go.Figure()
                        fig_trend.add_trace(go.Scatter(
                            x=trend_df["Date"],
                            y=trend_df["Trend"].dropna(),
                            name="Trend", 
                            mode="lines",
                            line=dict(color="#00C896", width=2)
                        ))
                        fig_trend.update_layout(
                            title="Long-term Trend (Detrended)",
                            xaxis_title="Time",
                            yaxis_title="Trend Component",
                            template="plotly_dark", 
                            height=350
                        )
                        st.plotly_chart(fig_trend, use_container_width=True, key="seasonal_trend_chart")
                
                # Monthly Seasonality Bar Chart
                if "monthly_pattern" in seasonal_data and seasonal_data["monthly_pattern"]:
                    st.markdown("#### 📅 Monthly Seasonality Pattern")
                    monthly_pattern = seasonal_data["monthly_pattern"]
                    if isinstance(monthly_pattern, dict):
                        monthly_df = pd.DataFrame(list(monthly_pattern.items()), columns=["Month", "Avg Sales"])
                        month_names_full = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                        monthly_df["Month"] = monthly_df["Month"].map(lambda x: month_names_full[x-1] if 1 <= x <= 12 else f"Month {x}")
                        monthly_df = monthly_df.sort_values("Month", key=lambda x: x.map({m: i for i, m in enumerate(month_names_full)}))
                        
                        fig_monthly = px.bar(
                            monthly_df, 
                            x="Month", 
                            y="Avg Sales", 
                            title=f"{selected_product} - Monthly Seasonality Pattern",
                            color="Avg Sales",
                            color_continuous_scale="Blues"
                        )
                        fig_monthly.update_layout(template="plotly_dark", height=400, showlegend=False)
                        st.plotly_chart(fig_monthly, use_container_width=True, key="monthly_seasonality_chart")
                        
                        peak_month_idx = seasonal_data.get("peak_month", 1)
                        low_month_idx = seasonal_data.get("low_month", 6)
                        peak_month_name = month_names_full[peak_month_idx-1] if 1 <= peak_month_idx <= 12 else f"Month {peak_month_idx}"
                        low_month_name = month_names_full[low_month_idx-1] if 1 <= low_month_idx <= 12 else f"Month {low_month_idx}"
                        
                        peak_avg = monthly_df[monthly_df["Month"] == peak_month_name]["Avg Sales"].iloc[0] if not monthly_df.empty else 0
                        low_avg = monthly_df[monthly_df["Month"] == low_month_name]["Avg Sales"].iloc[0] if not monthly_df.empty else 0
                        seasonal_variation = ((peak_avg - low_avg) / low_avg * 100) if low_avg > 0 else 0
                        
                        st.info(f"📅 **Seasonal Pattern:** Peak sales in **{peak_month_name}** ({peak_avg:.0f} units), lowest in **{low_month_name}** ({low_avg:.0f} units). "
                               f"Seasonal variation: **{seasonal_variation:.1f}%**. Bundle strategies recommended for peak months.")
                
                # Correlation Heatmap (if weather/promotion/holiday columns exist)
                correlation_cols = []
                if "sales_qty" in product_df.columns:
                    correlation_cols.append("sales_qty")
                for col in ["promotion", "holiday_flag", "weather_temp", "weather_rainfall"]:
                    if col in product_df.columns:
                        correlation_cols.append(col)
                
                if len(correlation_cols) >= 2:
                    st.markdown("#### 🔥 Correlation Heatmap")
                    corr_data = product_df[correlation_cols].select_dtypes(include=[np.number]).corr()
                    if not corr_data.empty:
                        fig_corr = px.imshow(
                            corr_data,
                            text_auto=".2f",
                            aspect="auto",
                            title="Sales vs External Factors Correlation",
                            color_continuous_scale="RdBu"
                        )
                        fig_corr.update_layout(template="plotly_dark", height=400)
                        st.plotly_chart(fig_corr, use_container_width=True, key="correlation_heatmap_chart")
                
                # Top Seasonal Products (across all products)
                st.markdown("#### 🌟 Top 3 Seasonal Products")
                try:
                    all_seasonal = []
                    for prod in features_df["product_name"].unique()[:10]:  # Sample first 10 for speed
                        try:
                            prod_seasonal = analyze_seasonality(features_df, prod)
                            if prod_seasonal and "monthly_pattern" in prod_seasonal:
                                pattern = prod_seasonal["monthly_pattern"]
                                if isinstance(pattern, dict) and len(pattern) > 0:
                                    peak_val = max(pattern.values())
                                    low_val = min(pattern.values())
                                    variation = ((peak_val - low_val) / low_val * 100) if low_val > 0 else 0
                                    all_seasonal.append({
                                        "product_name": prod,
                                        "seasonal_variation": variation,
                                        "peak_sales": peak_val
                                    })
                        except Exception:
                            continue
                    
                    if all_seasonal:
                        seasonal_df = pd.DataFrame(all_seasonal).sort_values("seasonal_variation", ascending=False).head(3)
                        for idx, row in seasonal_df.iterrows():
                            st.success(f"• **{row['product_name']}**: {row['seasonal_variation']:.1f}% seasonal variation — "
                                     f"Bundle with complementary products in peak season for higher margins.")
                except Exception:
                    pass
                
                # Save seasonal insights
                try:
                    os.makedirs(OUTPUT_DIR, exist_ok=True)
                    seasonal_csv_path = os.path.join(OUTPUT_DIR, "business_seasonal_insights.csv")
                    seasonal_summary = pd.DataFrame([{
                        "product_name": selected_product,
                        "peak_month": peak_month_name if 'peak_month_name' in locals() else "N/A",
                        "low_month": low_month_name if 'low_month_name' in locals() else "N/A",
                        "seasonal_variation_pct": seasonal_variation if 'seasonal_variation' in locals() else 0
                    }])
                    seasonal_summary.to_csv(seasonal_csv_path, index=False)
                except Exception:
                    pass
                
                # Download button
                with st.expander("📥 Export Seasonal Insights", expanded=False):
                    seasonal_text = f"""Seasonal Analysis for {selected_product}
Peak Month: {peak_month_name if 'peak_month_name' in locals() else 'N/A'}
Low Month: {low_month_name if 'low_month_name' in locals() else 'N/A'}
Seasonal Variation: {seasonal_variation:.1f}%
"""
                    st.download_button(
                        "⬇️ Download Seasonal Summary",
                        data=seasonal_text.encode('utf-8'),
                        file_name=f"seasonal_{selected_product.replace(' ', '_')}.txt",
                        mime="text/plain",
                        key="download_seasonal_subtab4"
                    )
            else:
                st.warning("⚠️ Seasonal analysis requires at least 52 weeks of data. Current history: {len(product_df)} weeks.")
                st.caption("💡 Use longer forecast horizon or select a product with more historical data.")
        
        with subtab5:
            st.markdown("### 💰 Pricing Opportunities")
            st.markdown("**Profit Intelligence: Price Elasticity Analysis**")
            st.caption("Elasticity-based pricing optimization with revenue gain projections")
            
            @st.cache_data(show_spinner="Analyzing pricing...")
            def get_pricing_opps(_df):
                return analyze_pricing_opportunities(_df)
            
            pricing_opps = get_pricing_opps(features_df)
            
            # Save pricing opportunities CSV
            try:
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                pricing_csv_path = os.path.join(OUTPUT_DIR, "business_pricing_opportunities.csv")
                if not pricing_opps.empty:
                    pricing_opps.to_csv(pricing_csv_path, index=False)
            except Exception:
                pass
            
            product_pricing = pricing_opps[pricing_opps["product_name"] == selected_product] if not pricing_opps.empty and "product_name" in pricing_opps.columns else None
            
            if product_pricing is not None and not product_pricing.empty:
                opp = product_pricing.iloc[0]
                elasticity = opp.get("elasticity", -1.2)
                current_price = opp.get("current_price", 0)
                revenue_gain = opp.get("revenue_gain", 0)
                
                # KPI Cards
                col_p1, col_p2, col_p3, col_p4 = st.columns(4)
                with col_p1:
                    st.metric("💵 Current Price", f"₹{current_price:.2f}", help="Current selling price")
                with col_p2:
                    elasticity_label = "Elastic" if elasticity < -1 else "Inelastic" if elasticity > -0.5 else "Unit Elastic"
                    st.metric("📊 Elasticity", f"{elasticity:.2f}", help=f"{elasticity_label} - Price sensitivity")
                with col_p3:
                    st.metric("💰 Revenue Gain (±5%)", f"₹{revenue_gain:,.0f}", help="Potential revenue change with ±5% price adjustment")
                with col_p4:
                    opp_status = "💰 Optimize" if abs(revenue_gain) > 1000 else "✅ Stable"
                    st.metric("Status", opp_status)
                
                # Recommendation
                if elasticity < -1.0:
                    st.info(f"💡 **Recommendation:** {selected_product} is **price-elastic** (elasticity: {elasticity:.2f}). Consider price reduction to boost volume and market share.")
                elif elasticity > -0.5:
                    suggested_pct = opp.get('suggested_change_pct', 5)
                    st.success(f"✅ **Opportunity:** {selected_product} is **price-inelastic** (elasticity: {elasticity:.2f}). Raising price by **{suggested_pct:.0f}%** could improve revenue by ₹{revenue_gain:,.0f}.")
                else:
                    st.info(f"➡️ {selected_product} has moderate elasticity ({elasticity:.2f}). Current pricing strategy appears optimal.")
                
                # Price vs Sales Scatter with Regression Fit
                if "price" in product_df.columns and "sales_qty" in product_df.columns:
                    st.markdown("#### 📊 Price vs Sales Relationship")
                    price_sales_df = product_df[["price", "sales_qty"]].dropna()
                    if len(price_sales_df) >= 5:
                        fig_scatter = px.scatter(
                            price_sales_df,
                            x="price",
                            y="sales_qty",
                            trendline="ols",
                            title=f"{selected_product} - Price vs Sales (with Regression Fit)",
                            labels={"price": "Price (₹)", "sales_qty": "Sales Quantity"}
                        )
                        fig_scatter.update_layout(template="plotly_dark", height=400)
                        st.plotly_chart(fig_scatter, use_container_width=True, key="price_sales_scatter_chart")
                        
                        # Add elasticity interpretation
                        if elasticity < -1:
                            st.caption(f"📉 **Elastic Product:** Sales decrease more than proportionally to price increases (elasticity: {elasticity:.2f}). Lower prices drive higher volume.")
                        elif elasticity > -0.5:
                            st.caption(f"📈 **Inelastic Product:** Sales are relatively insensitive to price changes (elasticity: {elasticity:.2f}). Price increases can boost revenue.")
                        else:
                            st.caption(f"📊 **Moderate Elasticity:** Sales respond proportionally to price changes (elasticity: {elasticity:.2f}).")
                
                # Interactive Price Slider Simulation
                st.markdown("#### 🎛️ Price Change Simulation")
                price_change_sim = st.slider("Simulate Price Change (%)", -10, 10, 0, 1, key="price_sim_subtab5")
                if price_change_sim != 0:
                    new_price = current_price * (1 + price_change_sim / 100)
                    # Simple demand model: % change in quantity = elasticity * % change in price
                    quantity_change_pct = elasticity * price_change_sim
                    current_revenue = opp.get("current_revenue", current_price * product_df["sales_qty"].mean() if "sales_qty" in product_df.columns else 0)
                    current_qty = current_revenue / current_price if current_price > 0 else 0
                    new_quantity = current_qty * (1 + quantity_change_pct / 100)
                    new_revenue = new_price * new_quantity
                    revenue_change = new_revenue - current_revenue
                    revenue_change_pct = (revenue_change / current_revenue * 100) if current_revenue > 0 else 0
                    
                    col_sim1, col_sim2, col_sim3 = st.columns(3)
                    with col_sim1:
                        st.metric("💵 New Price", f"₹{new_price:.2f}", delta=f"{price_change_sim:+.1f}%")
                    with col_sim2:
                        st.metric("📦 Projected Qty", f"{new_quantity:.0f}", delta=f"{quantity_change_pct:+.1f}%")
                    with col_sim3:
                        st.metric("💰 Projected Revenue", f"₹{new_revenue:,.0f}", delta=f"{revenue_change_pct:+.1f}%")
                
                # All Products Table
                st.markdown("#### 📋 All Products Pricing Opportunities")
                if not pricing_opps.empty:
                    display_cols = ["product_name", "current_price", "elasticity", "revenue_gain", "suggested_change_pct"]
                    available_cols = [col for col in display_cols if col in pricing_opps.columns]
                    st.dataframe(
                        pricing_opps[available_cols].sort_values("revenue_gain", ascending=False),
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Download button
                    with st.expander("📥 Export Pricing Opportunities", expanded=False):
                        pricing_csv = pricing_opps.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "⬇️ Download Pricing Opportunities CSV",
                            data=pricing_csv,
                            file_name="pricing_opportunities.csv",
                            mime="text/csv",
                            key="download_pricing_subtab5"
                        )
            else:
                st.info(f"Pricing analysis not available for {selected_product}. Ensure price and sales_qty columns exist.")
        
        with subtab6:
            st.markdown("### ⚙️ Dynamic Pricing Engine")
            st.markdown("**AI-Driven Price Optimizer**")
            
            elasticity = calculate_price_elasticity(features_df, selected_product)
            product_row = product_df.iloc[-1]
            current_price = product_row["price"] if "price" in product_row else product_df["price"].mean()
            margin = st.slider("Assumed Margin (%)", 10, 50, 30) / 100
            
            opt_result = optimize_price(current_price, elasticity, margin=margin)
            
            col_d1, col_d2, col_d3 = st.columns(3)
            with col_d1:
                st.metric("💰 Current Price", f"₹{opt_result['current_price']:.2f}")
            with col_d2:
                st.metric("🎯 Optimal Price", f"₹{opt_result['optimal_price']:.2f}", delta=f"{opt_result['price_change_pct']:+.1f}%")
            with col_d3:
                st.metric("📈 Profit Gain", f"₹{opt_result['profit_gain_abs']:.2f}", delta=f"{opt_result['profit_gain_pct']:+.1f}%")
            
            if opt_result["profit_gain_pct"] > 0:
                st.success(f"✅ **Recommendation:** Adjust {selected_product} price from ₹{current_price:.2f} to ₹{opt_result['optimal_price']:.2f} for **{opt_result['profit_gain_pct']:.1f}%** profit improvement.")
            else:
                st.info(f"Current pricing is near optimal for {selected_product}")
            
            # KPI Dashboard
            all_pricing = analyze_pricing_opportunities(features_df)
            if not all_pricing.empty:
                st.markdown("#### 📊 Dynamic Pricing KPIs")
                eligible_count = len(all_pricing[all_pricing["revenue_gain"] > 0])
                avg_gain = all_pricing["revenue_gain"].mean()
                total_projected = all_pricing["revenue_gain"].sum()
                
                col_k1, col_k2, col_k3 = st.columns(3)
                with col_k1:
                    st.metric("📦 Eligible Products", eligible_count)
                with col_k2:
                    st.metric("💰 Avg Revenue Gain", f"₹{avg_gain:,.0f}")
                with col_k3:
                    st.metric("💵 Total Projected Gain", f"₹{total_projected:,.0f}")
        
        with subtab7:
            st.markdown("### 📋 Executive Summary")
            st.markdown("**Auto-Generated Business Intelligence Report**")
            
            # Gather all insights
            anomalies = detect_sales_anomalies(features_df, selected_product)
            inventory = generate_inventory_alerts(features_df)
            pricing = analyze_pricing_opportunities(features_df)
            
            anomalies_count = len(anomalies)
            low_stock_count = len(inventory[inventory["status"] == "🔴 Low Stock"]) if not inventory.empty else 0
            total_revenue_gain = pricing["revenue_gain"].sum() if not pricing.empty else 0
            
            summary_text = generate_executive_summary(
                anomalies_count, low_stock_count, pricing, total_revenue_gain
            )
            
            st.markdown(summary_text)
            
            # Download button
            st.download_button(
                "⬇️ Download Insights as PDF (TXT)",
                summary_text,
                f"retailsense_insights_{selected_product}.txt",
                "text/plain",
                key=f"download_exec_summary_{selected_product}"
            )
    else:
        st.info("👆 Please select a product above to begin analysis")

with tab3:
    st.subheader("🚨 Sales Anomalies")
    
    # Check if forecast has been run
    if "forecast_result_subtab" not in st.session_state:
        st.info("⚠️ Please run the forecast first to generate insights.")
        st.stop()
    
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
            product_select = st.selectbox("Select Product", ["All Products"] + sorted(tab3_df["product_name"].unique().tolist()), key="tab3_product")
            
            if product_select == "All Products":
                display_df = tab3_df
            else:
                display_df = tab3_df[tab3_df["product_name"] == product_select]
            
            # Use business_insights for anomaly detection
            if product_select != "All Products":
                anomalies = detect_sales_anomalies(tab3_df, product_select)
                if not anomalies.empty:
                    st.dataframe(anomalies, use_container_width=True, hide_index=True)
                    
                    # Chart
                    fig = go.Figure()
                    if "week_start" in display_df.columns and "sales_qty" in display_df.columns:
                        display_df = display_df.copy()
                        
                        # Convert anomaly dates to match format
                        anomaly_dates = pd.to_datetime(anomalies["date"], errors='coerce').dt.date if "date" in anomalies.columns else []
                        display_dates = pd.to_datetime(display_df["week_start"], errors='coerce').dt.date
                        
                        normal_mask = ~display_dates.isin(anomaly_dates) if len(anomaly_dates) > 0 else pd.Series([True] * len(display_df))
                        normal_data = display_df[normal_mask]
                        
                        if not normal_data.empty:
                            fig.add_trace(go.Scatter(x=pd.to_datetime(normal_data["week_start"]), 
                                                    y=normal_data["sales_qty"], name="Normal", 
                                                    mode="markers", marker=dict(color="gray", size=5)))
                        if not anomalies.empty and "date" in anomalies.columns and "actual_sales" in anomalies.columns:
                            fig.add_trace(go.Scatter(x=pd.to_datetime(anomalies["date"]), y=anomalies["actual_sales"], 
                                                    name="Anomalies", mode="markers", 
                                                    marker=dict(color="red", symbol="x", size=10)))
                    fig.update_layout(title=f"{product_select} - Anomaly Detection", template="plotly_dark", height=500)
                    st.plotly_chart(fig, use_container_width=True, key=f"tab3_anomaly_chart_{product_select}")
                else:
                    st.success(f"✅ No anomalies detected for {product_select}")
            else:
                st.info("👆 Select a product to see anomaly analysis")
        else:
            st.warning("Required columns (product_name, sales_qty) not found in dataset")
    elif sales_anomalies is not None and not sales_anomalies.empty:
        st.dataframe(sales_anomalies.head(20))
        if {"date", "actual_sales"}.issubset(sales_anomalies.columns):
            fig = px.scatter(sales_anomalies, x="date", y="actual_sales", title="Sales Anomalies Timeline")
            st.plotly_chart(fig, use_container_width=True, key="tab3_anomalies_timeline_chart")
        download_button(sales_anomalies, "⬇️ Download Sales Anomalies", "sales_anomalies.csv", key="download_anomalies_tab3")
    else:
        st.info("💡 Upload data and run pipeline, or use the **Sales Forecasting** tab for detailed analysis.")

with tab4:
    st.subheader("📦 Inventory Alerts")
    
    # Check if forecast has been run
    if "forecast_result_subtab" not in st.session_state:
        st.info("⚠️ Please run the forecast first to generate insights.")
        st.stop()
    
    @st.cache_data(show_spinner="Analyzing inventory...")
    def load_tab4_data():
        if os.path.exists(FEATURES_DATA_PATH):
            try:
                return pd.read_csv(FEATURES_DATA_PATH, low_memory=False, encoding='utf-8')
            except:
                return None
        return None
    
    tab4_df = load_tab4_data()
    
    if tab4_df is not None and not tab4_df.empty and "stock_on_hand" in tab4_df.columns:
        alerts = generate_inventory_alerts(tab4_df)
        if not alerts.empty:
            st.dataframe(alerts, use_container_width=True, hide_index=True)
            
            # KPI cards
            low_stock = len(alerts[alerts["status"] == "🔴 Low Stock"])
            overstock = len(alerts[alerts["status"] == "🟡 Overstock"])
            optimal = len(alerts[alerts["status"] == "🟢 Optimal"])
            
            col1, col2, col3 = st.columns(3)
            col1.metric("🔴 Low Stock", low_stock)
            col2.metric("🟡 Overstock", overstock)
            col3.metric("🟢 Optimal", optimal)
        else:
            st.info("No inventory alerts generated")
    elif inventory_alerts is not None and not inventory_alerts.empty:
        st.dataframe(inventory_alerts.head(20))
        if "urgency" in inventory_alerts.columns:
            fig = px.pie(inventory_alerts, names="urgency", title="Inventory Alert Distribution")
            st.plotly_chart(fig, use_container_width=True, key="tab4_inventory_pie_chart")
        download_button(inventory_alerts, "⬇️ Download Inventory Alerts", "inventory_alerts.csv", key="download_inventory_tab4")
    else:
        st.info("💡 Upload data and run pipeline, or use the **Sales Forecasting** tab for detailed analysis.")

with tab5:
    st.subheader("🎯 Seasonal Insights")
    
    # Check if forecast has been run
    if "forecast_result_subtab" not in st.session_state:
        st.info("⚠️ Please run the forecast first to generate insights.")
        st.stop()
    
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
        product_select = st.selectbox("Select Product", ["All Products"] + sorted(tab5_df["product_name"].unique().tolist()), key="tab5_product")
        
        if product_select != "All Products":
            seasonal_data = analyze_seasonality(tab5_df, product_select)
            if seasonal_data and "monthly_pattern" in seasonal_data:
                monthly_df = pd.DataFrame(list(seasonal_data["monthly_pattern"].items()), columns=["Month", "Avg Sales"])
                month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                monthly_df["Month"] = monthly_df["Month"].map(lambda x: month_names[x-1])
                fig = px.bar(monthly_df, x="Month", y="Avg Sales", title=f"{product_select} - Monthly Seasonality")
                st.plotly_chart(fig, use_container_width=True, key=f"tab5_seasonal_chart_{product_select}")
        else:
            st.info("👆 Select a product to see seasonal patterns")
    elif seasonal_insights is not None and not seasonal_insights.empty:
        st.dataframe(seasonal_insights.head(20))
        if {"category", "variation_percent"}.issubset(seasonal_insights.columns):
            fig = px.bar(seasonal_insights, x="category", y="variation_percent", title="Seasonal Variation by Category")
            st.plotly_chart(fig, use_container_width=True, key="tab5_category_variation_chart")
        download_button(seasonal_insights, "⬇️ Download Seasonal Insights", "seasonal_insights.csv", key="download_seasonal_tab5")
    else:
        st.info("💡 Upload data and run pipeline, or use the **Sales Forecasting** tab for detailed analysis.")

with tab6:
    st.subheader("💰 Pricing Opportunities")
    
    # Check if forecast has been run
    if "forecast_result_subtab" not in st.session_state:
        st.info("⚠️ Please run the forecast first to generate insights.")
        st.stop()
    
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
        opps = analyze_pricing_opportunities(tab6_df)
        if not opps.empty:
            st.dataframe(opps.sort_values("revenue_gain", ascending=False), use_container_width=True, hide_index=True)
            
            # Top opportunities chart
            top_5 = opps.nlargest(5, "revenue_gain")
            fig = px.bar(top_5, x="product_name", y="revenue_gain", title="Top 5 Pricing Opportunities")
            st.plotly_chart(fig, use_container_width=True, key="tab6_top5_pricing_chart")
        else:
            st.info("No pricing opportunities found")
    elif pricing_opps is not None and not pricing_opps.empty:
        st.dataframe(pricing_opps.head(20))
        if "priority_score" in pricing_opps.columns:
            fig = px.histogram(pricing_opps, x="priority_score", nbins=20, title="Pricing Opportunity Priority Scores")
            st.plotly_chart(fig, use_container_width=True, key="tab6_priority_histogram_chart")
        download_button(pricing_opps, "⬇️ Download Pricing Opportunities", "pricing_opportunities.csv", key="download_pricing_tab6")
    else:
        st.info("💡 Upload data and run pipeline, or use the **Sales Forecasting** tab for detailed analysis.")

with tab7:
    st.subheader("⚙️ Dynamic Pricing Engine")
    
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
        product_select = st.selectbox("Select Product", sorted(tab7_df["product_name"].unique().tolist()), key="tab7_product")
        if product_select:
            elasticity = calculate_price_elasticity(tab7_df, product_select)
            product_row = tab7_df[tab7_df["product_name"] == product_select].iloc[-1]
            current_price = product_row["price"] if "price" in product_row else tab7_df["price"].mean()
            margin = st.slider("Assumed Margin (%)", 10, 50, 30, key="tab7_margin") / 100
            
            opt_result = optimize_price(current_price, elasticity, margin=margin)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("💰 Current Price", f"₹{opt_result['current_price']:.2f}")
            col2.metric("🎯 Optimal Price", f"₹{opt_result['optimal_price']:.2f}", delta=f"{opt_result['price_change_pct']:+.1f}%")
            col3.metric("📈 Profit Gain", f"₹{opt_result['profit_gain_abs']:.2f}", delta=f"{opt_result['profit_gain_pct']:+.1f}%")
            
            if opt_result["profit_gain_pct"] > 0:
                st.success(f"✅ Adjust {product_select} price from ₹{current_price:.2f} to ₹{opt_result['optimal_price']:.2f} for {opt_result['profit_gain_pct']:.1f}% profit improvement.")
    else:
        st.info("💡 Upload data and run pipeline, or use the **Sales Forecasting** tab for detailed analysis.")

with tab8:
    st.subheader("🧾 Data Summary (Uploaded CSV)")
    if uploaded_ready:
        df_full = load_csv(UPLOADED_FILE_PATH)
        if df_full is not None and not df_full.empty:
            total_cells = int(df_full.shape[0] * df_full.shape[1])
            missing_cells = int(df_full.isna().sum().sum())
            missing_cells_pct = round((missing_cells / total_cells * 100.0), 2) if total_cells else 0.0
            duplicate_rows = int(df_full.duplicated().sum())
            mem_bytes = int(df_full.memory_usage(deep=True).sum())
            mem_mb = round(mem_bytes / (1024 * 1024), 2)

            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Rows", df_full.shape[0])
            k2.metric("Columns", df_full.shape[1])
            k3.metric("Missing Cells (%)", f"{missing_cells_pct}%")
            k4.metric("Duplicate Rows", duplicate_rows)
            k5.metric("Memory (MB)", mem_mb)

            prof = profile_dataframe(df_full)
            st.dataframe(prof, use_container_width=True, hide_index=True)
            download_button(prof, "⬇️ Download Data Profile", "data_profile.csv", key="download_profile_exec_summary")
        else:
            st.info("No data available. Please upload a CSV from the sidebar.")
    else:
        st.info("No data available. Please upload a CSV from the sidebar.")

st.markdown("---")
st.markdown(
    f"<div style='text-align: center; color: grey;'>"
    f"🛍️ RetailSense Dashboard | AI Retail Analytics • Last Updated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
    f"</div>",
    unsafe_allow_html=True
)

# Note: Legacy tabs are above (tab3-tab7 show info messages pointing to tab2 sub-tabs)

with tab8:
    st.subheader("🧾 Data Summary (Uploaded CSV)")
    if uploaded_ready:
        df_full = load_csv(UPLOADED_FILE_PATH)
        if df_full is not None and not df_full.empty:
            total_cells = int(df_full.shape[0] * df_full.shape[1])
            missing_cells = int(df_full.isna().sum().sum())
            missing_cells_pct = round((missing_cells / total_cells * 100.0), 2) if total_cells else 0.0
            duplicate_rows = int(df_full.duplicated().sum())
            mem_bytes = int(df_full.memory_usage(deep=True).sum())
            mem_mb = round(mem_bytes / (1024 * 1024), 2)

            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Rows", df_full.shape[0])
            k2.metric("Columns", df_full.shape[1])
            k3.metric("Missing Cells (%)", f"{missing_cells_pct}%")
            k4.metric("Duplicate Rows", duplicate_rows)
            k5.metric("Memory (MB)", mem_mb)

            prof = profile_dataframe(df_full)
            st.dataframe(prof, use_container_width=True, hide_index=True)
            download_button(prof, "⬇️ Download Data Profile", "data_profile.csv", key="download_profile_tab8")
        else:
            st.info("No data available. Please upload a CSV from the sidebar.")
    else:
        st.info("No data available. Please upload a CSV from the sidebar.")

st.markdown("---")
st.markdown(
    f"<div style='text-align: center; color: grey;'>"
    f"🛍️ RetailSense Dashboard | AI Retail Analytics • Last Updated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
    f"</div>",
    unsafe_allow_html=True
)