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
from prophet import Prophet
import io
from utils.advanced_forecasting import train_ensemble, train_ensemble_for_app, run_hybrid_forecast, simulate_forecast_with_scenarios
from utils.business_insights import (
    detect_sales_anomalies, generate_inventory_alerts, analyze_seasonality,
    calculate_price_elasticity, analyze_pricing_opportunities, optimize_price,
    generate_executive_summary
)

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
    if os.path.exists(file_path):
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            st.error(f"❌ Error reading {file_path}: {e}")
            return None
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
    df = ts.rename(columns={"date": "ds", "sales_qty": "y"})[["ds", "y"]].dropna()
    m = Prophet(interval_width=0.8, yearly_seasonality=True, weekly_seasonality=True)
    m.fit(df)
    future = m.make_future_dataframe(periods=horizon_days, freq="D")
    forecast = m.predict(future)
    forecast = forecast.rename(columns={"ds": "date", "yhat": "yhat", "yhat_lower": "yhat_lower", "yhat_upper": "yhat_upper"})
    hist_fit = forecast.merge(ts, on="date", how="left")
    return hist_fit, forecast[forecast["date"] > ts["date"].max()][["date", "yhat", "yhat_lower", "yhat_upper"]]

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
            st.subheader("📈 Sales Forecasting")
            st.markdown("*Hybrid Ensemble Forecasting: Prophet + XGBoost + LightGBM*")
            
            # Professional color palette - DARK THEME (Placement-ready)
            C_FORECAST = "#00C896"       # Teal for forecast line (primary)
            C_FORECAST_80_FILL = "rgba(0, 200, 150, 0.35)"  # Teal 80% CI
            C_FORECAST_95_FILL = "rgba(0, 200, 150, 0.15)"  # Teal 95% CI
            C_HISTORY = "#888888"        # Gray for historical data
            C_TARGET = "#d62728"        # Red for target marker
            C_START_LINE = "#666666"     # Grey for separator
            C_WHAT_IF = "#FFD43B"        # Amber for What-If scenarios
            C_BASELINE = C_FORECAST
            C_ERROR = "#FF6B6B"
            C_ANOMALY = "#FF6B6B"        # Red for anomalies
            C_CHANGEPOINT = "#FFD43B"    # Amber for changepoints
            C_FILL80 = C_FORECAST_80_FILL
            C_FILL95 = C_FORECAST_95_FILL
            OUTPUT_DIR = r"F:\RetailSense_Lite\outputs"
            
            # Import new functions
            from utils.advanced_forecasting import run_hybrid_forecast, simulate_forecast_with_scenarios
            
            # Prepare time series data from product_df
            ts = product_df[["week_start", "sales_qty"]].copy()
            ts.columns = ["date", "sales_qty"]
            ts["date"] = pd.to_datetime(ts["date"])
            ts = ts.sort_values("date").reset_index(drop=True)
            
            # Add other columns if available
            if "stock_on_hand" in product_df.columns:
                ts["stock_on_hand"] = product_df["stock_on_hand"].values
            if "price" in product_df.columns:
                ts["price"] = product_df["price"].values
            
            # ==== Input validation: require minimum history ====
            if len(ts) < 8:
                st.warning("⚠️ Not enough history to forecast. Please select a product with ≥ 8 weeks of data.")
                st.stop()
            
            # Overview KPIs
            c1, c2, c3, c4 = st.columns(4)
            avg_sales = float(ts["sales_qty"].tail(8).mean()) if not ts.empty else 0.0
            last_8_weeks = float(ts["sales_qty"].tail(8).sum()) if not ts.empty else 0.0
            stock_level = float(ts["stock_on_hand"].tail(1).iloc[0]) if "stock_on_hand" in ts.columns and not ts["stock_on_hand"].tail(1).isna().all() else np.nan
            c1.metric("Average Sales (last 8w)", f"{avg_sales:,.2f}")
            c2.metric("Last 8 Weeks Sales", f"{last_8_weeks:,.0f}")
            c3.metric("Stock Level", "-" if np.isnan(stock_level) else f"{stock_level:,.0f}")
            c4.metric("Series Length", len(ts))
            
            # === TOP CONTROLS ROW ===
            st.markdown("---")
            st.markdown("### 🎯 Forecast Configuration")
            
            control_col1, control_col2, control_col3, control_col4 = st.columns(4)
            
            with control_col1:
                st.markdown("**📦 Product**")
                st.info(f"{selected_product}")
            
            with control_col2:
                # Model preference selector
                model_type = st.selectbox(
                    "🤖 Model Type",
                    options=["Hybrid", "Prophet", "XGBoost", "LightGBM"],
                    index=0,
                    help="Hybrid uses weighted ensemble of all models (recommended). Individual models for comparison.",
                    key="model_preference_subtab"
                )
            
            with control_col3:
                # Forecast duration quick select
                duration_options = {
                    "6 Months": 26,
                    "1 Year": 52,
                    "3 Years": 156
                }
                duration_label = st.selectbox(
                    "📅 Duration",
                    options=list(duration_options.keys()),
                    index=0,
                    key="duration_select_subtab"
                )
                selected_horizon_weeks = duration_options[duration_label]
                st.session_state["selected_horizon_weeks"] = selected_horizon_weeks
            
            with control_col4:
                # Custom end date
                last_date = pd.to_datetime(ts["date"]).max()
                forecast_start = last_date + pd.DateOffset(weeks=1)
                max_picker_date = last_date + pd.DateOffset(weeks=156)
                default_end = last_date + pd.DateOffset(weeks=selected_horizon_weeks)
                
                custom_end_date = st.date_input(
                    "📅 Custom End Date",
                    value=default_end.date(),
                    min_value=forecast_start.date(),
                    max_value=max_picker_date.date(),
                    help="Override duration with specific end date",
                    key="custom_end_date_subtab"
                )
                if pd.to_datetime(custom_end_date) != default_end:
                    # Recalculate horizon from custom date
                    custom_end_dt = pd.to_datetime(custom_end_date)
                    selected_horizon_weeks = max(1, int((custom_end_dt - last_date).days / 7))
                    st.session_state["selected_horizon_weeks"] = selected_horizon_weeks
            
            # Performance mode
            col_perf1, col_perf2 = st.columns([3, 1])
            with col_perf1:
                fast_mode = st.toggle(
                    "⚡ Fast Mode (Recommended)", 
                    value=True,
                    help="Fast Mode: 10-15s | Slow Mode: Up to 3min with maximum detail",
                    key="forecast_fast_mode_subtab_new"
                )
            with col_perf2:
                run_forecast_btn = st.button("🚀 Run Forecast", type="primary", use_container_width=True, key="run_forecast_btn_new")
            
            st.markdown("---")
            
            # === DYNAMIC FORECAST CONTROLS (Legacy - keeping for compatibility) ===
            with st.expander("⚙️ Advanced Settings", expanded=False):
                # Compute last date in dataset
                last_date = pd.to_datetime(ts["date"]).max()
                st.info(f"📅 Last known data date: *{last_date.strftime('%Y-%m-%d')}*")
                st.caption("Forecast will start from the week after the last known date and extend to your selected future date.")
                
                # === PERFORMANCE MODE TOGGLE ===
                st.divider()
                col_mode1, col_mode2 = st.columns([2, 1])
                with col_mode1:
                    fast_mode = st.toggle(
                        "⚡ Fast Mode (Recommended for Demo)", 
                        value=True,
                        help="Fast Mode: Streamlined features, lightweight models (~10-15s). Toggle OFF for Slow Mode with MAXIMUM detail (up to 3 min).",
                        key="forecast_fast_mode"
                    )
                with col_mode2:
                    if fast_mode:
                        st.success("⚡ Fast (70%)")
                    else:
                        st.error("🔥 Slow (200%+)")
                
                if fast_mode:
                    st.caption("⚡ *Fast Mode (70% Detail):* Streamlined features, lightweight models (10-15 seconds)")
                else:
                    st.caption("🔥 *Slow Mode (200% Detail):* MAXIMUM features, EXTREME model capacity, ALL seasonalities (up to 3 minutes) - Best for review!")
                
                # Manual cache clear button - also clears output files
                if st.button("🗑 Clear Cache & Force Retrain", help="Clear all cached models and retrain from scratch", type="primary", key="clear_forecast_cache"):
                    # Clear session state
                    if "forecast_cache" in st.session_state:
                        del st.session_state["forecast_cache"]
                    if "interactive_fc_ready_subtab" in st.session_state:
                        del st.session_state["interactive_fc_ready_subtab"]
                    if "forecast_result_subtab" in st.session_state:
                        del st.session_state["forecast_result_subtab"]
                    # Clear cached forecast function
                    st.cache_data.clear()
                    # Clear output files
                    try:
                        forecast_csv_path = os.path.join(OUTPUT_DIR, "forecasting_results.csv")
                        if os.path.exists(forecast_csv_path):
                            os.remove(forecast_csv_path)
                        metrics_json_path = os.path.join(OUTPUT_DIR, "forecasting_metrics.json")
                        if os.path.exists(metrics_json_path):
                            os.remove(metrics_json_path)
                    except Exception:
                        pass
                    st.success("✅ Cache and output files cleared! Forecast will retrain on next run.")
                    st.rerun()
                
                st.divider()
                
                # === FUTURE END DATE SELECTOR ===
                st.markdown("#### 🎯 Select Forecast Horizon")
                st.info("💡 *Choose how far into the future you want to forecast* - Model will predict from last data date to your selected date")
                
                # Quick horizon buttons
                st.markdown("*Quick Select:*")
                horizon_btn_col1, horizon_btn_col2, horizon_btn_col3, horizon_btn_col4 = st.columns(4)
                
                selected_horizon_weeks = st.session_state.get("selected_horizon_weeks", 26)
                
                with horizon_btn_col1:
                    if st.button("📅 3 Months", use_container_width=True, key="horizon_3m_subtab"):
                        selected_horizon_weeks = 13
                        st.session_state["selected_horizon_weeks"] = 13
                with horizon_btn_col2:
                    if st.button("📅 6 Months", use_container_width=True, key="horizon_6m_subtab"):
                        selected_horizon_weeks = 26
                        st.session_state["selected_horizon_weeks"] = 26
                with horizon_btn_col3:
                    if st.button("📅 1 Year", use_container_width=True, key="horizon_1y_subtab"):
                        selected_horizon_weeks = 52
                        st.session_state["selected_horizon_weeks"] = 52
                with horizon_btn_col4:
                    if st.button("📅 3 Years", use_container_width=True, key="horizon_3y_subtab"):
                        selected_horizon_weeks = 156
                        st.session_state["selected_horizon_weeks"] = 156
                
                # Calculate forecast date range (from last_date + 1 week to future)
                forecast_start = last_date + pd.DateOffset(weeks=1)
                default_end = last_date + pd.DateOffset(weeks=26)  # Default 6 months ahead
                max_picker_date = last_date + pd.DateOffset(weeks=156)  # Allow up to 3 years
                
                col_date1, col_date2 = st.columns(2)
                with col_date1:
                    target_mode = st.radio("Forecast type:", ["Specific Date", "Specific Week", "Specific Month"], horizontal=True, key="forecast_mode_subtab")
                
                if target_mode == "Specific Date":
                    with col_date2:
                        target_date = st.date_input(
                            "📅 Forecast until (end date)",
                            value=default_end.date(),
                            min_value=forecast_start.date(),
                            max_value=max_picker_date.date(),
                            help=f"Select the end date for the forecast. Predictions will run from {last_date.strftime('%Y-%m-%d')} to this date.",
                            key="target_date_subtab"
                        )
                    # Don't store in session_state with widget key - use different key for processed value
                    target_date_dt = pd.to_datetime(target_date)
                    forecast_days = (target_date_dt - last_date).days
                    forecast_weeks = max(1, int(forecast_days / 7))
                    st.success(f"🔮 Model will forecast from *{last_date.strftime('%b %d, %Y')}* to *{target_date.strftime('%B %d, %Y')}* ({forecast_weeks} weeks ahead)")
                    selected_horizon_weeks = forecast_weeks
                
                elif target_mode == "Specific Week":
                    with col_date2:
                        # Generate future weeks starting from last_date
                        num_weeks = 52  # Show up to 52 weeks
                        week_options = []
                        for i in range(1, num_weeks + 1):
                            week_start = last_date + pd.DateOffset(weeks=i)
                            week_end = week_start + pd.DateOffset(weeks=1)
                            week_label = f"Week {i}: {week_start.strftime('%b %d')} - {week_end.strftime('%b %d, %Y')}"
                            week_options.append((week_label, week_start, week_end, i))
                        
                        if week_options:
                            selected_week_idx = st.selectbox(
                                "📅 Pick a future week",
                                options=range(len(week_options)),
                                format_func=lambda x: week_options[x][0],
                                help="Select a week in the future to see predicted sales",
                                key="selected_week_subtab"
                            )
                            selected_week_data = week_options[selected_week_idx]
                            # Don't store in session_state with widget keys - just use the values
                            selected_horizon_weeks = selected_week_data[3]
                            st.success(f"🔮 Model will predict sales for *{selected_week_data[0]}*")
                
                else:  # Specific Month
                    with col_date2:
                        # Generate future months starting from last_date
                        future_months = pd.date_range(forecast_start, max_picker_date, freq='MS')
                        month_options = [d.strftime("%B %Y") for d in future_months]
                        if not month_options:
                            # Fallback: at least show next month
                            month_options = [(last_date + pd.DateOffset(weeks=4)).strftime("%B %Y")]
                        target_month_idx = st.selectbox("📅 Forecast until month", options=range(len(month_options)), format_func=lambda x: month_options[x], help="Select end month for forecast", key="target_month_subtab")
                        target_month = month_options[target_month_idx]
                        target_month_date = pd.to_datetime(target_month, format="%B %Y")
                        # Don't store in session_state with widget key - just use the value
                        forecast_weeks = max(1, int((target_month_date - last_date).days / 7))
                        selected_horizon_weeks = forecast_weeks
                        st.success(f"🔮 Model will forecast from *{last_date.strftime('%b %Y')}* through *{target_month}*")
                
                # Store horizon in session state and use it as part of cache key
                prev_horizon = st.session_state.get("selected_horizon_weeks", None)
                st.session_state["selected_horizon_weeks"] = selected_horizon_weeks
                
                # If horizon changed, invalidate cached forecast
                if prev_horizon is not None and prev_horizon != selected_horizon_weeks:
                    if "forecast_result_subtab" in st.session_state:
                        del st.session_state["forecast_result_subtab"]
                    if "interactive_fc_ready_subtab" in st.session_state:
                        del st.session_state["interactive_fc_ready_subtab"]
                
                # Cache clear button
                if st.button("🗑 Clear Cache", key="clear_cache_advanced"):
                    st.cache_data.clear()
                    if "forecast_result_subtab" in st.session_state:
                        del st.session_state["forecast_result_subtab"]
                    st.success("✅ Cache cleared!")
                    st.rerun()
            
            # Determine if we should generate forecast
            current_horizon = st.session_state.get("selected_horizon_weeks", 26)
            cached_result = st.session_state.get("forecast_result_subtab")
            cached_horizon = st.session_state.get("cached_horizon_weeks")
            cached_product = st.session_state.get("cached_product_forecast", "")
            cached_model = st.session_state.get("cached_model_type", "hybrid")
            
            # Check if we need to regenerate
            need_regenerate = (run_forecast_btn or 
                             (cached_result is None) or 
                             (cached_horizon != current_horizon) or
                             (cached_product != selected_product) or
                             (cached_model != model_type.lower()))
            
            use_cached = (cached_result is not None and 
                         cached_horizon == current_horizon and 
                         cached_product == selected_product and
                         cached_model == model_type.lower() and
                         not run_forecast_btn)
            
            if need_regenerate:
                st.session_state["interactive_fc_ready_subtab"] = True
                
                # Calculate exact horizon
                horizon_weeks = current_horizon
                custom_end = pd.to_datetime(custom_end_date) if 'custom_end_date' in locals() else None
                
                with st.spinner(f"⚡ Training {model_type} model for {selected_product} ({horizon_weeks} weeks ahead)..."):
                    try:
                        @st.cache_data(ttl=3600, show_spinner=False)
                        def cached_hybrid_forecast(_df_hash, _product, _horizon, _end_date, _model_type, _fast_mode):
                            return run_hybrid_forecast(
                                features_df, 
                                product=_product,
                                horizon_weeks=_horizon,
                                end_date=_end_date,
                                model_type=_model_type.lower(),
                                fast_mode=_fast_mode
                            )
                        
                        # Create cache key
                        df_hash = hash(str(features_df.head(100).values.tobytes()))
                        res_dict = cached_hybrid_forecast(
                            df_hash, selected_product, horizon_weeks, custom_end, model_type, fast_mode
                        )
                        
                        st.session_state["forecast_result_subtab"] = res_dict
                        st.session_state["cached_horizon_weeks"] = horizon_weeks
                        st.session_state["cached_product_forecast"] = selected_product
                        st.session_state["cached_model_type"] = model_type.lower()
                        st.success(f"✅ Forecast generated using {model_type} model!")
                    except Exception as e:
                        st.error(f"❌ Forecast generation failed: {str(e)}")
                        with st.expander("🔍 View Error Details"):
                            import traceback
                            st.code(traceback.format_exc())
                        st.warning("💡 Try Fast Mode or select a product with more history (≥30 weeks recommended)")
                        st.stop()
            
            # Display forecast results (either newly generated or cached)
            if "forecast_result_subtab" in st.session_state:
                res_dict = st.session_state["forecast_result_subtab"]
                
                # Extract DataFrames and metrics from EnsembleResult object
                forecast_df = res_dict.forecast_df if hasattr(res_dict, 'forecast_df') else pd.DataFrame()
                if isinstance(forecast_df, pd.DataFrame) and forecast_df.empty:
                    # No fallback needed with EnsembleResult object
                
                history_df = res_dict.history_df if hasattr(res_dict, 'history_df') else pd.DataFrame()
                if isinstance(history_df, pd.DataFrame) and history_df.empty:
                    # No fallback needed with EnsembleResult object
                
                metrics = res_dict.metrics if hasattr(res_dict, 'metrics') else {}
                insights = res_dict.details if hasattr(res_dict, 'details') else {}
                feature_importances = res_dict.feature_importances if hasattr(res_dict, 'feature_importances') else {}
                prophet_components = res_dict.prophet_components if hasattr(res_dict, 'prophet_components') else {}
                
                if forecast_df.empty or history_df.empty:
                    st.error("❌ Forecast result is empty. Please retry.")
                    st.stop()
                
                # Ensure date columns are datetime
                if "date" in forecast_df.columns:
                    forecast_df["date"] = pd.to_datetime(forecast_df["date"])
                if "date" in history_df.columns:
                    history_df["date"] = pd.to_datetime(history_df["date"])
                
                # Calculate comprehensive KPIs
                month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                
                # Ensure date columns are datetime
                if "date" in forecast_df.columns:
                    forecast_df["date"] = pd.to_datetime(forecast_df["date"])
                if "date" in history_df.columns:
                    history_df["date"] = pd.to_datetime(history_df["date"])
                
                # Growth percentage (next 4 weeks vs last 4 weeks)
                if "yhat" in forecast_df.columns and len(forecast_df) >= 4 and "sales_qty" in history_df.columns and len(history_df) >= 4:
                    forecast_next_4 = forecast_df.head(4)["yhat"].mean()
                    history_last_4 = history_df["sales_qty"].tail(4).mean()
                    growth_pct = ((forecast_next_4 - history_last_4) / (history_last_4 + 1e-6)) * 100
                else:
                    growth_pct = 0.0
                
                # Next quarter (13 weeks)
                if "yhat" in forecast_df.columns and len(forecast_df) >= 13:
                    next_q_forecast = float(forecast_df.head(13)["yhat"].sum())
                elif "yhat" in forecast_df.columns:
                    next_q_forecast = float(forecast_df["yhat"].sum())
                else:
                    next_q_forecast = 0.0
                
                # Peak month
                if len(forecast_df) > 0 and "date" in forecast_df.columns and "yhat" in forecast_df.columns:
                    forecast_by_month = forecast_df.groupby(forecast_df["date"].dt.month)["yhat"].mean()
                    peak_month_idx = int(forecast_by_month.idxmax()) if not forecast_by_month.empty else 1
                else:
                    peak_month_idx = 1
                
                # Model metrics
                r2_score_val = metrics.get("ensemble_r2", metrics.get("r2", 0))
                rmse_val = metrics.get("ensemble_rmse", metrics.get("rmse", 0))
                mae_val = metrics.get("ensemble_mae", metrics.get("mae", 0))
                
                # Additional calculations
                next_week_total = float(forecast_df.head(1)["yhat"].sum()) if len(forecast_df) > 0 and "yhat" in forecast_df.columns else 0.0
                
                # Min month
                if len(forecast_df) > 0 and "date" in forecast_df.columns and "yhat" in forecast_df.columns:
                    forecast_by_month = forecast_df.groupby(forecast_df["date"].dt.month)["yhat"].mean()
                    min_month_idx = int(forecast_by_month.idxmin()) if not forecast_by_month.empty else 1
                else:
                    min_month_idx = 1
                
                # Stockout risk
                stockout_risk = np.nan
                if "stock_on_hand" in history_df.columns and len(history_df) > 0:
                    avg_stock = float(history_df["stock_on_hand"].tail(4).mean()) if not history_df["stock_on_hand"].tail(4).isna().all() else np.nan
                    if not np.isnan(avg_stock) and avg_stock > 0:
                        stockout_risk = float(np.clip((next_week_total - avg_stock) / max(avg_stock, 1), 0, 1))
                
                stockout_display = "Low" if np.isnan(stockout_risk) or stockout_risk < 0.3 else "Med" if stockout_risk < 0.7 else "High"
                
                display_horizon = st.session_state.get("selected_horizon_weeks", 26)
                
                # === BUSINESS KPI CARDS ===
                st.markdown("---")
                st.markdown("### 📊 Forecast Metrics")
                
                kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                with kpi1:
                    st.metric(
                        "📈 Predicted Growth %", 
                        f"{growth_pct:+.1f}%", 
                        delta=f"{growth_pct:+.1f}%",
                        help="Expected growth rate comparing next 4 weeks vs last 4 weeks historical"
                    )
                with kpi2:
                    st.metric(
                        "🧮 RMSE / Accuracy", 
                        f"{rmse_val:.2f}", 
                        help="Root Mean Square Error — measures forecast accuracy. Lower is better."
                    )
                with kpi3:
                    st.metric(
                        "💰 Next Quarter Expected Sales", 
                        f"{next_q_forecast:,.0f}", 
                        help="Total forecasted sales for next 13 weeks (one quarter)"
                    )
                with kpi4:
                    st.metric(
                        "🕓 Predicted Peak Month", 
                        month_names[peak_month_idx-1], 
                        help="Month with highest predicted sales volume"
                    )
                
                # === INTERACTIVE CHART CONTROLS ===
                st.markdown("### 📊 Interactive Forecast Chart")
                
                # Control panel for chart features
                chart_col1, chart_col2, chart_col3 = st.columns(3)
                with chart_col1:
                    show_95_ci = st.checkbox("Show 95% Confidence Interval", value=True, help="Wider uncertainty band")
                    show_80_ci = st.checkbox("Show 80% Confidence Interval", value=True, help="Narrower uncertainty band")
                with chart_col2:
                    show_seasonal = st.checkbox("Show Seasonal Component", value=False, help="Display Prophet seasonal decomposition")
                    highlight_peaks = st.checkbox("Highlight Peak Months", value=True, help="Mark months with highest sales")
                with chart_col3:
                    show_trend = st.checkbox("Show Trend Line", value=True, help="Display underlying trend")
                    show_anomalies = st.checkbox("Show Anomalies", value=True, help="Mark detected anomalies in historical data")
                
                # Interactive Chart with anomalies, changepoints, and dual CI bands
                fig = go.Figure()
                
                # Historical data (gray as per requirements)
                fig.add_trace(go.Scatter(
                        x=history_df["date"], y=history_df["sales_qty"],
                        name="Historical Sales", mode="lines+markers",
                        line=dict(color="#888888", width=2),  # Gray for historical
                        marker=dict(size=6, color="#888888"),
                        hovertemplate="Date: %{x}<br>Sales: %{y:,.0f}<extra></extra>"
                ))
                
                # Detect and plot anomalies on historical data (if enabled)
                if show_anomalies:
                    try:
                        anomalies_df = detect_sales_anomalies(features_df, selected_product)
                        if not anomalies_df.empty and "date" in anomalies_df.columns and "actual_sales" in anomalies_df.columns:
                            anom_dates = pd.to_datetime(anomalies_df["date"], errors='coerce')
                            anom_values = anomalies_df["actual_sales"]
                            valid_mask = anom_dates.notna() & anom_values.notna()
                            if valid_mask.sum() > 0:
                                # Color code by severity if available
                                if "severity" in anomalies_df.columns:
                                    for severity in anomalies_df["severity"].unique():
                                        sev_mask = anomalies_df["severity"] == severity
                                        anom_sev_dates = anom_dates[sev_mask & valid_mask]
                                        anom_sev_values = anom_values[sev_mask & valid_mask]
                                        if len(anom_sev_dates) > 0:
                                            color_map = {"severe": "#FF0000", "moderate": "#FFA500", "mild": "#FFD700"}
                                            color = color_map.get(severity.lower(), C_ANOMALY)
                                            fig.add_trace(go.Scatter(
                                                x=anom_sev_dates, 
                                                y=anom_sev_values,
                                                mode="markers", 
                                                name=f"Anomalies ({severity})", 
                                                marker=dict(color=color, symbol="x", size=14, line=dict(width=2, color="white"))
                                            ))
                                else:
                                    fig.add_trace(go.Scatter(
                                        x=anom_dates[valid_mask], 
                                        y=anom_values[valid_mask],
                                        mode="markers", 
                                        name="Anomalies", 
                                        marker=dict(color=C_ANOMALY, symbol="x", size=12, line=dict(width=2, color="white")),
                                        hovertemplate="<b>ANOMALY</b><br>Date: %{x}<br>Sales: %{y:,.0f}<extra></extra>"
                                    ))
                    except Exception:
                        pass  # Silently fail if anomaly detection errors
                
                # Forecast mean line (with smoothing for better visualization)
                fig.add_trace(go.Scatter(
                        x=forecast_df["date"], y=forecast_df["yhat"],
                        name="Forecast Mean", mode="lines",
                        line=dict(color="#00C896", width=3, smoothing=1.3),
                        hovertemplate="Date: %{x}<br>Forecast: %{y:,.0f}<extra></extra>"
                ))
                
                # Calculate confidence intervals
                has_95 = "yhat_upper_95" in forecast_df.columns and "yhat_lower_95" in forecast_df.columns
                
                if has_95:
                    yhat_upper95 = forecast_df["yhat_upper_95"].values
                    yhat_lower95 = forecast_df["yhat_lower_95"].values
                    yhat_upper80 = forecast_df["yhat_upper"].values
                    yhat_lower80 = forecast_df["yhat_lower"].values
                elif "yhat_upper" in forecast_df.columns and "yhat_lower" in forecast_df.columns:
                    yhat_mean = forecast_df["yhat"].values
                    yhat_upper95 = forecast_df["yhat_upper"].values
                    yhat_lower95 = forecast_df["yhat_lower"].values
                    # Approximate 80% CI as narrower band
                    ci_range_95 = (yhat_upper95 - yhat_lower95) / 2
                    ci_range_80 = ci_range_95 * (1.28 / 1.96)
                    yhat_upper80 = yhat_mean + ci_range_80
                    yhat_lower80 = yhat_mean - ci_range_80
                else:
                    yhat_upper95 = yhat_upper80 = yhat_lower95 = yhat_lower80 = None
                
                # 95% CI band (outer, lighter) - if enabled
                if show_95_ci and yhat_upper95 is not None:
                    fig.add_trace(go.Scatter(
                        x=pd.concat([forecast_df["date"], forecast_df["date"][::-1]]),
                        y=pd.concat([pd.Series(yhat_upper95), pd.Series(yhat_lower95)[::-1]]),
                        fill='toself', fillcolor='rgba(0, 200, 150, 0.15)', line=dict(color='rgba(255,255,255,0)'),
                        hoverinfo="skip", showlegend=True, name='95% Confidence Interval',
                        hovertemplate="95% CI: %{y:,.0f}<extra></extra>"
                    ))
                
                # 80% CI band (inner, darker) - if enabled
                if show_80_ci and yhat_upper80 is not None:
                    fig.add_trace(go.Scatter(
                        x=pd.concat([forecast_df["date"], forecast_df["date"][::-1]]),
                        y=pd.concat([pd.Series(yhat_upper80), pd.Series(yhat_lower80)[::-1]]),
                        fill='toself', fillcolor='rgba(0, 200, 150, 0.35)', line=dict(color='rgba(255,255,255,0)'),
                        hoverinfo="skip", showlegend=True, name='80% Confidence Interval',
                        hovertemplate="80% CI: %{y:,.0f}<extra></extra>"
                    ))
                
                # Highlight peak months if enabled
                if highlight_peaks and len(forecast_df) > 0:
                    forecast_by_month = forecast_df.groupby(forecast_df["date"].dt.month)["yhat"].mean()
                    if not forecast_by_month.empty:
                        peak_month = forecast_by_month.idxmax()
                        peak_dates = forecast_df[forecast_df["date"].dt.month == peak_month]["date"]
                        peak_values = forecast_df[forecast_df["date"].dt.month == peak_month]["yhat"]
                        if len(peak_dates) > 0:
                            fig.add_trace(go.Scatter(
                                x=peak_dates, y=peak_values,
                                mode="markers", name="Peak Months",
                                marker=dict(color="#FFD43B", symbol="star", size=10, line=dict(width=1, color="orange")),
                                hovertemplate="<b>PEAK MONTH</b><br>Date: %{x}<br>Forecast: %{y:,.0f}<extra></extra>"
                            ))
                
                # Add changepoints from Prophet if available
                changepoints = prophet_components.get("changepoints", [])
                if changepoints and len(changepoints) > 0:
                    try:
                        for cp in changepoints[:5]:  # Limit to first 5 changepoints
                            if isinstance(cp, (pd.Timestamp, str)):
                                cp_dt = pd.to_datetime(cp, errors='coerce')
                                if cp_dt.notna().any() if isinstance(cp_dt, pd.Series) else not pd.isna(cp_dt):
                                    cp_val = cp_dt if isinstance(cp_dt, pd.Timestamp) else cp_dt.iloc[0] if isinstance(cp_dt, pd.Series) else None
                                    if cp_val and last_date <= cp_val <= forecast_df["date"].max():
                                        fig.add_shape(
                                            type="line",
                                            x0=cp_val, x1=cp_val,
                                            y0=0, y1=1, yref="paper",
                                            line=dict(color=C_CHANGEPOINT, width=1.5, dash="dot"),
                                            opacity=0.6
                                        )
                    except Exception:
                        pass  # Silently skip changepoint rendering if it fails
                
                # Add vertical line separating history and forecast
                last_date_dt = pd.Timestamp(last_date).to_pydatetime()
                fig.add_shape(
                    type="line",
                    x0=last_date_dt, x1=last_date_dt,
                    y0=0, y1=1,
                    yref="paper",
                    line=dict(color=C_START_LINE, width=2, dash="dash"),
                    opacity=0.7
                )
                fig.add_annotation(
                    x=last_date_dt, y=0.95, yref="paper",
                    text="Forecast Start",
                    showarrow=False, xshift=10,
                    bgcolor="rgba(100,100,100,0.8)",
                    bordercolor="gray",
                    font=dict(color="white", size=10)
                )
                
                # Get horizon for title
                title_horizon = st.session_state.get("selected_horizon_weeks", 26)
                
                # Add trend line if enabled
                if show_trend and not history_df.empty:
                    # Simple trend from historical data
                    trend_dates = history_df["date"]
                    trend_values = history_df["sales_qty"].rolling(window=4, center=True).mean()
                    fig.add_trace(go.Scatter(
                        x=trend_dates, y=trend_values,
                        name="Trend", mode="lines",
                        line=dict(color="#888888", width=1, dash="dot"),
                        hovertemplate="Trend: %{y:,.0f}<extra></extra>"
                    ))
                
                # Add seasonal component if available and enabled
                if show_seasonal and prophet_components:
                    try:
                        if isinstance(prophet_components, dict):
                            # Try to extract seasonal data
                            pass  # Will handle in Prophet components section
                        elif isinstance(prophet_components, pd.DataFrame):
                            if "yearly" in prophet_components.columns:
                                season_df = prophet_components[["date", "yearly"]].copy()
                                season_df["date"] = pd.to_datetime(season_df["date"])
                                season_df = season_df[season_df["date"] <= history_df["date"].max()]
                                if len(season_df) > 0:
                                    # Normalize seasonal component to visible scale
                                    seasonal_normalized = season_df["yearly"] * (history_df["sales_qty"].mean() / season_df["yearly"].abs().max()) * 0.3
                                    fig.add_trace(go.Scatter(
                                        x=season_df["date"], y=seasonal_normalized,
                                        name="Seasonal Component", mode="lines",
                                        line=dict(color="#FFD43B", width=2, dash="dash"),
                                        hovertemplate="Seasonal: %{y:,.0f}<extra></extra>"
                                    ))
                    except Exception:
                        pass
                
                fig.update_layout(
                    title=f"{selected_product} - Sales Forecast ({title_horizon} weeks ahead)",
                    xaxis_title="Date",
                    yaxis_title="Sales Quantity",
                    template="plotly_dark",
                    height=650,
                    hovermode='x unified',
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    xaxis=dict(rangeslider=dict(visible=True, thickness=0.1)),
                    yaxis=dict(gridcolor='rgba(128, 128, 128, 0.2)')
                )
                st.plotly_chart(fig, use_container_width=True, key="main_forecast_chart")
                
                # === WHAT-IF SCENARIO SIMULATION ===
                st.markdown("---")
                st.markdown("### 💡 What-If Scenario Simulation")
                st.caption("Simulate different business scenarios to see impact on forecasted sales")
                
                whatif_col1, whatif_col2, whatif_col3 = st.columns(3)
                
                with whatif_col1:
                    price_delta = st.slider(
                        "💰 Price Change (%)",
                        min_value=-20,
                        max_value=20,
                        value=0,
                        step=1,
                        help="Adjust price: -20% to +20%",
                        key="whatif_price_slider"
                    )
                
                with whatif_col2:
                    promotion_flag = st.checkbox(
                        "🎯 Active Promotion",
                        value=False,
                        help="Simulate promotion effect (+15-25% boost)",
                        key="whatif_promotion"
                    )
                
                with whatif_col3:
                    holiday_flag = st.checkbox(
                        "🎉 Holiday Period",
                        value=False,
                        help="Simulate holiday effect (+20-35% boost)",
                        key="whatif_holiday"
                    )
                
                # Calculate elasticity for product
                try:
                    from utils.business_insights import calculate_price_elasticity
                    elasticity = calculate_price_elasticity(features_df, selected_product)
                except:
                    elasticity = -1.2  # Default
                
                # Simulate forecast
                if price_delta != 0 or promotion_flag or holiday_flag:
                    sim_forecast = simulate_forecast_with_scenarios(
                        forecast_df, 
                        price_delta=price_delta,
                        promotion_flag=promotion_flag,
                        holiday_flag=holiday_flag,
                        elasticity=elasticity
                    )
                    
                    # Calculate impact
                    base_total = forecast_df["yhat"].sum()
                    sim_total = sim_forecast["yhat_simulated"].sum()
                    impact_pct = ((sim_total - base_total) / base_total) * 100
                    
                    # What-If Chart
                    fig_whatif = go.Figure()
                    
                    # Base forecast
                    fig_whatif.add_trace(go.Scatter(
                        x=forecast_df["date"],
                        y=forecast_df["yhat"],
                        name="Base Forecast",
                        mode="lines",
                        line=dict(color=C_FORECAST, width=3),
                        hovertemplate="Base: %{y:,.0f}<extra></extra>"
                    ))
                    
                    # Simulated forecast
                    fig_whatif.add_trace(go.Scatter(
                        x=sim_forecast["date"],
                        y=sim_forecast["yhat_simulated"],
                        name="Simulated Forecast",
                        mode="lines",
                        line=dict(color=C_WHAT_IF, width=3, dash="dash"),
                        hovertemplate="Simulated: %{y:,.0f}<extra></extra>"
                    ))
                    
                    # Confidence intervals for simulated
                    if "yhat_upper_simulated" in sim_forecast.columns:
                        fig_whatif.add_trace(go.Scatter(
                            x=pd.concat([sim_forecast["date"], sim_forecast["date"][::-1]]),
                            y=pd.concat([pd.Series(sim_forecast["yhat_upper_simulated"]), pd.Series(sim_forecast["yhat_lower_simulated"])[::-1]]),
                            fill='toself',
                            fillcolor='rgba(255, 212, 59, 0.2)',
                            line=dict(color='rgba(255,255,255,0)'),
                            name='Simulated CI',
                            showlegend=False,
                            hoverinfo="skip"
                        ))
                    
                    fig_whatif.update_layout(
                        title="What-If Scenario Comparison",
                        xaxis_title="Date",
                        yaxis_title="Sales Quantity",
                        template="plotly_dark",
                        height=500,
                        hovermode='x unified',
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    
                    st.plotly_chart(fig_whatif, use_container_width=True, key="whatif_chart")
                    
                    # Impact summary
                    scenario_desc = []
                    if price_delta != 0:
                        scenario_desc.append(f"{price_delta:+.0f}% price change")
                    if promotion_flag:
                        scenario_desc.append("promotion active")
                    if holiday_flag:
                        scenario_desc.append("holiday period")
                    
                    scenario_text = " + ".join(scenario_desc) if scenario_desc else "baseline"
                    st.success(
                        f"📊 **Scenario:** {scenario_text} → "
                        f"Projected sales {'increase' if impact_pct > 0 else 'decrease'} by **{abs(impact_pct):.1f}%** "
                        f"over next {min(60, len(forecast_df)*7)} days. "
                        f"Total impact: {sim_total:,.0f} vs {base_total:,.0f} units."
                    )
                else:
                    st.info("💡 Adjust sliders above to simulate different business scenarios")
                
                # Save forecast CSV
                try:
                    os.makedirs(OUTPUT_DIR, exist_ok=True)
                    forecast_csv_path = os.path.join(OUTPUT_DIR, "forecasting_results.csv")
                    forecast_output = forecast_df.copy()
                    if "product_name" not in forecast_output.columns:
                        forecast_output["product_name"] = selected_product
                    forecast_output.to_csv(forecast_csv_path, index=False)
                except Exception as e:
                    st.warning(f"Could not save forecast CSV: {e}")
                
                # === BUSINESS INSIGHTS SECTION ===
                st.markdown("---")
                st.markdown("### 🧠 Business Intelligence Insights")
                
                # Get insights from forecast result or generate
                narrative = insights.get("narrative", "") if insights else ""
                top_drivers = insights.get("top_drivers", []) if insights else []
                recommendations = insights.get("recommendations", []) if insights else []
                
                # Generate narrative if not provided
                if not narrative:
                    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                    if growth_pct > 5:
                        narrative = f"{selected_product} sales expected to rise {growth_pct:.1f}% in the next month, driven by seasonal demand. Consider bundling with complementary products to leverage seasonality."
                    elif growth_pct < -5:
                        narrative = f"{selected_product} sales expected to decline {abs(growth_pct):.1f}% — review market conditions and consider promotional strategies."
                    else:
                        narrative = f"{selected_product} sales expected to remain stable with {abs(growth_pct):.1f}% variation. Peak sales predicted for {month_names[peak_month_idx-1]}."
                
                # Display narrative
                st.markdown(f"**📝 Executive Summary:** {narrative}")
                
                # Top Drivers section
                if top_drivers or feature_importances:
                    st.markdown("#### 🎯 Top Forecast Drivers")
                    drivers_col1, drivers_col2 = st.columns(2)
                    
                    with drivers_col1:
                        if top_drivers:
                            for i, driver in enumerate(top_drivers[:3], 1):
                                st.markdown(f"{i}. {driver}")
                        else:
                            # Fallback to feature importances
                            if feature_importances and isinstance(feature_importances, dict):
                                top_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)[:3]
                                for i, (feat, imp) in enumerate(top_features, 1):
                                    st.markdown(f"{i}. {feat} (importance: {imp:.2f})")
                    
                    with drivers_col2:
                        if len(top_drivers) > 3:
                            for i, driver in enumerate(top_drivers[3:], 4):
                                st.markdown(f"{i}. {driver}")
                        elif feature_importances and isinstance(feature_importances, dict) and len(feature_importances) > 3:
                            top_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)[3:6]
                            for i, (feat, imp) in enumerate(top_features, 4):
                                st.markdown(f"{i}. {feat} (importance: {imp:.2f})")
                
                # Recommendations section
                if recommendations:
                    st.markdown("#### 💡 Recommendations")
                    for rec in recommendations:
                        st.markdown(f"- {rec}")
                else:
                    # Generate default recommendations
                    st.markdown("#### 💡 Recommendations")
                    st.markdown(f"- Increase stock before {month_names[peak_month_idx-1]} to prepare for peak demand")
                    st.markdown("- Leverage seasonal promotions during peak months")
                    st.markdown("- Monitor price elasticity for optimization opportunities")
                
                # === TOP 5 FORECASTED WEEKS TABLE ===
                st.markdown("---")
                st.markdown("#### 📋 Top 5 Forecasted Weeks")
                
                if len(forecast_df) >= 5 and "yhat" in forecast_df.columns:
                    top_weeks = forecast_df.nlargest(5, "yhat")[["date", "yhat", "yhat_lower", "yhat_upper"]].copy()
                    top_weeks["date"] = pd.to_datetime(top_weeks["date"]).dt.strftime("%Y-%m-%d")
                    top_weeks["CI_Range"] = (top_weeks["yhat_upper"] - top_weeks["yhat_lower"]).round(0)
                    
                    # Calculate % growth
                    if len(history_df) > 0 and "sales_qty" in history_df.columns:
                        baseline = history_df["sales_qty"].tail(4).mean()
                        top_weeks["Growth_%"] = ((top_weeks["yhat"] - baseline) / baseline * 100).round(1)
                    else:
                        top_weeks["Growth_%"] = 0.0
                    
                    # Rename columns for display
                    top_weeks_display = top_weeks.rename(columns={
                        "date": "Date",
                        "yhat": "Forecasted Sales",
                        "CI_Range": "CI Range",
                        "Growth_%": "% Growth"
                    })
                    
                    st.dataframe(
                        top_weeks_display[["Date", "Forecasted Sales", "CI Range", "% Growth"]],
                        use_container_width=True,
                        hide_index=True
                    )
                
                # Download buttons
                st.markdown("---")
                st.markdown("#### 📥 Export Forecast")
                
                col_dl1, col_dl2, col_dl3 = st.columns(3)
                with col_dl1:
                    # CSV download
                    csv_data = forecast_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Forecast CSV",
                        data=csv_data,
                        file_name=f"forecast_{selected_product.replace(' ', '_')}_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        key="download_forecast_csv_new"
                    )
                with col_dl2:
                    # Metrics JSON
                    metrics_json = json.dumps(metrics, indent=2)
                    st.download_button(
                        label="📊 Download Metrics JSON",
                        data=metrics_json,
                        file_name=f"metrics_{selected_product.replace(' ', '_')}.json",
                        mime="application/json",
                        key="download_metrics_json_new"
                    )
                with col_dl3:
                    # PDF report placeholder
                    st.caption("📄 PDF Report")
                    st.caption("(Requires reportlab package)")
                    # Note: Full PDF generation would require reportlab or similar library
                
                # Create tabs for Forecast Overview, Model Insights, and Business Implications
                tab_overview, tab_insights, tab_business = st.tabs([
                    "📊 Forecast Overview",
                    "🔍 Model Insights",
                    "📈 Business Implications"
                ])
                
                with tab_overview:
                    st.info(f"💡 {narrative}")
                    st.markdown(f"### 📊 Forecast Overview for {selected_product}")
                    st.markdown(f"**Forecast Horizon:** {display_horizon} weeks ({display_horizon * 7} days)")
                    st.markdown(f"**Last Known Date:** {last_date.strftime('%Y-%m-%d')}")
                    if not forecast_df.empty and "date" in forecast_df.columns:
                        forecast_end = forecast_df["date"].max()
                        st.markdown(f"**Forecast End Date:** {forecast_end.strftime('%Y-%m-%d')}")
                    
                    # Display forecast chart here as well
                    st.plotly_chart(fig, use_container_width=True, key="overview_forecast_chart")
                
                with tab_insights:
                    st.markdown("### 🔍 Model Insights")
                    
                    # Model Performance Metrics
                    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                    with col_m1:
                        st.metric("R² Score", f"{r2_score_val:.3f}", help="Coefficient of determination")
                    with col_m2:
                        st.metric("RMSE", f"{rmse_val:.2f}", help="Root Mean Squared Error")
                    with col_m3:
                        st.metric("MAE", f"{mae_val:.2f}", help="Mean Absolute Error")
                    with col_m4:
                        smape_val = metrics.get("ensemble_smape", metrics.get("smape", 0))
                        st.metric("sMAPE", f"{smape_val:.2f}%", help="Symmetric Mean Absolute Percentage Error")
                    
                    # Explain Forecast Drivers Section
                    st.markdown("#### Feature Importances")
                    with st.expander("🔍 Explain Forecast Drivers", expanded=True):
                        if feature_importances and len(feature_importances) > 0:
                            # Convert dict to DataFrame for display
                            if isinstance(feature_importances, dict):
                                fi_df = pd.DataFrame(list(feature_importances.items()), columns=["feature", "importance"])
                                fi_df = fi_df.sort_values("importance", ascending=False).head(10)
                            else:
                                fi_df = feature_importances.head(10) if hasattr(feature_importances, 'head') else pd.DataFrame()
                            
                            if not fi_df.empty:
                                st.markdown("#### 🎯 Top 10 Forecast Drivers")
                                
                                # Try SHAP if available
                                shap_used = False
                                if SHAP_AVAILABLE:
                                    try:
                                        # Try to get model from details for SHAP
                                        details = res_dict.get("details", {})
                                        if "xgb_model" in details or "lgbm_model" in details:
                                            st.caption("🔬 Computing SHAP values (this may take a moment)...")
                                            # Note: Full SHAP implementation would require model and X_test
                                            # For now, show feature importances
                                            st.info("💡 SHAP available but requires model access. Showing feature importances instead.")
                                    except Exception:
                                        pass
                                
                                if not shap_used:
                                    if SHAP_AVAILABLE:
                                        st.caption("📊 Showing permutation importances (SHAP not available for this model type)")
                                    else:
                                        st.caption("📊 Showing model feature importances (SHAP not installed - install with `pip install shap`)")
                                
                                # Display feature importance chart
                                driver_fig = px.bar(
                                    fi_df, x="importance", y="feature",
                                    orientation='h', title="🎯 Feature Influence on Forecast",
                                    labels={"importance": "Importance Score", "feature": "Feature"},
                                    color="importance",
                                    color_continuous_scale="Viridis"
                                )
                                driver_fig.update_layout(
                                    template="plotly_dark", 
                                    height=400,
                                    xaxis_title="Feature Importance",
                                    yaxis_title="Features",
                                    showlegend=False
                                )
                                st.plotly_chart(driver_fig, use_container_width=True, key="feature_importance_chart")
                                
                                st.caption("💡 Features with higher importance have greater influence on forecast predictions")
                                
                                # Show table
                                st.dataframe(fi_df, use_container_width=True, hide_index=True)
                                
                                # SHAP Values (if available)
                                if SHAP_AVAILABLE:
                                    try:
                                        from utils.advanced_forecasting import compute_shap_values
                                        # Note: Would need model access and feature matrix for full SHAP
                                        st.info("🔬 Full SHAP analysis requires model access. Feature importances above show relative importance.")
                                    except:
                                        pass
                            else:
                                st.info("Feature importances not available for this model.")
                        else:
                            st.info("Feature importance analysis not available. Model may not support feature importance extraction.")
                    
                    # Prophet Components
                    st.markdown("#### Prophet Decomposition Components")
                    if prophet_components:
                        changepoints = prophet_components.get("changepoints", [])
                        if changepoints:
                            st.info(f"**Changepoints detected:** {len(changepoints)} trend change points identified by Prophet model")
                            if len(changepoints) > 0:
                                cp_df = pd.DataFrame({"changepoint": pd.to_datetime(changepoints[:10])})
                                st.dataframe(cp_df, use_container_width=True, hide_index=True)
                        else:
                            st.info("No changepoints detected. Trend is relatively stable.")
                    else:
                        st.info("Prophet components not available for this model.")
                    
                    # Ensemble weights
                    details = res_dict.get("details", {})
                    if "weights" in details:
                        weights = details["weights"]
                        st.markdown("#### Ensemble Model Weights")
                        weights_df = pd.DataFrame(list(weights.items()), columns=["Model", "Weight"])
                        weights_fig = px.bar(weights_df, x="Model", y="Weight", title="Ensemble Model Weights")
                        weights_fig.update_layout(template="plotly_dark", height=300)
                        st.plotly_chart(weights_fig, use_container_width=True, key="ensemble_weights_chart")
                
                with tab_business:
                    st.markdown("### 📈 Business Implications")
                    
                    # Business insight summary
                    st.markdown("#### Executive Summary")
                    business_summary = f"""
**Product:** {selected_product}

**Sales Trend Direction:** {'📈 Upward Growth' if growth_pct > 0 else '📉 Declining Trend' if growth_pct < -5 else '➡️ Stable'}

**Key Projections:**
- **Next Quarter Forecast:** {next_q_forecast:,.0f} units
- **Growth Rate:** {growth_pct:+.1f}% (vs previous quarter)
- **Peak Sales Month:** {month_names[peak_month_idx-1]} (expected highest demand)
- **Minimum Sales Month:** {month_names[min_month_idx-1]} (expected lowest demand)

**Model Confidence:**
- **R² Score:** {r2_score_val:.3f} ({'High' if r2_score_val > 0.7 else 'Moderate' if r2_score_val > 0.5 else 'Low'} confidence)
- **Forecast Accuracy (RMSE):** {rmse_val:.2f} units

**Business Recommendations:**
"""
                    
                    if growth_pct > 10:
                        business_summary += f"- ✅ **Strong Growth Expected:** Increase inventory and marketing focus for {selected_product}\n"
                    elif growth_pct < -10:
                        business_summary += f"- ⚠️ **Declining Trend:** Investigate market factors and consider promotional strategies\n"
                    
                    if peak_month_idx != min_month_idx:
                        business_summary += f"- 📅 **Seasonal Planning:** Prepare for peak demand in {month_names[peak_month_idx-1]}, reduce inventory in {month_names[min_month_idx-1]}\n"
                    
                    if not np.isnan(stockout_risk) and stockout_risk > 0.5:
                        business_summary += f"- 🚨 **Stock Alert:** High stockout risk detected - consider increasing inventory levels\n"
                    
                    business_summary += f"\n**Automated Insight:** {narrative}"
                    
                    st.markdown(business_summary)
                    
                    # Trend direction visualization
                    st.markdown("#### Trend Analysis")
                    trend_direction = "📈 Growing" if growth_pct > 5 else "📉 Declining" if growth_pct < -5 else "➡️ Stable"
                    st.metric("Trend Direction", trend_direction, delta=f"{growth_pct:+.1f}%")
                    
                    # Monthly forecast breakdown
                    if len(forecast_df) > 0 and "date" in forecast_df.columns and "yhat" in forecast_df.columns:
                        monthly_forecast = forecast_df.groupby(forecast_df["date"].dt.to_period("M"))["yhat"].sum().reset_index()
                        monthly_forecast["Month"] = monthly_forecast["date"].astype(str)
                        monthly_fig = px.bar(monthly_forecast, x="Month", y="yhat", 
                                            title="Monthly Forecast Breakdown",
                                            labels={"yhat": "Forecasted Sales", "Month": "Month"})
                        monthly_fig.update_layout(template="plotly_dark", height=400, xaxis_tickangle=-45)
                        st.plotly_chart(monthly_fig, use_container_width=True, key="monthly_breakdown_chart")
                
                # What-if Price Slider (outside tabs, always visible)
                with st.expander("💡 Pricing Simulation (What-If)", expanded=False):
                    if "price" in ts.columns and ts["price"].notna().any():
                        current_price = float(ts["price"].dropna().iloc[-1])
                        
                        # Calculate elasticity
                        try:
                            elasticity = calculate_price_elasticity(features_df, selected_product)
                        except Exception:
                            elasticity = -1.2  # Conservative default
                            st.caption("⚠️ Elasticity calculation failed, using conservative estimate")
                        
                        price_change_pct = st.slider("Adjust price (%)", -10, 10, 0, 1, key="price_slider_subtab")
                        new_price = current_price * (1 + price_change_pct / 100.0)
                        
                        # Simple demand model: new_qty = old_qty * (new_price/old_price) ** elasticity
                        avg_recent_qty = float(ts["sales_qty"].tail(8).mean()) if len(ts) >= 8 else float(ts["sales_qty"].mean())
                        projected_qty = avg_recent_qty * ((new_price / current_price) ** elasticity) if elasticity != 0 else avg_recent_qty
                        projected_revenue = projected_qty * new_price
                        current_revenue = avg_recent_qty * current_price
                        revenue_change = projected_revenue - current_revenue
                        revenue_change_pct = (revenue_change / current_revenue * 100) if current_revenue > 0 else 0
                        
                        col_p1, col_p2, col_p3 = st.columns(3)
                        with col_p1:
                            st.metric("💰 Current Price", f"₹{current_price:.2f}")
                            st.metric("📦 Current Weekly Qty", f"{avg_recent_qty:.0f}")
                        with col_p2:
                            st.metric("💵 New Price", f"₹{new_price:.2f}", delta=f"{price_change_pct:+.1f}%")
                            st.metric("📊 Projected Weekly Qty", f"{projected_qty:.0f}", delta=f"{(projected_qty/avg_recent_qty - 1)*100:+.1f}%")
                        with col_p3:
                            st.metric("💸 Current Revenue", f"₹{current_revenue:.0f}")
                            st.metric("📈 Projected Revenue", f"₹{projected_revenue:.0f}", delta=f"{revenue_change_pct:+.1f}%")
                        
                        # Mini comparison chart
                        rev_df = pd.DataFrame({
                            'Scenario': ['Current', 'New Price'],
                            'Weekly Revenue': [current_revenue, projected_revenue]
                        })
                        rev_fig = px.bar(rev_df, x='Scenario', y='Weekly Revenue', 
                                        title="Revenue Comparison", color='Scenario',
                                        color_discrete_map={'Current': C_HISTORY, 'New Price': C_FORECAST})
                        rev_fig.update_layout(template="plotly_dark", height=300, showlegend=False)
                        st.plotly_chart(rev_fig, use_container_width=True, key="revenue_comparison_chart")
                        
                        elasticity_label = "Elastic" if elasticity < -1 else "Inelastic" if elasticity > -0.5 else "Unit Elastic"
                        st.caption(f"💡 Elasticity: **{elasticity:.2f}** ({elasticity_label}) | "
                                 f"Price {'increase' if price_change_pct > 0 else 'decrease'} of {abs(price_change_pct)}% "
                                 f"{'increases' if revenue_change > 0 else 'decreases'} revenue by ₹{abs(revenue_change):,.0f}/week")
                    else:
                        st.info("Price data not available for this product.")
                
                # Download buttons and export
                with st.expander("📥 Export & Download", expanded=False):
                    col_d1, col_d2, col_d3 = st.columns(3)
                    
                    with col_d1:
                        # Download forecast CSV
                        forecast_csv = forecast_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "⬇️ Download Forecast CSV",
                            data=forecast_csv,
                            file_name=f"forecast_{selected_product.replace(' ', '_')}.csv",
                            mime="text/csv",
                            key="download_forecast_subtab"
                        )
                    
                    with col_d2:
                        # Download metrics JSON
                        metrics_json = json.dumps(metrics, indent=2).encode('utf-8')
                        st.download_button(
                            "⬇️ Download Metrics JSON",
                            data=metrics_json,
                            file_name=f"metrics_{selected_product.replace(' ', '_')}.json",
                            mime="application/json",
                            key="download_metrics_subtab"
                        )
                    
                    with col_d3:
                        # Clear cache button - clears cache and output files
                        if st.button("🗑 Clear Cache & Retrain", key="clear_cache_subtab"):
                            st.cache_data.clear()
                            if "interactive_fc_ready_subtab" in st.session_state:
                                del st.session_state["interactive_fc_ready_subtab"]
                            if "forecast_result_subtab" in st.session_state:
                                del st.session_state["forecast_result_subtab"]
                            if "cached_horizon_weeks" in st.session_state:
                                del st.session_state["cached_horizon_weeks"]
                            # Clear output files
                            try:
                                forecast_csv_path = os.path.join(OUTPUT_DIR, "forecasting_results.csv")
                                if os.path.exists(forecast_csv_path):
                                    os.remove(forecast_csv_path)
                                metrics_json_path = os.path.join(OUTPUT_DIR, "forecasting_metrics.json")
                                if os.path.exists(metrics_json_path):
                                    os.remove(metrics_json_path)
                            except Exception:
                                pass
                            st.success("✅ Cache and outputs cleared! Refresh to retrain.")
                            st.rerun()
                    
                    # Export insight text
                    # Get current horizon for export
                    export_horizon = st.session_state.get("selected_horizon_weeks", display_horizon if 'display_horizon' in locals() else 26)
                    
                    insight_text = f"""RetailSense Forecast Summary for {selected_product}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Forecast Horizon: {export_horizon} weeks
Model Performance:
  - R² Score: {r2_score_val:.3f}
  - RMSE: {rmse_val:.2f}
  - MAE: {mae_val:.2f}

Key Insights:
  - Next Week Forecast: {next_week_total:.0f} units
  - Growth Rate: {growth_pct:+.1f}%
  - Peak Month: {month_names[peak_month_idx-1]}
  - Minimum Month: {month_names[min_month_idx-1]}
  - Stockout Risk: {stockout_display}

{narrative}
"""
                    st.download_button(
                        "⬇️ Download Insight Text",
                        data=insight_text.encode('utf-8'),
                        file_name=f"insight_{selected_product.replace(' ', '_')}.txt",
                        mime="text/plain",
                        key="download_insight_subtab"
                    )
            
            else:
                st.info("💡 Click '🚀 Generate Forecast' button above to see predictions")
        
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
    
    # Try to load data_with_all_features.csv for anomaly detection
    @st.cache_data(show_spinner="Loading data...")
    def load_tab3_data():
        if os.path.exists(FEATURES_DATA_PATH):
            try:
                return pd.read_csv(FEATURES_DATA_PATH, low_memory=False, encoding='utf-8')
            except:
                return None
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