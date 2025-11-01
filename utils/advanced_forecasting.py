# utils/advanced_forecasting.py
"""
Placement-grade hybrid forecasting engine (Prophet + XGBoost + LightGBM)
- Works with weekly data using week_start as date column and sales_qty as target
- Supports horizons up to 156 weeks (3 years)
- Prophet handles long-term seasonality and provides uncertainty intervals
- XGBoost / LightGBM capture short-term patterns and feature-based learning
- Generates realistic, interpretable forecasts with comprehensive metrics
- Saves results to outputs/ directory with visualizations
- Deterministic demo behavior via np.random.seed(42)
"""

from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math
import json
import time
import logging

# ML libs
import xgboost as xgb
import lightgbm as lgb

# Prophet
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    warnings.warn("Prophet not available. Install with: pip install prophet")

# For sMAPE and MAPE
def smape(actual, forecast):
    """Symmetric Mean Absolute Percentage Error"""
    actual = np.array(actual, dtype=float)
    forecast = np.array(forecast, dtype=float)
    denom = (np.abs(actual) + np.abs(forecast)) / 2.0
    mask = denom == 0
    denom[mask] = 1.0
    out = np.abs(forecast - actual) / denom
    out[mask] = 0.0
    return np.mean(out) * 100.0

def mape(actual, forecast):
    """Mean Absolute Percentage Error"""
    actual = np.array(actual, dtype=float)
    forecast = np.array(forecast, dtype=float)
    mask = actual != 0
    return np.mean(np.abs((actual[mask] - forecast[mask]) / actual[mask])) * 100.0

# Setup output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
FORECAST_CSV = os.path.join(OUTPUT_DIR, "forecasting_results.csv")
FORECAST_METRICS_JSON = os.path.join(OUTPUT_DIR, "forecasting_metrics.json")
FORECAST_VISUALS_DIR = os.path.join(OUTPUT_DIR, "forecast_visuals")
os.makedirs(FORECAST_VISUALS_DIR, exist_ok=True)

# Deterministic seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EnsembleResult:
    """Container for ensemble forecasting results"""
    history: pd.DataFrame  # Historical data with fitted values
    forecast: pd.DataFrame  # Future forecast with confidence intervals
    metrics: Dict[str, float]  # Model performance metrics
    residuals: pd.DataFrame  # Residual analysis
    details: Dict[str, Any]  # Additional model details (weights, feature importances, etc.)
    feature_importances: Optional[pd.DataFrame] = None  # Feature importance for XGBoost/LightGBM
    prophet_components: Optional[pd.DataFrame] = None  # Prophet decomposition (trend, seasonality)

# -------------------------
# Preprocessing Utilities
# -------------------------
def _preprocess_weekly_data(df: pd.DataFrame, date_col: str = "week_start", target_col: str = "sales_qty") -> pd.DataFrame:
    """
    Preprocess time series data for weekly forecasting.
    
    Args:
        df: Input dataframe with date and sales columns
        date_col: Name of date column (default: 'week_start')
        target_col: Name of target column (default: 'sales_qty')
    
    Returns:
        Preprocessed dataframe with weekly frequency
    """
    df = df.copy()
    
    # Convert date column - handle both 'week_start' and 'date'
    if date_col not in df.columns:
        if "date" in df.columns:
            date_col = "date"
        else:
            raise ValueError(f"Date column '{date_col}' or 'date' not found in dataframe")
    
    # Convert to datetime
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    
    # Select and rename columns
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe")
    
    df = df[[date_col, target_col]].copy()
    df.columns = ["date", "sales_qty"]
    
    # Remove duplicates (keep last)
    df = df.drop_duplicates(subset=["date"], keep="last")
    df = df.sort_values("date").reset_index(drop=True)
    
    # Handle missing, zero, and negative values
    # Replace negative with zero
    df["sales_qty"] = df["sales_qty"].clip(lower=0)
    
    # Handle zeros - replace with small positive value if it's isolated
    # But keep legitimate zeros
    zero_mask = df["sales_qty"] == 0
    if zero_mask.sum() > 0:
        # If more than 50% are zeros, use forward fill; otherwise interpolate
        if zero_mask.sum() / len(df) > 0.5:
            df["sales_qty"] = df["sales_qty"].replace(0, np.nan).fillna(method="ffill").fillna(method="bfill")
        else:
            # Interpolate isolated zeros
            df["sales_qty"] = df["sales_qty"].replace(0, np.nan)
    
    # Set date as index
    df = df.set_index("date")
    
    # Resample to weekly frequency (W-SUN: week ending on Sunday)
    df_weekly = df.resample("W-SUN").agg({
        "sales_qty": "sum"  # Sum sales for the week
    })
    
    # Fill missing weeks with intelligent interpolation
    df_weekly["sales_qty"] = df_weekly["sales_qty"].interpolate(
        method="time", 
        limit=8  # Interpolate up to 8 missing weeks
    ).bfill().ffill()
    
    # Final cleanup: ensure non-negative and clip extreme outliers
    df_weekly["sales_qty"] = df_weekly["sales_qty"].clip(lower=0)
    
    # Clip outliers (values > mean + 3*std)
    mean_val = df_weekly["sales_qty"].mean()
    std_val = df_weekly["sales_qty"].std()
    if not pd.isna(std_val) and std_val > 0:
        upper_bound = mean_val + 3 * std_val
        df_weekly["sales_qty"] = df_weekly["sales_qty"].clip(upper=upper_bound)
    
    # Reset index
    df_weekly = df_weekly.reset_index()
    df_weekly.columns = ["date", "sales_qty"]
    
    # Ensure minimum data points
    if len(df_weekly) < 8:
        raise ValueError(f"Insufficient data: need at least 8 weeks, got {len(df_weekly)}")
    
    logger.info(f"Preprocessed to {len(df_weekly)} weekly observations")
    return df_weekly

def _create_weekly_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create lag features and time-based features for weekly data.
    
    Args:
        df: DataFrame with 'date' and 'sales_qty' columns
    
    Returns:
        DataFrame with additional feature columns
    """
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)
    
    # Lag features (1, 2, 3, 7 weeks ago - for weekly data)
    for lag in [1, 2, 3, 7]:
        df[f"lag_{lag}"] = df["sales_qty"].shift(lag)
    
    # Rolling statistics (4, 8, 12, 26 weeks)
    for window in [4, 8, 12, 26]:
        df[f"rolling_mean_{window}"] = df["sales_qty"].rolling(window=window, min_periods=1).mean()
        df[f"rolling_std_{window}"] = df["sales_qty"].rolling(window=window, min_periods=1).std().fillna(0.0)
    
    # Time-based features
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["year"] = df["date"].dt.year
    df["weekofyear"] = df["date"].dt.isocalendar().week
    
    # Cyclical encoding for month (sine/cosine)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    
    # Cyclical encoding for week of year
    df["week_sin"] = np.sin(2 * np.pi * df["weekofyear"] / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["weekofyear"] / 52)
    
    # Fill NaN values in lag features
    lag_cols = [c for c in df.columns if c.startswith("lag_")]
    for col in lag_cols:
        df[col] = df[col].fillna(df["sales_qty"].rolling(4, min_periods=1).mean())
    
    return df

# -------------------------
# Model Wrappers
# -------------------------
def _fit_prophet(df_history: pd.DataFrame, horizon_weeks: int, debug: bool = False) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[Any], Optional[pd.DataFrame]]:
    """
    Fit Prophet model on weekly data.
    
    Returns:
        hist_fit: Historical fitted values
        future_forecast: Future forecast with confidence intervals
        prophet_model: Fitted Prophet model
        components: Prophet decomposition components
    """
    if not PROPHET_AVAILABLE:
        logger.warning("Prophet not available, skipping Prophet model")
        return None, None, None, None
    
    df = df_history.copy()
    
    # Prepare Prophet format
    df = df.rename(columns={"date": "ds", "sales_qty": "y"})
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df = df.dropna(subset=["ds", "y"])
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["y"])
    df = df[df["y"] >= 0]
    df = df.sort_values("ds").drop_duplicates("ds")
    
    if len(df) < 10:
        logger.warning(f"Prophet skipped: Too few data points ({len(df)}). Need ≥10.")
        return None, None, None, None
    
    try:
        # Configure Prophet for weekly data
        m = Prophet(
            yearly_seasonality=True,  # Annual seasonality
            weekly_seasonality=False,  # Disable weekly for weekly aggregated data
            daily_seasonality=False,   # Disable daily
            seasonality_mode="multiplicative",
            changepoint_prior_scale=0.05,  # Lower = less flexible trend changes
            seasonality_prior_scale=10.0,
            n_changepoints=min(25, max(5, int(len(df) / 4))),
            changepoint_range=0.95
        )
        
        # Add monthly seasonality if enough data
        if len(df) > 52:
            m.add_seasonality(name="monthly", period=30.5, fourier_order=5)
        
        # Log-transform to handle multiplicative seasonality better
        y_original = df["y"].values
        df["y"] = np.log1p(df["y"])
        
        # Fit model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(df)
        
        # Historical fit
        hist_pred = m.predict(df)
        hist_fit = hist_pred[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        hist_fit["yhat"] = np.expm1(hist_fit["yhat"])
        hist_fit["yhat_lower"] = np.expm1(hist_fit["yhat_lower"])
        hist_fit["yhat_upper"] = np.expm1(hist_fit["yhat_upper"])
        hist_fit = hist_fit.rename(columns={"ds": "date"})
        
        # Future forecast
        future = m.make_future_dataframe(periods=horizon_weeks, freq="W", include_history=False)
        fut_pred = m.predict(future)
        fut_forecast = fut_pred[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        fut_forecast["yhat"] = np.expm1(fut_forecast["yhat"])
        fut_forecast["yhat_lower"] = np.expm1(fut_forecast["yhat_lower"])
        fut_forecast["yhat_upper"] = np.expm1(fut_forecast["yhat_upper"])
        fut_forecast = fut_forecast.rename(columns={"ds": "date"})
        
        # Prophet components
        components = hist_pred[["ds", "trend", "yearly", "monthly"]].copy() if "monthly" in hist_pred.columns else hist_pred[["ds", "trend", "yearly"]].copy()
        components = components.rename(columns={"ds": "date"})
        
        logger.info("Prophet model fitted successfully")
        return hist_fit, fut_forecast, m, components
    
    except Exception as e:
        logger.warning(f"Prophet fitting failed: {e}")
        if debug:
            import traceback
            traceback.print_exc()
        return None, None, None, None

def _fit_xgb(X_train, y_train, X_val=None, y_val=None, fast_mode=True):
    """Train XGBoost model"""
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "seed": RANDOM_SEED,
        "verbosity": 0,
        "max_depth": 6 if fast_mode else 8,
        "learning_rate": 0.1 if fast_mode else 0.05,
        "n_estimators": 200 if fast_mode else 500,
    }
    
    num_round = params.pop("n_estimators")
    dtrain = xgb.DMatrix(X_train, label=y_train)
    evals = [(dtrain, "train")]
    
    if X_val is not None:
        dval = xgb.DMatrix(X_val, label=y_val)
        evals.append((dval, "val"))
    
    model = xgb.train(
        params, 
        dtrain, 
        num_round, 
        evals=evals, 
        early_stopping_rounds=20 if X_val is not None else None,
        verbose_eval=False
    )
    return model

def _fit_lgb(X_train, y_train, X_val=None, y_val=None, fast_mode=True):
    """Train LightGBM model"""
    params = {
        "objective": "regression",
        "metric": "rmse",
        "seed": RANDOM_SEED,
        "verbosity": -1,
        "max_depth": 6 if fast_mode else 8,
        "learning_rate": 0.1 if fast_mode else 0.05,
        "num_iterations": 200 if fast_mode else 500,
    }
    
    num_iter = params.pop("num_iterations")
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_sets = [train_data]
    valid_names = ["train"]
    
    if X_val is not None:
        val_data = lgb.Dataset(X_val, label=y_val)
        valid_sets.append(val_data)
        valid_names.append("val")
    
    model = lgb.train(
        params,
        train_data,
        num_iter,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=[lgb.early_stopping(stopping_rounds=20)] if X_val is not None else None
    )
    return model

def _rmse(actual, predicted):
    """Calculate RMSE"""
    return float(np.sqrt(mean_squared_error(actual, predicted)))

def _mae(actual, predicted):
    """Calculate MAE"""
    return float(mean_absolute_error(actual, predicted))

# -------------------------
# Main Ensemble Training Function
# -------------------------
def train_ensemble(df: pd.DataFrame, horizon_weeks: int = 156, debug: bool = False, fast_mode: bool = True) -> EnsembleResult:
    """
    Train hybrid ensemble model for weekly sales forecasting.
    
    Args:
        df: Input dataframe with date column ('week_start' or 'date') and 'sales_qty'
        horizon_weeks: Forecast horizon in weeks (default: 156 weeks = 3 years)
        debug: Enable debug logging
        fast_mode: Use faster, lighter models (default: True)
    
    Returns:
        EnsembleResult with history, forecast, metrics, and details
    """
    start_time = time.time()
    
    # Validate input
    if df.empty:
        raise ValueError("Input dataframe is empty")
    
    # Identify date column
    date_col = "week_start" if "week_start" in df.columns else ("date" if "date" in df.columns else None)
    if date_col is None:
        raise ValueError("Date column ('week_start' or 'date') not found in dataframe")
    
    if "sales_qty" not in df.columns:
        raise ValueError("Target column 'sales_qty' not found in dataframe")
    
    # Preprocess to weekly frequency
    logger.info("Preprocessing data to weekly frequency...")
    df_weekly = _preprocess_weekly_data(df, date_col=date_col, target_col="sales_qty")
    
    if len(df_weekly) < 8:
        raise ValueError(f"Insufficient data after preprocessing: {len(df_weekly)} weeks. Need at least 8 weeks.")
    
    history_df = df_weekly.copy()
    
    # Cap horizon at 156 weeks (3 years)
    horizon_weeks = min(int(horizon_weeks), 156)
    
    logger.info(f"Training ensemble for {horizon_weeks} weeks ahead (from {len(history_df)} historical weeks)")
    
    # Step 1: Fit Prophet (long-term seasonality)
    logger.info("Fitting Prophet model...")
    hist_prophet, fut_prophet, prophet_model, prophet_components = _fit_prophet(
        history_df, horizon_weeks=horizon_weeks, debug=debug
    )
    
    # Step 2: Create features for ML models
    logger.info("Creating features for ML models...")
    feat_df = _create_weekly_features(history_df)
    
    # Select feature columns (exclude date and target)
    exclude = {"date", "sales_qty"}
    feature_cols = [c for c in feat_df.columns if c not in exclude]
    
    # Prepare ML data
    feat_df_clean = feat_df.dropna(subset=["sales_qty"]).reset_index(drop=True)
    
    if len(feat_df_clean) < 10:
        raise ValueError(f"Insufficient clean data for ML: {len(feat_df_clean)} rows. Need ≥10.")
    
    # Fill remaining NaN in features
    X_all = feat_df_clean[feature_cols].fillna(method="ffill").fillna(method="bfill").fillna(0.0)
    y_all = feat_df_clean["sales_qty"].values
    
    # Train/validation split (85/15)
    split_idx = int(len(X_all) * 0.85)
    X_train = X_all.iloc[:split_idx].values
    y_train = y_all[:split_idx]
    X_val = X_all.iloc[split_idx:].values if split_idx < len(X_all) else None
    y_val = y_all[split_idx:] if split_idx < len(X_all) else None
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val) if X_val is not None else None
    
    # Step 3: Train ML models
    ml_models = {}
    rmse_scores = {}
    feature_importances = {}
    
    # XGBoost
    logger.info("Training XGBoost...")
    try:
        xgb_model = _fit_xgb(X_train_scaled, y_train, X_val_scaled, y_val, fast_mode=fast_mode)
        ml_models["xgb"] = xgb_model
        
        # Validation RMSE
        if X_val_scaled is not None:
            xgb_preds = xgb_model.predict(xgb.DMatrix(X_val_scaled))
            rmse_scores["xgb"] = _rmse(y_val, xgb_preds)
        else:
            rmse_scores["xgb"] = np.nan
        
        # Feature importances
        try:
            importances = xgb_model.get_score(importance_type="gain")
            # Convert to DataFrame
            xgb_importance_df = pd.DataFrame({
                "feature": list(importances.keys()),
                "importance": list(importances.values())
            }).sort_values("importance", ascending=False)
            feature_importances["xgb"] = xgb_importance_df
        except:
            feature_importances["xgb"] = None
        
        logger.info(f"XGBoost RMSE: {rmse_scores['xgb']:.2f}")
    except Exception as e:
        logger.warning(f"XGBoost training failed: {e}")
        ml_models["xgb"] = None
        rmse_scores["xgb"] = np.inf
        feature_importances["xgb"] = None
    
    # LightGBM
    logger.info("Training LightGBM...")
    try:
        lgb_model = _fit_lgb(X_train_scaled, y_train, X_val_scaled, y_val, fast_mode=fast_mode)
        ml_models["lgbm"] = lgb_model
        
        # Validation RMSE
        if X_val_scaled is not None:
            lgb_preds = lgb_model.predict(X_val_scaled)
            rmse_scores["lgbm"] = _rmse(y_val, lgb_preds)
        else:
            rmse_scores["lgbm"] = np.nan
        
        # Feature importances
        try:
            importances = lgb_model.feature_importance(importance_type="gain")
            # Map feature indices to names
            lgb_importance_df = pd.DataFrame({
                "feature": feature_cols,
                "importance": importances
            }).sort_values("importance", ascending=False)
            feature_importances["lgbm"] = lgb_importance_df
        except:
            feature_importances["lgbm"] = None
        
        logger.info(f"LightGBM RMSE: {rmse_scores['lgbm']:.2f}")
    except Exception as e:
        logger.warning(f"LightGBM training failed: {e}")
        ml_models["lgbm"] = None
        rmse_scores["lgbm"] = np.inf
        feature_importances["lgbm"] = None
    
    # Prophet RMSE
    if hist_prophet is not None and not hist_prophet.empty:
        hist_merge = history_df.merge(hist_prophet[["date", "yhat"]], on="date", how="left")
        prophet_rmse = _rmse(
            hist_merge["sales_qty"].fillna(0).values,
            hist_merge["yhat"].fillna(0).values
        )
        rmse_scores["prophet"] = prophet_rmse
        logger.info(f"Prophet RMSE: {prophet_rmse:.2f}")
    else:
        rmse_scores["prophet"] = np.inf
    
    # Step 4: Calculate ensemble weights (inverse RMSE)
    weights = {}
    inv_rmse = {}
    for model_name, rmse in rmse_scores.items():
        if rmse != np.inf and not np.isnan(rmse) and rmse > 0:
            inv_rmse[model_name] = 1.0 / (rmse + 1e-6)
        else:
            inv_rmse[model_name] = 0.0
    
    total_inv = sum(inv_rmse.values())
    if total_inv > 0:
        for model_name in rmse_scores.keys():
            weights[model_name] = inv_rmse[model_name] / total_inv
    else:
        # Fallback: equal weights
        weights = {"prophet": 0.4, "xgb": 0.3, "lgbm": 0.3}
        for model_name in ["prophet", "xgb", "lgbm"]:
            if model_name not in rmse_scores or rmse_scores[model_name] == np.inf:
                weights[model_name] = 0.0
        
        # Renormalize
        total_w = sum(weights.values())
        if total_w > 0:
            weights = {k: v / total_w for k, v in weights.items()}
    
    logger.info(f"Ensemble weights: {weights}")
    
    # Step 5: Generate in-sample predictions for metrics
    X_all_scaled = scaler.transform(X_all.fillna(method="ffill").fillna(method="bfill").fillna(0.0))
    
    in_sample_preds = {}
    
    # Prophet in-sample
    if hist_prophet is not None:
        hist_pred_aligned = hist_prophet.set_index("date").reindex(feat_df_clean["date"])["yhat"].values
        in_sample_preds["prophet"] = hist_pred_aligned
    
    # XGBoost in-sample
    if ml_models.get("xgb") is not None:
        xgb_in = ml_models["xgb"].predict(xgb.DMatrix(X_all_scaled))
        in_sample_preds["xgb"] = xgb_in
    
    # LightGBM in-sample
    if ml_models.get("lgbm") is not None:
        lgb_in = ml_models["lgbm"].predict(X_all_scaled)
        in_sample_preds["lgbm"] = lgb_in
    
    # Blend in-sample predictions
    ensemble_in_sample = np.zeros(len(feat_df_clean))
    for i in range(len(feat_df_clean)):
        pred = 0.0
        total_weight = 0.0
        for model_name, weight in weights.items():
            if model_name in in_sample_preds and weight > 0:
                pred += weight * in_sample_preds[model_name][i]
                total_weight += weight
        ensemble_in_sample[i] = pred / total_weight if total_weight > 0 else 0.0
    
    # Calculate metrics
    actuals = feat_df_clean["sales_qty"].values
    ensemble_rmse = _rmse(actuals, ensemble_in_sample)
    ensemble_mae = _mae(actuals, ensemble_in_sample)
    ensemble_smape = smape(actuals, ensemble_in_sample)
    ensemble_mape = mape(actuals, ensemble_in_sample)
    
    try:
        ensemble_r2 = float(r2_score(actuals, ensemble_in_sample))
    except:
        ensemble_r2 = np.nan
    
    metrics = {
        "prophet_rmse": float(rmse_scores.get("prophet", np.nan)),
        "xgb_rmse": float(rmse_scores.get("xgb", np.nan)),
        "lgbm_rmse": float(rmse_scores.get("lgbm", np.nan)),
        "ensemble_rmse": ensemble_rmse,
        "ensemble_mae": ensemble_mae,
        "ensemble_smape": ensemble_smape,
        "ensemble_mape": ensemble_mape,
        "ensemble_r2": ensemble_r2,
    }
    
    # Step 6: Generate future forecast
    logger.info(f"Generating {horizon_weeks}-week forecast...")
    
    last_date = history_df["date"].max()
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(weeks=1),
        periods=horizon_weeks,
        freq="W-SUN"
    )
    
    forecast_rows = []
    recursive_history = history_df.copy()
    
    # Helper function to build features for a future week
    def build_future_features(hist_df: pd.DataFrame, future_date: pd.Timestamp):
        """Build feature vector for a future week"""
        row = {"date": future_date}
        last = hist_df.copy().reset_index(drop=True)
        
        # Lags
        for lag in [1, 2, 3, 7]:
            if len(last) >= lag:
                row[f"lag_{lag}"] = float(last["sales_qty"].iloc[-lag])
            else:
                row[f"lag_{lag}"] = float(last["sales_qty"].tail(4).mean()) if len(last) > 0 else 0.0
        
        # Rolling stats
        for window in [4, 8, 12, 26]:
            tail_len = min(window, len(last))
            row[f"rolling_mean_{window}"] = float(last["sales_qty"].tail(tail_len).mean()) if tail_len > 0 else 0.0
            row[f"rolling_std_{window}"] = float(last["sales_qty"].tail(tail_len).std()) if tail_len > 1 else 0.0
        
        # Time features
        row["month"] = future_date.month
        row["quarter"] = future_date.quarter
        row["year"] = future_date.year
        row["weekofyear"] = future_date.isocalendar().week
        
        # Cyclical
        row["month_sin"] = np.sin(2 * np.pi * row["month"] / 12)
        row["month_cos"] = np.cos(2 * np.pi * row["month"] / 12)
        row["week_sin"] = np.sin(2 * np.pi * row["weekofyear"] / 52)
        row["week_cos"] = np.cos(2 * np.pi * row["weekofyear"] / 52)
        
        return row
    
    # Generate forecasts week by week
    for week_idx in range(horizon_weeks):
        current_date = future_dates[week_idx]
        
        # Prophet prediction
        prophet_pred = None
        prophet_lower = None
        prophet_upper = None
        if fut_prophet is not None and not fut_prophet.empty:
            try:
                prophet_row = fut_prophet[fut_prophet["date"] == current_date]
                if not prophet_row.empty:
                    prophet_pred = float(prophet_row["yhat"].iloc[0])
                    prophet_lower = float(prophet_row["yhat_lower"].iloc[0])
                    prophet_upper = float(prophet_row["yhat_upper"].iloc[0])
            except:
                pass
        
        # ML predictions (recursive)
        xgb_pred = None
        lgb_pred = None
        
        if len(recursive_history) >= 7:  # Need some history for features
            try:
                row = build_future_features(recursive_history, current_date)
                X_row = pd.DataFrame([row])[feature_cols]
                X_row = X_row.fillna(method="ffill").fillna(method="bfill").fillna(0.0)
                X_row_scaled = scaler.transform(X_row.values)
                
                if ml_models.get("xgb") is not None:
                    xgb_pred = float(ml_models["xgb"].predict(xgb.DMatrix(X_row_scaled))[0])
                    xgb_pred = max(0.0, xgb_pred)  # Ensure non-negative
                
                if ml_models.get("lgbm") is not None:
                    lgb_pred = float(ml_models["lgbm"].predict(X_row_scaled)[0])
                    lgb_pred = max(0.0, lgb_pred)  # Ensure non-negative
            except Exception as e:
                if debug:
                    logger.warning(f"ML prediction failed for week {week_idx}: {e}")
        
        # Fallback if no predictions
        if prophet_pred is None:
            prophet_pred = float(recursive_history["sales_qty"].tail(8).mean()) if len(recursive_history) >= 8 else 0.0
        
        # Ensemble blend
        blend_pred = 0.0
        total_weight = 0.0
        
        if weights.get("prophet", 0) > 0 and prophet_pred is not None:
            blend_pred += weights["prophet"] * prophet_pred
            total_weight += weights["prophet"]
        
        if weights.get("xgb", 0) > 0 and xgb_pred is not None:
            blend_pred += weights["xgb"] * xgb_pred
            total_weight += weights["xgb"]
        
        if weights.get("lgbm", 0) > 0 and lgb_pred is not None:
            blend_pred += weights["lgbm"] * lgb_pred
            total_weight += weights["lgbm"]
        
        if total_weight > 0:
            final_pred = blend_pred / total_weight
        else:
            final_pred = prophet_pred if prophet_pred is not None else 0.0
        
        # Ensure non-negative
        final_pred = max(0.0, final_pred)
        
        # Calculate confidence intervals
        if prophet_lower is not None and prophet_upper is not None:
            ci_lower = min(prophet_lower, final_pred * 0.85)
            ci_upper = max(prophet_upper, final_pred * 1.15)
        else:
            # Use residual-based CI
            residuals = actuals - ensemble_in_sample
            resid_std = np.std(residuals)
            ci_lower = max(0.0, final_pred - 1.96 * resid_std)
            ci_upper = final_pred + 1.96 * resid_std
        
        forecast_rows.append({
            "date": current_date,
            "yhat": final_pred,
            "yhat_lower": ci_lower,
            "yhat_upper": ci_upper,
        })
        
        # Update recursive history
        new_row = pd.DataFrame([{"date": current_date, "sales_qty": final_pred}])
        recursive_history = pd.concat([recursive_history, new_row], ignore_index=True)
    
    forecast_df = pd.DataFrame(forecast_rows)
    
    # Light smoothing to reduce micro-jitter
    if len(forecast_df) >= 3:
        forecast_df["yhat"] = forecast_df["yhat"].rolling(window=3, center=True, min_periods=1).mean()
    
    # Step 7: Prepare residuals
    residuals_df = pd.DataFrame({
        "date": feat_df_clean["date"],
        "actual": actuals,
        "predicted": ensemble_in_sample,
        "residual": actuals - ensemble_in_sample
    })
    
    # Step 8: Prepare historical fit
    hist_fit = history_df.copy()
    hist_fit["fitted"] = ensemble_in_sample[:len(history_df)] if len(ensemble_in_sample) >= len(history_df) else ensemble_in_sample
    
    # Step 9: Save results
    try:
        # Save forecast CSV
        forecast_df[["date", "yhat", "yhat_lower", "yhat_upper"]].to_csv(
            FORECAST_CSV, index=False
        )
        
        # Save metrics JSON
        with open(FORECAST_METRICS_JSON, "w") as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Results saved to {OUTPUT_DIR}")
    except Exception as e:
        logger.warning(f"Failed to save results: {e}")
    
    # Step 10: Prepare details
    details = {
        "weights": weights,
        "training_obs": len(history_df),
        "horizon_weeks": horizon_weeks,
        "duration_seconds": time.time() - start_time,
        "prophet_available": hist_prophet is not None,
        "feature_count": len(feature_cols)
    }
    
    # Combine feature importances
    combined_importance = None
    if feature_importances.get("xgb") is not None or feature_importances.get("lgbm") is not None:
        importance_dict = {}
        for model_name in ["xgb", "lgbm"]:
            if feature_importances.get(model_name) is not None:
                df_imp = feature_importances[model_name]
                for _, row in df_imp.iterrows():
                    feat = row["feature"]
                    imp = row["importance"]
                    if feat not in importance_dict:
                        importance_dict[feat] = 0.0
                    importance_dict[feat] += imp * weights.get(model_name, 0.5)
        
        if importance_dict:
            combined_importance = pd.DataFrame({
                "feature": list(importance_dict.keys()),
                "importance": list(importance_dict.values())
            }).sort_values("importance", ascending=False)
    
    logger.info(f"Forecast generation complete in {details['duration_seconds']:.2f}s")
    
    return EnsembleResult(
        history=hist_fit,
        forecast=forecast_df,
        metrics=metrics,
        residuals=residuals_df,
        details=details,
        feature_importances=combined_importance,
        prophet_components=prophet_components
    )

# === APP WRAPPER: train_ensemble_for_app ===
import hashlib
from typing import Union

def _data_hash(df: pd.DataFrame):
    """Simple hash of dataframe head and shape for caching keys."""
    h = hashlib.sha1()
    h.update(str(df.shape).encode())
    if len(df) > 0 and 'date' in df.columns and 'sales_qty' in df.columns:
        h.update(pd.util.hash_pandas_object(df[['date','sales_qty']].head(10), index=True).values.tobytes())
    return h.hexdigest()[:10]

def train_ensemble_for_app(ts: pd.DataFrame,
                           horizon_days: int = 90,
                           fast_mode: bool = True,
                           debug: bool = False) -> dict:
    """
    App-friendly wrapper around train_ensemble.
    Ensures output fields used by the Streamlit app:
      history (df), forecast (df), metrics (dict),
      feature_importances (dict), prophet_components (dict), details (dict)
    Also saves forecast CSV to FORECAST_CSV.
    
    Args:
        ts: Time series dataframe with 'date' (or 'week_start') and 'sales_qty'
        horizon_days: Forecast horizon in days (converted to weeks)
        fast_mode: Use faster models
        debug: Enable debug logging
    
    Returns:
        dict with keys: history, forecast, metrics, feature_importances, prophet_components, details
    """
    # Defensive checks and conversions expected by train_ensemble
    df = ts.copy()
    if 'date' not in df.columns:
        if 'week_start' in df.columns:
            df = df.rename(columns={'week_start':'date'})
        else:
            raise ValueError("Input ts must contain 'date' or 'week_start' column")
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    if 'sales_qty' not in df.columns:
        # try revenue fallback
        if 'sales' in df.columns:
            df = df.rename(columns={'sales':'sales_qty'})
        else:
            raise ValueError("Input ts must contain 'sales_qty' column")

    if len(df) < 8:
        raise ValueError("Not enough history (need >=8 rows) to run forecasting")

    # Convert horizon_days to horizon_weeks (round up)
    horizon_weeks = max(1, int(np.ceil(horizon_days / 7)))
    
    # call existing train_ensemble (may raise)
    try:
        res = train_ensemble(df, horizon_weeks=horizon_weeks, fast_mode=fast_mode, debug=debug)
    except Exception as e:
        if debug:
            logger.error(f"train_ensemble failed: {e}")
        raise

    # Normalize outputs: ensure DataFrames and dicts exist
    # res is an EnsembleResult object, not a dict
    history = getattr(res, 'history', None)
    if history is None:
        history = pd.DataFrame()
    
    forecast = getattr(res, 'forecast', None)
    if forecast is None:
        forecast = pd.DataFrame()
    
    metrics = getattr(res, 'metrics', None)
    if metrics is None:
        metrics = {}
    
    details = getattr(res, 'details', None)
    if details is None:
        details = {}
    
    feature_importances = getattr(res, 'feature_importances', None)
    prophet_components = getattr(res, 'prophet_components', None)

    # Convert feature_importances DataFrame to dict if needed
    if feature_importances is not None:
        if isinstance(feature_importances, pd.DataFrame):
            if len(feature_importances) > 0 and 'feature' in feature_importances.columns and 'importance' in feature_importances.columns:
                feature_importances_dict = dict(zip(feature_importances['feature'], feature_importances['importance']))
            else:
                feature_importances_dict = {}
        elif isinstance(feature_importances, dict):
            feature_importances_dict = feature_importances
        else:
            feature_importances_dict = {}
    else:
        feature_importances_dict = {}
    
    # Try to get prophet components/changepoints if available
    prophet_components_dict = {}
    if prophet_components is not None:
        if isinstance(prophet_components, pd.DataFrame):
            prophet_components_dict = prophet_components.to_dict()
        elif isinstance(prophet_components, dict):
            prophet_components_dict = prophet_components
    
    # Try to extract changepoints from details or prophet model
    try:
        if 'prophet_model' in details:
            prophet_model = details.get('prophet_model')
            if prophet_model is not None and hasattr(prophet_model, 'changepoints'):
                prophet_components_dict['changepoints'] = prophet_model.changepoints.tolist() if hasattr(prophet_model.changepoints, 'tolist') else list(prophet_model.changepoints)
    except Exception:
        pass

    # Save forecast CSV with product_name if present
    try:
        out_df = forecast.copy()
        if 'product_name' not in out_df.columns:
            # Try to get product_name from input df
            if 'product_name' in df.columns:
                out_df['product_name'] = df['product_name'].iloc[0] if len(df) > 0 else None
            elif 'product' in df.columns:
                out_df['product_name'] = df['product'].iloc[0] if len(df) > 0 else None
        os.makedirs(os.path.dirname(FORECAST_CSV), exist_ok=True)
        out_df.to_csv(FORECAST_CSV, index=False)
        if debug:
            logger.info(f"Saved forecast to {FORECAST_CSV}")
    except Exception as e:
        if debug:
            logger.warning(f"Failed to save forecast CSV: {e}")

    return {
        "history": history,
        "forecast": forecast,
        "metrics": metrics,
        "feature_importances": feature_importances_dict,
        "prophet_components": prophet_components_dict,
        "details": details
    }
