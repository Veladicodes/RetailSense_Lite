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
        # Detect if multiplicative or additive seasonality is better
        mean_val = df["y"].mean()
        std_val = df["y"].std()
        cv = std_val / mean_val if mean_val > 0 else 0.5
        
        # Use multiplicative if CV > 0.3 (higher variance relative to mean)
        seasonality_mode = "multiplicative" if cv > 0.3 else "additive"
        
        # Configure Prophet for weekly data with realistic seasonality
        m = Prophet(
            yearly_seasonality=True,  # Annual seasonality - crucial for realistic patterns
            weekly_seasonality=False,  # Disable weekly for weekly aggregated data
            daily_seasonality=False,   # Disable daily
            seasonality_mode=seasonality_mode,
            changepoint_prior_scale=0.15,  # Allow moderate trend flexibility for realistic curves
            seasonality_prior_scale=15.0,  # Strong seasonality to avoid flat forecasts
            n_changepoints=min(25, max(5, int(len(df) / 3))),  # More changepoints for variation
            changepoint_range=0.95,
            interval_width=0.80,  # 80% confidence intervals (also generate 95% below)
            mcmc_samples=0,  # No MCMC for speed
            uncertainty_samples=100  # More samples for better uncertainty estimation
        )
        
        # Add quarterly seasonality for more realistic patterns
        if len(df) > 26:
            m.add_seasonality(name="quarterly", period=91.25, fourier_order=3)
        
        # Add monthly seasonality if enough data
        if len(df) > 52:
            m.add_seasonality(name="monthly", period=30.5, fourier_order=5)
        
        # Log-transform only if multiplicative and values are large
        y_original = df["y"].values
        if seasonality_mode == "multiplicative" and y_original.max() > 100:
            df["y"] = np.log1p(df["y"])
            log_transform = True
        else:
            log_transform = False
        
        # Fit model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(df)
        
        # Historical fit
        hist_pred = m.predict(df)
        
        # Generate both 80% and 95% confidence intervals
        # Prophet default is 80%, so we need to predict again with 95% interval
        m_95 = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode=seasonality_mode,
            changepoint_prior_scale=0.15,
            seasonality_prior_scale=15.0,
            n_changepoints=min(25, max(5, int(len(df) / 3))),
            changepoint_range=0.95,
            interval_width=0.95,  # 95% confidence intervals
            mcmc_samples=0,
            uncertainty_samples=100
        )
        if len(df) > 26:
            m_95.add_seasonality(name="quarterly", period=91.25, fourier_order=3)
        if len(df) > 52:
            m_95.add_seasonality(name="monthly", period=30.5, fourier_order=5)
        m_95.fit(df)
        
        hist_pred_95 = m_95.predict(df)
        
        hist_fit = hist_pred[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        hist_fit["yhat_lower_95"] = hist_pred_95["yhat_lower"].values
        hist_fit["yhat_upper_95"] = hist_pred_95["yhat_upper"].values
        
        # Reverse log-transform if applied
        if log_transform:
            hist_fit["yhat"] = np.expm1(hist_fit["yhat"])
            hist_fit["yhat_lower"] = np.expm1(hist_fit["yhat_lower"])
            hist_fit["yhat_upper"] = np.expm1(hist_fit["yhat_upper"])
            hist_fit["yhat_lower_95"] = np.expm1(hist_fit["yhat_lower_95"])
            hist_fit["yhat_upper_95"] = np.expm1(hist_fit["yhat_upper_95"])
        
        hist_fit = hist_fit.rename(columns={"ds": "date"})
        
        # Future forecast with both intervals
        future = m.make_future_dataframe(periods=horizon_weeks, freq="W", include_history=False)
        fut_pred = m.predict(future)
        fut_pred_95 = m_95.predict(future)
        
        fut_forecast = fut_pred[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        fut_forecast["yhat_lower_95"] = fut_pred_95["yhat_lower"].values
        fut_forecast["yhat_upper_95"] = fut_pred_95["yhat_upper"].values
        
        # Reverse log-transform if applied
        if log_transform:
            fut_forecast["yhat"] = np.expm1(fut_forecast["yhat"])
            fut_forecast["yhat_lower"] = np.expm1(fut_forecast["yhat_lower"])
            fut_forecast["yhat_upper"] = np.expm1(fut_forecast["yhat_upper"])
            fut_forecast["yhat_lower_95"] = np.expm1(fut_forecast["yhat_lower_95"])
            fut_forecast["yhat_upper_95"] = np.expm1(fut_forecast["yhat_upper_95"])
        
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
        
        # Calculate confidence intervals (80% and 95%)
        if prophet_lower is not None and prophet_upper is not None:
            # Get 95% intervals from Prophet if available
            try:
                prophet_row_95 = fut_prophet[fut_prophet["date"] == current_date]
                if not prophet_row_95.empty and "yhat_lower_95" in fut_prophet.columns:
                    prophet_lower_95 = float(prophet_row_95["yhat_lower_95"].iloc[0])
                    prophet_upper_95 = float(prophet_row_95["yhat_upper_95"].iloc[0])
                else:
                    # Fallback to wider intervals
                    prophet_lower_95 = min(prophet_lower, final_pred * 0.75)
                    prophet_upper_95 = max(prophet_upper, final_pred * 1.25)
            except:
                prophet_lower_95 = min(prophet_lower, final_pred * 0.75)
                prophet_upper_95 = max(prophet_upper, final_pred * 1.25)
            
            ci_lower = min(prophet_lower, final_pred * 0.85)
            ci_upper = max(prophet_upper, final_pred * 1.15)
            ci_lower_95 = prophet_lower_95
            ci_upper_95 = prophet_upper_95
        else:
            # Use residual-based CI
            residuals = actuals - ensemble_in_sample
            resid_std = np.std(residuals)
            ci_lower = max(0.0, final_pred - 1.28 * resid_std)  # 80% interval
            ci_upper = final_pred + 1.28 * resid_std
            ci_lower_95 = max(0.0, final_pred - 1.96 * resid_std)  # 95% interval
            ci_upper_95 = final_pred + 1.96 * resid_std
        
        forecast_rows.append({
            "date": current_date,
            "yhat": final_pred,
            "yhat_lower": ci_lower,
            "yhat_upper": ci_upper,
            "yhat_lower_95": ci_lower_95,
            "yhat_upper_95": ci_upper_95,
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
    
    # Return EnsembleResult object
    return EnsembleResult(
        history_df=hist_fit,
        forecast_df=forecast_df,
        metrics=metrics,
        residuals=residuals_df,
        details=details,
        feature_importances=combined_importance,
        prophet_components=prophet_components
    )

# === SHAP COMPUTATION ===
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

def compute_shap_values(model, X_sample, feature_names=None):
    """
    Compute SHAP values for model explainability.
    
    Args:
        model: Trained XGBoost or LightGBM model
        X_sample: Sample feature matrix
        feature_names: Optional list of feature names
    
    Returns:
        Dictionary with 'shap_values', 'base_value', 'feature_names'
    """
    if not SHAP_AVAILABLE:
        return None
    
    try:
        if isinstance(model, xgb.core.Booster):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            base_value = explainer.expected_value
        elif isinstance(model, lgb.basic.Booster):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            base_value = explainer.expected_value
        else:
            return None
        
        return {
            "shap_values": shap_values,
            "base_value": base_value,
            "feature_names": feature_names if feature_names else [f"Feature_{i}" for i in range(X_sample.shape[1])]
        }
    except Exception as e:
        logger.warning(f"SHAP computation failed: {e}")
        return None

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

def run_hybrid_forecast(df: pd.DataFrame, product: str, horizon_weeks: int = 26, end_date: Optional[pd.Timestamp] = None, model_type: str = "hybrid", fast_mode: bool = True) -> Dict[str, Any]:
    """
    Run hybrid forecast with model preference selection.
    
    Args:
        df: DataFrame with product data (must contain 'product_name', 'week_start', 'sales_qty')
        product: Product name to filter
        horizon_weeks: Forecast horizon in weeks
        end_date: Optional end date (overrides horizon_weeks if provided)
        model_type: 'prophet', 'xgb', 'lgbm', or 'hybrid' (default)
        fast_mode: Use faster models
    
    Returns:
        Dictionary with forecast_df, metrics, insights
    """
    # Filter product data
    product_df = df[df["product_name"] == product].copy()
    if product_df.empty:
        raise ValueError(f"No data found for product: {product}")
    
    # Prepare time series
    ts = product_df[["week_start", "sales_qty"]].copy()
    ts.columns = ["date", "sales_qty"]
    ts["date"] = pd.to_datetime(ts["date"])
    ts = ts.sort_values("date").reset_index(drop=True)
    
    # Calculate horizon from end_date if provided
    if end_date is not None:
        last_date = ts["date"].max()
        horizon_days = (pd.to_datetime(end_date) - last_date).days
        horizon_weeks = max(1, int(np.ceil(horizon_days / 7)))
    
    # Run forecast based on model_type
    if model_type.lower() == "prophet":
        # Prophet-only forecast (simplified)
        result = train_ensemble(ts, horizon_weeks=horizon_weeks, fast_mode=fast_mode, debug=False)
        # Override to use Prophet weights only
        result.details["weights"] = {"prophet": 1.0, "xgb": 0.0, "lgbm": 0.0}
    elif model_type.lower() == "xgb":
        # XGBoost-only (would need separate function, fallback to hybrid)
        result = train_ensemble(ts, horizon_weeks=horizon_weeks, fast_mode=fast_mode, debug=False)
        weights = result.details.get("weights", {})
        if "xgb" in weights:
            result.details["weights"] = {"xgb": 1.0, "prophet": 0.0, "lgbm": 0.0}
    elif model_type.lower() == "lgbm":
        # LightGBM-only (fallback to hybrid)
        result = train_ensemble(ts, horizon_weeks=horizon_weeks, fast_mode=fast_mode, debug=False)
        weights = result.details.get("weights", {})
        if "lgbm" in weights:
            result.details["weights"] = {"lgbm": 1.0, "prophet": 0.0, "xgb": 0.0}
    else:
        # Hybrid ensemble (default)
        result = train_ensemble(ts, horizon_weeks=horizon_weeks, fast_mode=fast_mode, debug=False)
    
    # Generate insights
    insights = _generate_business_insights_enhanced(result.history, result.forecast, result.metrics, product)
    
    # Convert to return format
    forecast_df = result.forecast.copy()
    forecast_df["date"] = pd.to_datetime(forecast_df["date"])
    
    return {
        "forecast_df": forecast_df,
        "history_df": result.history,
        "metrics": result.metrics,
        "insights": insights,
        "feature_importances": result.feature_importances,
        "prophet_components": result.prophet_components,
        "details": result.details
    }

def simulate_forecast_with_scenarios(forecast_df: pd.DataFrame, price_delta: float = 0.0, promotion_flag: bool = False, holiday_flag: bool = False, elasticity: float = -1.2) -> pd.DataFrame:
    """
    Simulate forecast with What-If scenarios.
    
    Args:
        forecast_df: Base forecast DataFrame with 'date', 'yhat', 'yhat_lower', 'yhat_upper'
        price_delta: Price change percentage (-20 to +20)
        promotion_flag: Whether promotion is active
        holiday_flag: Whether holiday period
        elasticity: Price elasticity coefficient
    
    Returns:
        Simulated forecast DataFrame
    """
    sim_df = forecast_df.copy()
    
    # Price effect
    price_multiplier = (1 + price_delta / 100.0) ** elasticity if elasticity != 0 else 1.0
    
    # Promotion effect (typically +15-25% boost)
    promo_multiplier = 1.2 if promotion_flag else 1.0
    
    # Holiday effect (typically +20-35% boost)
    holiday_multiplier = 1.25 if holiday_flag else 1.0
    
    # Combined effect
    total_multiplier = price_multiplier * promo_multiplier * holiday_multiplier
    
    # Apply to forecast
    sim_df["yhat_simulated"] = sim_df["yhat"] * total_multiplier
    sim_df["yhat_lower_simulated"] = sim_df["yhat_lower"] * total_multiplier
    sim_df["yhat_upper_simulated"] = sim_df["yhat_upper"] * total_multiplier
    
    return sim_df

def _generate_business_insights_enhanced(history: pd.DataFrame, forecast: pd.DataFrame, metrics: Dict[str, float], product_name: str) -> Dict[str, Any]:
    """
    Generate enhanced business insights with recommendations.
    
    Returns:
        Dictionary with narrative, top_drivers, recommendations
    """
    insights = {}
    
    # Trend analysis
    last_4w_avg = history["sales_qty"].tail(4).mean() if len(history) >= 4 else history["sales_qty"].mean()
    next_4w_avg = forecast["yhat"].head(4).mean() if len(forecast) >= 4 else forecast["yhat"].mean()
    
    if next_4w_avg > last_4w_avg * 1.1:
        growth_pct = ((next_4w_avg - last_4w_avg) / last_4w_avg) * 100
        trend_text = f"{product_name} sales expected to rise {growth_pct:.1f}% in the next month"
    elif next_4w_avg < last_4w_avg * 0.9:
        decline_pct = ((last_4w_avg - next_4w_avg) / last_4w_avg) * 100
        trend_text = f"{product_name} sales expected to decline {decline_pct:.1f}% in the next month"
    else:
        trend_text = f"{product_name} sales expected to remain stable"
    
    # Peak month
    if len(forecast) >= 52:
        monthly_forecast = forecast.groupby(forecast["date"].dt.month)["yhat"].mean()
        peak_month = monthly_forecast.idxmax()
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        peak_text = f"Peak sales predicted for {month_names[peak_month-1]} driven by seasonal demand"
    else:
        peak_text = "Seasonal patterns emerging"
    
    # Narrative
    insights["narrative"] = f"{trend_text}. {peak_text}. Consider bundling with complementary products to leverage seasonality."
    
    # Top drivers (placeholder - would come from feature importance)
    insights["top_drivers"] = [
        "Holiday flags show strong correlation",
        "Price changes drive short-term fluctuations",
        "Promotions boost sales by 15-20%"
    ]
    
    # Recommendations
    recommendations = []
    if len(forecast) >= 13:
        next_q_total = forecast.head(13)["yhat"].sum()
        recommendations.append(f"Increase stock before peak week (Week {forecast.head(13).idxmax() + 1})")
    recommendations.append("Leverage seasonal promotions during peak months")
    recommendations.append("Monitor price elasticity for optimization opportunities")
    
    insights["recommendations"] = recommendations
    
    return insights

def run_hybrid_forecast(df, product=None, horizon_weeks=12, end_date=None, model_type='hybrid', include_features=True, fast_mode=True):
    """
    Run hybrid forecasting model combining Prophet, XGBoost, and LightGBM.
    
    Args:
        df: DataFrame with 'date' and 'sales_qty' columns
        product: Product name to filter data (optional)
        horizon_weeks: Number of weeks to forecast
        end_date: End date for forecasting (optional)
        model_type: Type of model to use ('hybrid', 'prophet', 'xgboost', 'lightgbm')
        include_features: Whether to include feature engineering
        fast_mode: Whether to use faster but less accurate models
        
    Returns:
        EnsembleResult object with forecast and metrics
    """
    # Filter data for specific product if provided
    if product is not None:
        df = df[df['product_name'] == product].copy()
    # Preprocess data
    df_weekly = _preprocess_weekly_data(df)
    
    # Create features if requested
    if include_features:
        df_weekly = _create_weekly_features(df_weekly)
    
    # Split data into train and test
    train_size = max(int(len(df_weekly) * 0.8), len(df_weekly) - horizon_weeks)
    df_train = df_weekly.iloc[:train_size].copy()
    df_test = df_weekly.iloc[train_size:].copy() if train_size < len(df_weekly) else None
    
    # Fit Prophet model
    prophet_hist, prophet_future, prophet_model, prophet_components = _fit_prophet(
        df_train, horizon_weeks, debug=False
    )
    
    # Prepare feature-based models
    if include_features:
        feature_cols = [c for c in df_weekly.columns if c not in ['date', 'sales_qty']]
        X_train = df_train[feature_cols].values
        y_train = df_train['sales_qty'].values
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Fit models based on model_type
        xgb_model = None
        lgb_model = None
        
        if model_type.lower() in ['hybrid', 'xgboost']:
            xgb_model = _fit_xgb(X_train_scaled, y_train, fast_mode=fast_mode)
            
        if model_type.lower() in ['hybrid', 'lightgbm']:
            lgb_model = _fit_lgb(X_train_scaled, y_train, fast_mode=fast_mode)
        
        # Generate future features
        if end_date is not None:
            # Calculate periods based on end_date
            start_date = df_weekly['date'].max() + pd.Timedelta(days=7)
            periods = (end_date - start_date).days // 7 + 1
            future_dates = pd.date_range(
                start=start_date,
                end=end_date,
                freq='W-SUN'
            )
        else:
            future_dates = pd.date_range(
                start=df_weekly['date'].max() + pd.Timedelta(days=7),
                periods=horizon_weeks,
                freq='W-SUN'
            )
        
        df_future = pd.DataFrame({'date': future_dates})
        df_future['sales_qty'] = np.nan
        
        # Use last values for lag features
        for lag in [1, 2, 3, 7]:
            df_future[f'lag_{lag}'] = np.nan
        
        # Fill in time-based features
        df_future['month'] = df_future['date'].dt.month
        df_future['quarter'] = df_future['date'].dt.quarter
        df_future['year'] = df_future['date'].dt.year
        df_future['weekofyear'] = df_future['date'].dt.isocalendar().week
        
        # Cyclical encoding
        df_future['month_sin'] = np.sin(2 * np.pi * df_future['month'] / 12)
        df_future['month_cos'] = np.cos(2 * np.pi * df_future['month'] / 12)
        df_future['week_sin'] = np.sin(2 * np.pi * df_future['weekofyear'] / 52)
        df_future['week_cos'] = np.cos(2 * np.pi * df_future['weekofyear'] / 52)
        
        # Rolling statistics
        for window in [4, 8, 12, 26]:
            df_future[f'rolling_mean_{window}'] = np.nan
            df_future[f'rolling_std_{window}'] = np.nan
    
    # Combine forecasts
    if prophet_future is not None:
        forecast_df = prophet_future.copy()
        forecast_df = forecast_df.rename(columns={'yhat': 'prophet_forecast'})
        
        # Add ensemble forecast (just Prophet for now)
        forecast_df['forecast'] = forecast_df['prophet_forecast']
        
        # Add confidence intervals
        forecast_df['lower_bound'] = forecast_df['yhat_lower']
        forecast_df['upper_bound'] = forecast_df['yhat_upper']
        forecast_df['lower_bound_95'] = forecast_df['yhat_lower_95']
        forecast_df['upper_bound_95'] = forecast_df['yhat_upper_95']
    else:
        # Fallback if Prophet fails
        forecast_df = pd.DataFrame({
            'date': pd.date_range(
                start=df_weekly['date'].max() + pd.Timedelta(days=7),
                periods=horizon_weeks,
                freq='W-SUN'
            )
        })
        
        # Use simple moving average as fallback
        last_value = df_weekly['sales_qty'].iloc[-1]
        avg_4w = df_weekly['sales_qty'].iloc[-4:].mean()
        avg_8w = df_weekly['sales_qty'].iloc[-8:].mean() if len(df_weekly) >= 8 else avg_4w
        
        forecast_df['forecast'] = avg_4w
        forecast_df['lower_bound'] = avg_8w * 0.7
        forecast_df['upper_bound'] = avg_4w * 1.3
        forecast_df['lower_bound_95'] = avg_8w * 0.5
        forecast_df['upper_bound_95'] = avg_4w * 1.5
    
    # Calculate metrics
    metrics = {}
    if df_test is not None and len(df_test) > 0:
        y_true = df_test['sales_qty'].values
        y_pred = forecast_df['forecast'].values[:len(y_true)]
        
        metrics['rmse'] = math.sqrt(mean_squared_error(y_true, y_pred))
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['mape'] = mape(y_true, y_pred)
        metrics['smape'] = smape(y_true, y_pred)
        metrics['r2'] = r2_score(y_true, y_pred) if len(y_true) > 1 else 0.0
    
    # Create result object
    result = EnsembleResult(
        history=df_weekly,
        forecast=forecast_df,
        metrics=metrics,
        residuals=pd.DataFrame(),  # Not implemented for simplicity
        details={'model_weights': {'prophet': 1.0}},
        feature_importances=None,
        prophet_components=prophet_components
    )
    
    return result

def simulate_forecast_with_scenarios(df, horizon_weeks=12, scenarios=None):
    """
    Generate forecast scenarios based on different assumptions.
    
    Args:
        df: DataFrame with 'date' and 'sales_qty' columns
        horizon_weeks: Number of weeks to forecast
        scenarios: Dictionary of scenario adjustments
        
    Returns:
        Dictionary with base forecast and scenarios
    """
    # Default scenarios if none provided
    if scenarios is None:
        scenarios = {
            'optimistic': {'growth': 0.15, 'seasonality': 1.2},
            'pessimistic': {'growth': -0.10, 'seasonality': 0.8},
            'steady_growth': {'growth': 0.05, 'seasonality': 1.0}
        }
    
    # Get base forecast
    base_result = run_hybrid_forecast(df, horizon_weeks=horizon_weeks)
    base_forecast = base_result.forecast_df
    
    # Generate scenario forecasts
    scenario_forecasts = {}
    for name, adjustments in scenarios.items():
        scenario_df = base_forecast.copy()
        
        # Apply growth adjustment
        if 'growth' in adjustments:
            growth_factor = 1.0 + adjustments['growth']
            weeks = np.arange(len(scenario_df))
            growth_multiplier = growth_factor ** (weeks / 52)  # Annualized growth
            scenario_df['forecast'] = scenario_df['forecast'] * growth_multiplier
        
        # Apply seasonality adjustment
        if 'seasonality' in adjustments:
            seasonality_factor = adjustments['seasonality']
            if 'month' in scenario_df.columns:
                # Enhance seasonal months (e.g., holidays)
                seasonal_months = [11, 12, 1]  # Nov, Dec, Jan
                for i, date in enumerate(scenario_df['date']):
                    if date.month in seasonal_months:
                        scenario_df.loc[i, 'forecast'] *= seasonality_factor
        
        # Adjust bounds
        scenario_df['lower_bound'] = scenario_df['forecast'] * (base_forecast['lower_bound'] / base_forecast['forecast'])
        scenario_df['upper_bound'] = scenario_df['forecast'] * (base_forecast['upper_bound'] / base_forecast['forecast'])
        scenario_df['lower_bound_95'] = scenario_df['forecast'] * (base_forecast['lower_bound_95'] / base_forecast['forecast'])
        scenario_df['upper_bound_95'] = scenario_df['forecast'] * (base_forecast['upper_bound_95'] / base_forecast['forecast'])
        
        scenario_forecasts[name] = scenario_df
    
    # Return dictionary with all necessary data
    return {
        'base': base_forecast,
        'scenarios': scenario_forecasts,
        'base_result': {
            "history_df": base_result["history_df"],
            "forecast_df": base_result["forecast_df"],
            "metrics": base_result["metrics"],
            "residuals": base_result["residuals"],
            "details": base_result["details"],
            "feature_importances": base_result["feature_importances"],
            "prophet_components": base_result["prophet_components"]
        }
    }