# utils/business_insights.py
"""
Placement-Grade Business Intelligence Module
Handles: Anomaly Detection, Inventory Optimization, Seasonal Analysis, Pricing Intelligence
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# ============================================================================
# PART 2: Sales Anomaly Detection
# ============================================================================

def detect_sales_anomalies(df: pd.DataFrame, product_name: str, method: str = "hybrid") -> pd.DataFrame:
    """
    Hybrid anomaly detection using Z-score, IQR, and Isolation Forest.
    
    Returns DataFrame with columns: date, actual_sales, expected_sales, deviation_pct, severity, suggested_action
    """
    product_df = df[df["product_name"] == product_name].copy()
    if len(product_df) < 10:
        return pd.DataFrame(columns=["date", "actual_sales", "expected_sales", "deviation_pct", "severity", "suggested_action"])
    
    product_df = product_df.sort_values("week_start").reset_index(drop=True)
    product_df["week_start"] = pd.to_datetime(product_df["week_start"])
    
    # Calculate expected sales (rolling mean with seasonality)
    window = min(8, len(product_df) // 2)
    product_df["expected_sales"] = product_df["sales_qty"].rolling(window=window, center=True).mean()
    product_df["expected_sales"] = product_df["expected_sales"].fillna(method="ffill").fillna(method="bfill")
    
    # Method 1: Z-score
    mean_sales = product_df["sales_qty"].mean()
    std_sales = product_df["sales_qty"].std()
    product_df["z_score"] = (product_df["sales_qty"] - mean_sales) / (std_sales + 1e-6)
    
    # Method 2: IQR
    Q1 = product_df["sales_qty"].quantile(0.25)
    Q3 = product_df["sales_qty"].quantile(0.75)
    IQR = Q3 - Q1
    product_df["iqr_lower"] = Q1 - 1.5 * IQR
    product_df["iqr_upper"] = Q3 + 1.5 * IQR
    
    # Method 3: Isolation Forest
    if method == "hybrid" and len(product_df) >= 20:
        features = product_df[["sales_qty", "price", "stock_on_hand"]].fillna(0)
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        product_df["iso_outlier"] = iso_forest.fit_predict(features)
    else:
        product_df["iso_outlier"] = 1  # No anomaly
    
    # Combine methods
    product_df["is_anomaly"] = (
        (product_df["z_score"].abs() > 2.5) |
        (product_df["sales_qty"] < product_df["iqr_lower"]) |
        (product_df["sales_qty"] > product_df["iqr_upper"]) |
        (product_df["iso_outlier"] == -1)
    )
    
    # Calculate deviation
    product_df["deviation_pct"] = ((product_df["sales_qty"] - product_df["expected_sales"]) / (product_df["expected_sales"] + 1e-6)) * 100
    
    # Severity scoring
    product_df["severity"] = "normal"
    product_df.loc[product_df["deviation_pct"].abs() > 50, "severity"] = "severe"
    product_df.loc[(product_df["deviation_pct"].abs() > 25) & (product_df["deviation_pct"].abs() <= 50), "severity"] = "moderate"
    product_df.loc[(product_df["deviation_pct"].abs() > 10) & (product_df["deviation_pct"].abs() <= 25), "severity"] = "mild"
    
    # Generate suggested actions
    product_df["suggested_action"] = "Monitor"
    product_df.loc[(product_df["is_anomaly"]) & (product_df["deviation_pct"] < -30), "suggested_action"] = "Investigate supply chain / Check stock levels"
    product_df.loc[(product_df["is_anomaly"]) & (product_df["deviation_pct"] > 30), "suggested_action"] = "Increase inventory / Capitalize on demand surge"
    product_df.loc[(product_df["is_anomaly"]) & (product_df["severity"] == "severe"), "suggested_action"] = "URGENT: Review pricing strategy & supply"
    
    anomalies = product_df[product_df["is_anomaly"]].copy()
    
    return pd.DataFrame({
        "date": anomalies["week_start"],
        "actual_sales": anomalies["sales_qty"],
        "expected_sales": anomalies["expected_sales"],
        "deviation_pct": anomalies["deviation_pct"].round(2),
        "severity": anomalies["severity"],
        "suggested_action": anomalies["suggested_action"]
    })


# ============================================================================
# PART 3: Inventory Alerts
# ============================================================================

def generate_inventory_alerts(df: pd.DataFrame, horizon_weeks: int = 4) -> pd.DataFrame:
    """
    Predict demand and flag inventory issues.
    Returns DataFrame with: product_name, stock, predicted_demand, status, days_to_stockout, suggested_reorder_qty
    """
    from sklearn.ensemble import GradientBoostingRegressor
    
    alerts = []
    
    for product in df["product_name"].unique():
        product_df = df[df["product_name"] == product].copy()
        product_df = product_df.sort_values("week_start").reset_index(drop=True)
        
        if len(product_df) < 8:
            continue
        
        # Simple demand prediction (XGBoost-style features)
        product_df["lag_1"] = product_df["sales_qty"].shift(1)
        product_df["lag_2"] = product_df["sales_qty"].shift(2)
        product_df["ma_4"] = product_df["sales_qty"].rolling(4).mean()
        
        # Train model
        train_df = product_df.dropna()
        if len(train_df) < 5:
            continue
        
        X = train_df[["lag_1", "lag_2", "ma_4", "price", "promotion"]].fillna(0)
        y = train_df["sales_qty"]
        
        try:
            model = GradientBoostingRegressor(n_estimators=50, random_state=42)
            model.fit(X, y)
            
            # Predict next week
            last_row = product_df.iloc[-1]
            next_features = np.array([[
                last_row["sales_qty"] if not pd.isna(last_row["sales_qty"]) else last_row["ma_4"],
                product_df.iloc[-2]["sales_qty"] if len(product_df) >= 2 else last_row["ma_4"],
                last_row["ma_4"] if not pd.isna(last_row["ma_4"]) else product_df["sales_qty"].mean(),
                last_row["price"] if not pd.isna(last_row["price"]) else product_df["price"].mean(),
                0  # No promotion assumed
            ]])
            predicted_demand = max(0, model.predict(next_features)[0])
            
            current_stock = last_row["stock_on_hand"] if not pd.isna(last_row["stock_on_hand"]) else 0
            
            # Status determination
            if current_stock < predicted_demand * 1.1:
                status = "🔴 Low Stock"
                days_to_stockout = max(1, int((current_stock / (predicted_demand + 1e-6)) * 7))
                suggested_reorder = int(predicted_demand * horizon_weeks * 1.5)
            elif current_stock > predicted_demand * 2:
                status = "🟡 Overstock"
                days_to_stockout = None
                suggested_reorder = 0
            else:
                status = "🟢 Optimal"
                days_to_stockout = None
                suggested_reorder = int(max(0, predicted_demand * horizon_weeks - current_stock))
            
            alerts.append({
                "product_name": product,
                "stock": int(current_stock),
                "predicted_demand": round(predicted_demand, 1),
                "status": status,
                "days_to_stockout": days_to_stockout,
                "suggested_reorder_qty": suggested_reorder
            })
        except Exception as e:
            continue
    
    return pd.DataFrame(alerts)


# ============================================================================
# PART 4: Seasonal Insights
# ============================================================================

def analyze_seasonality(df: pd.DataFrame, product_name: Optional[str] = None) -> Dict:
    """
    Decompose time series and identify seasonal patterns.
    """
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    analysis_df = df.copy()
    if product_name:
        analysis_df = analysis_df[analysis_df["product_name"] == product_name]
    
    if len(analysis_df) < 52:
        return {}
    
    analysis_df = analysis_df.sort_values("week_start").reset_index(drop=True)
    analysis_df["week_start"] = pd.to_datetime(analysis_df["week_start"])
    analysis_df = analysis_df.set_index("week_start")
    
    # Aggregate by week
    weekly = analysis_df.groupby("week_start")["sales_qty"].sum().resample("W").sum()
    
    # Decompose
    try:
        decomp = seasonal_decompose(weekly, model="multiplicative", period=min(52, len(weekly) // 2))
        
        # Monthly seasonality
        analysis_df["month"] = analysis_df.index.month
        monthly_avg = analysis_df.groupby("month")["sales_qty"].mean()
        peak_month = monthly_avg.idxmax()
        low_month = monthly_avg.idxmin()
        
        # Seasonal correlation
        seasonal_corr = {}
        if "promotion" in analysis_df.columns:
            seasonal_corr["promotion"] = analysis_df["sales_qty"].corr(analysis_df["promotion"])
        if "holiday_flag" in analysis_df.columns:
            seasonal_corr["holiday"] = analysis_df["sales_qty"].corr(analysis_df["holiday_flag"])
        
        return {
            "trend": decomp.trend,
            "seasonal": decomp.seasonal,
            "residual": decomp.resid,
            "peak_month": peak_month,
            "low_month": low_month,
            "monthly_pattern": monthly_avg.to_dict(),
            "correlations": seasonal_corr
        }
    except Exception:
        return {}


# ============================================================================
# PART 5: Pricing Opportunities (Elasticity)
# ============================================================================

def calculate_price_elasticity(df: pd.DataFrame, product_name: str) -> float:
    """
    Calculate price elasticity: E = (%ΔQ) / (%ΔP)
    Returns elasticity value (typically negative)
    """
    product_df = df[df["product_name"] == product_name].copy()
    product_df = product_df.sort_values("week_start").reset_index(drop=True)
    
    if len(product_df) < 12 or product_df["price"].nunique() < 3:
        return -1.2  # Default moderate elasticity
    
    # Calculate percentage changes
    product_df["pct_change_price"] = product_df["price"].pct_change()
    product_df["pct_change_quantity"] = product_df["sales_qty"].pct_change()
    
    # Remove zeros and extreme outliers
    valid = product_df[
        (product_df["pct_change_price"].abs() > 0.01) &
        (product_df["pct_change_quantity"].abs() < 2.0) &
        (product_df["pct_change_price"].abs() < 1.0)
    ].dropna(subset=["pct_change_price", "pct_change_quantity"])
    
    if len(valid) < 5:
        return -1.2
    
    # Elasticity = slope of log-log regression
    log_price = np.log(valid["price"].values + 1e-6)
    log_quantity = np.log(valid["sales_qty"].values + 1e-6)
    
    try:
        elasticity = np.polyfit(log_price, log_quantity, 1)[0]
        return float(elasticity)
    except Exception:
        return -1.2


def analyze_pricing_opportunities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify pricing opportunities for all products.
    """
    opportunities = []
    
    for product in df["product_name"].unique():
        elasticity = calculate_price_elasticity(df, product)
        product_df = df[df["product_name"] == product]
        
        avg_price = product_df["price"].mean()
        avg_quantity = product_df["sales_qty"].mean()
        current_revenue = avg_price * avg_quantity
        
        # Price recommendation
        if elasticity < -1.0:  # Elastic (price-sensitive)
            recommendation = "Consider price reduction"
            suggested_change = -5  # 5% reduction
        elif elasticity > -0.5:  # Inelastic
            recommendation = "Price increase opportunity"
            suggested_change = 5  # 5% increase
        else:
            recommendation = "Maintain current pricing"
            suggested_change = 0
        
        # Revenue simulation
        new_price = avg_price * (1 + suggested_change / 100)
        new_quantity = avg_quantity * (1 + (elasticity * suggested_change / 100))
        new_revenue = new_price * new_quantity
        revenue_gain = new_revenue - current_revenue
        
        opportunities.append({
            "product_name": product,
            "current_price": round(avg_price, 2),
            "elasticity": round(elasticity, 2),
            "recommendation": recommendation,
            "suggested_change_pct": suggested_change,
            "current_revenue": round(current_revenue, 2),
            "projected_revenue": round(new_revenue, 2),
            "revenue_gain": round(revenue_gain, 2)
        })
    
    return pd.DataFrame(opportunities).sort_values("revenue_gain", ascending=False)


# ============================================================================
# PART 6: Dynamic Pricing Engine
# ============================================================================

def optimize_price(current_price: float, elasticity: float, margin: float = 0.3, cost: float = None) -> Dict:
    """
    Calculate optimal price using elasticity and margin.
    """
    if cost is None:
        cost = current_price * (1 - margin)
    
    # Optimal price formula: P* = cost / (1 + 1/elasticity)
    try:
        if elasticity < -1:  # Elastic
            optimal_price = cost / (1 + 1/abs(elasticity))
        else:  # Inelastic - can price higher
            optimal_price = current_price * 1.1  # Conservative 10% increase
        
        optimal_price = max(cost * 1.1, optimal_price)  # Ensure minimum margin
        
        expected_quantity_change = elasticity * ((optimal_price - current_price) / current_price) * 100
        new_quantity = 1 + (expected_quantity_change / 100)
        
        profit_old = (current_price - cost) * 1.0
        profit_new = (optimal_price - cost) * new_quantity
        profit_gain = profit_new - profit_old
        
        return {
            "current_price": current_price,
            "optimal_price": round(optimal_price, 2),
            "price_change_pct": round(((optimal_price - current_price) / current_price) * 100, 1),
            "expected_quantity_change_pct": round(expected_quantity_change, 1),
            "profit_gain_pct": round((profit_gain / profit_old) * 100, 1) if profit_old > 0 else 0,
            "profit_gain_abs": round(profit_gain, 2)
        }
    except Exception:
        return {
            "current_price": current_price,
            "optimal_price": current_price,
            "price_change_pct": 0,
            "expected_quantity_change_pct": 0,
            "profit_gain_pct": 0,
            "profit_gain_abs": 0
        }


# ============================================================================
# PART 7: Executive Summary Generator
# ============================================================================

def generate_executive_summary(
    anomalies_count: int,
    low_stock_count: int,
    pricing_opportunities: pd.DataFrame,
    total_revenue_gain: float
) -> str:
    """
    Auto-generate executive summary text.
    """
    summary = f"⚡ **RetailSense AI Insights Summary**\n\n"
    summary += f"📊 **Problem Detection:**\n"
    summary += f"- {anomalies_count} sales anomalies detected requiring investigation\n"
    summary += f"- {low_stock_count} products flagged for low stock alerts\n\n"
    
    summary += f"💰 **Revenue Opportunities:**\n"
    summary += f"- {len(pricing_opportunities)} products identified for pricing optimization\n"
    summary += f"- Total projected revenue gain: ₹{total_revenue_gain:,.0f} next quarter\n\n"
    
    summary += f"🎯 **Recommended Actions:**\n"
    summary += f"1. Address {low_stock_count} stock-out risks immediately\n"
    summary += f"2. Review {anomalies_count} anomaly cases for root causes\n"
    summary += f"3. Implement dynamic pricing for top {min(5, len(pricing_opportunities))} high-opportunity products\n"
    
    return summary