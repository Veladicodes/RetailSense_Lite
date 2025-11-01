# RetailSense Lite - Placement Demo Guide

## 🎯 Overview

RetailSense Lite is a **placement-ready retail analytics dashboard** showcasing advanced ML forecasting, anomaly detection, inventory optimization, and dynamic pricing capabilities.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Streamlit
- Required packages: `pandas`, `numpy`, `plotly`, `prophet`, `xgboost`, `lightgbm`, `scikit-learn`

### Running the Dashboard

```bash
# Activate virtual environment (if using)
# source retailsense_env/bin/activate  # Linux/Mac
# retailsense_env\Scripts\activate     # Windows

# Run Streamlit app
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`

## 📊 Key Features for Placement Demo

### 1. 📈 Forecast Explorer (Main Feature)

**Location:** Sales Forecasting tab → Forecast Explorer sub-tab

**Highlights:**
- **Hybrid Ensemble Model**: Prophet + XGBoost + LightGBM
- **Multi-horizon forecasting**: 3 months, 6 months, 1 year, 3 years, or custom date
- **Confidence Intervals**: 80% and 95% CI bands
- **Anomaly Overlay**: Red X markers on historical anomalies
- **Changepoint Detection**: Yellow dotted lines from Prophet
- **Comprehensive KPIs**: Next week, Growth %, Peak/Min months, R², Stockout risk
- **Explain Forecast Drivers**: Feature importance with SHAP fallback
- **What-if Pricing**: Interactive price slider with elasticity simulation
- **Export Options**: CSV, JSON metrics, Insight text

**What to Show:**
1. Select a product (e.g., "Milk" or "Bread")
2. Choose forecast horizon (6 months recommended for demo)
3. Click "Generate 3-Year Forecast"
4. Highlight the dual CI bands and anomaly detection
5. Expand "Explain Forecast Drivers" to show feature importance
6. Show "Pricing Simulation" with price elasticity effects

### 2. 🚨 Sales Anomalies

**Location:** Sales Forecasting tab → Sales Anomalies sub-tab

**Highlights:**
- Hybrid detection: Z-score + IQR + Isolation Forest
- Severity classification: Mild/Moderate/Severe
- Suggested actions for each anomaly
- Interactive timeline visualization

### 3. 📦 Inventory Alerts

**Location:** Sales Forecasting tab → Inventory Alerts sub-tab

**Highlights:**
- Predictive demand forecasting
- Low stock / Overstock / Optimal status
- Days to stockout calculation
- Suggested reorder quantities

### 4. 🌦️ Seasonal Insights

**Location:** Sales Forecasting tab → Seasonal Insights sub-tab

**Highlights:**
- Trend decomposition
- Monthly seasonality patterns
- Correlation analysis

### 5. 💰 Pricing Opportunities

**Location:** Sales Forecasting tab → Pricing Opportunities sub-tab

**Highlights:**
- Price elasticity calculation
- Revenue gain projections
- Optimal price suggestions

### 6. ⚙️ Dynamic Pricing Engine

**Location:** Sales Forecasting tab → Dynamic Pricing Engine sub-tab

**Highlights:**
- AI-driven price optimization
- Profit gain calculations
- Multi-product analysis

## 📁 Dataset

**Primary Data Source:** `F:\RetailSense_Lite\data\processed\data_with_all_features.csv`

**Required Columns:**
- `week_start` (date)
- `product_name` (string)
- `sales_qty` (numeric)
- `price` (optional, for pricing features)
- `stock_on_hand` (optional, for inventory features)

## 🔧 Technical Stack

### Models Used
- **Prophet**: Long-term seasonality and trend
- **XGBoost**: Short-term pattern learning
- **LightGBM**: Feature-based residual learning
- **Isolation Forest**: Anomaly detection
- **GradientBoostingRegressor**: Inventory demand prediction

### Output Files
All outputs saved to `F:\RetailSense_Lite\outputs\`:
- `forecasting_results.csv`
- `business_sales_anomalies.csv`
- `business_inventory_alerts.csv`
- `business_seasonal_insights.csv`
- `business_pricing_opportunities.csv`

## 🎤 Presentation Tips

### Opening (30 seconds)
- "This dashboard converts raw sales data into actionable business insights"
- Show the clean UI with dark theme and professional styling

### Forecast Explorer Demo (2 minutes)
1. **Select Product**: "Let me show you forecasting for Milk"
2. **Generate Forecast**: Click button, show loading spinner
3. **Interpret Results**: Point out:
   - KPI cards (growth %, peak month, R²)
   - Dual CI bands ("80% and 95% confidence intervals")
   - Anomalies ("Red X marks show unusual sales patterns")
   - Changepoints ("Yellow lines indicate trend shifts")

4. **Explain Drivers**: Expand to show feature importance
5. **Pricing Simulation**: Adjust slider to show revenue impact

### Key Talking Points
- **Technical Depth**: "Hybrid ensemble combining 3 models for robustness"
- **Business Value**: "Identified 3 stockouts and 4 pricing opportunities"
- **Scalability**: "Works with any retail dataset with minimal configuration"
- **Interpretability**: "SHAP-style feature importance explains predictions"

## ✅ Test Results

Run the test script to verify:
```bash
python utils/tests/test_forecast_flow.py
```

**Expected Output:**
- ✅ Forecast file generated
- ✅ Ensemble RMSE/MAE calculated
- ✅ Feature importances extracted
- ✅ CSV saved correctly

**Sample Test Results:**
- Sample product: Apples
- Ensemble RMSE: ~38.85
- Ensemble MAE: ~29.09
- Ensemble R²: ~0.77
- Forecast rows: 13 (for 13-week horizon)

## 🔍 Troubleshooting

### "Dataset not found" error
- Run Full Pipeline (Phase 1 + Phase 2)
- Check that `data_with_all_features.csv` exists
- Verify file is not corrupted

### Forecast fails
- Ensure product has ≥8 weeks of history
- Try Fast Mode (recommended for demo)
- Check console for specific error messages

### SHAP not available
- Install: `pip install shap`
- Otherwise, app falls back to permutation importance
- Message shown in UI: "SHAP not installed"

## 📝 Code Structure

### Key Files
- `app.py`: Main Streamlit dashboard
- `utils/advanced_forecasting.py`: Hybrid ensemble model
- `utils/business_insights.py`: Anomaly, inventory, pricing logic
- `utils/tests/test_forecast_flow.py`: Test script

### Key Functions
- `train_ensemble_for_app()`: App-friendly wrapper for forecasting
- `detect_sales_anomalies()`: Hybrid anomaly detection
- `generate_inventory_alerts()`: Stock optimization
- `calculate_price_elasticity()`: Pricing intelligence

## 💡 One-Line Auto Insight Example

**Executive Summary Format:**
> "⚡ RetailSense AI detected 3 anomalies, 2 stock-outs, and identified 4 high-margin price opportunities. Expected overall profit gain: +₹3.7 L next quarter."

**Forecast Insight Format:**
> "Milk sales expected to grow +14% in Q1 2026 driven by seasonality recovery after monsoon. Peak sales predicted for January. Model accuracy (R²): 0.774."

## 🎓 Interview Preparation

### Questions to Prepare For

**Technical:**
1. "How does the ensemble model work?"
   - Answer: Prophet handles seasonality, XGBoost/LightGBM learn patterns, weighted blend by validation RMSE

2. "What's your approach to anomaly detection?"
   - Answer: Hybrid method combining statistical (Z-score, IQR) with ML (Isolation Forest) for robustness

3. "How do you handle missing data?"
   - Answer: Forward-fill for temporal data, interpolation for gaps, robust to missing values

**Business:**
1. "What's the business impact?"
   - Answer: Prevents stockouts, optimizes pricing, identifies opportunities for 3-5% revenue lift

2. "How would you scale this?"
   - Answer: Modular design, caching, batch processing, cloud deployment ready

## 📞 Support

For issues or questions during placement:
1. Check test output: `python utils/tests/test_forecast_flow.py`
2. Verify data file exists and is readable
3. Try Fast Mode for faster demo times
4. Check console/terminal for detailed error messages

---

**Last Updated:** 2025-11-01
**Version:** 1.0 (Placement-Ready)

