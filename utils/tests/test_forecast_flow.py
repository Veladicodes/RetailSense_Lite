"""
Test script for forecast flow verification.
Exercises train_ensemble_for_app wrapper and verifies output CSV generation.
"""
import pandas as pd
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from utils.advanced_forecasting import train_ensemble_for_app

DATA_PATH = r"F:\RetailSense_Lite\data\processed\data_with_all_features.csv"
OUT_PATH = r"F:\RetailSense_Lite\outputs\forecasting_results.csv"

def test_flow():
    """Test the forecast flow end-to-end"""
    print("=" * 60)
    print("Testing Forecast Flow")
    print("=" * 60)
    
    # Check if data file exists
    if not os.path.exists(DATA_PATH):
        print(f"❌ Data file not found: {DATA_PATH}")
        print("   Please ensure Phase 2 has completed successfully.")
        return False
    
    # Load data
    print(f"\n📂 Loading data from: {DATA_PATH}")
    try:
        df = pd.read_csv(DATA_PATH, low_memory=False)
        print(f"   ✓ Loaded {len(df)} rows, {len(df.columns)} columns")
    except Exception as e:
        print(f"   ❌ Failed to load data: {e}")
        return False
    
    # Get sample product
    if "product_name" not in df.columns:
        print("   ❌ 'product_name' column not found")
        return False
    
    sample_product = df['product_name'].unique()[0]
    print(f"\n📦 Sample product: {sample_product}")
    
    # Prepare time series
    product_df = df[df['product_name'] == sample_product].copy()
    if "week_start" not in product_df.columns:
        print("   ❌ 'week_start' column not found")
        return False
    
    ts = product_df[['week_start', 'sales_qty']].rename(columns={'week_start': 'date'})
    ts['date'] = pd.to_datetime(ts['date'], errors='coerce')
    ts = ts.dropna(subset=['date', 'sales_qty']).sort_values('date')
    
    print(f"   ✓ Time series: {len(ts)} data points")
    print(f"   ✓ Date range: {ts['date'].min()} to {ts['date'].max()}")
    print(f"   ✓ Sales range: {ts['sales_qty'].min():.0f} to {ts['sales_qty'].max():.0f}")
    
    if len(ts) < 8:
        print(f"   ❌ Insufficient data: need ≥8 weeks, got {len(ts)}")
        return False
    
    # Run forecast
    print(f"\n🔮 Running forecast (horizon: 90 days, fast_mode: True)...")
    try:
        res = train_ensemble_for_app(ts, horizon_days=90, fast_mode=True, debug=True)
        print("   ✓ Forecast completed successfully")
    except Exception as e:
        print(f"   ❌ Forecast failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Check results
    print(f"\n📊 Checking results...")
    
    # Check metrics
    metrics = res.get('metrics', {})
    if metrics:
        print(f"   Metrics:")
        for key, val in metrics.items():
            if isinstance(val, (int, float)):
                print(f"     - {key}: {val:.4f}")
            else:
                print(f"     - {key}: {val}")
    else:
        print("   ⚠️  No metrics returned")
    
    # Check forecast dataframe
    forecast = res.get('forecast')
    if forecast is not None:
        if isinstance(forecast, pd.DataFrame):
            print(f"   ✓ Forecast DataFrame: {len(forecast)} rows")
            if 'yhat' in forecast.columns:
                print(f"     - Forecast mean range: {forecast['yhat'].min():.2f} to {forecast['yhat'].max():.2f}")
        else:
            print(f"   ⚠️  Forecast is not a DataFrame: {type(forecast)}")
    else:
        print("   ❌ No forecast returned")
        return False
    
    # Check output file
    print(f"\n💾 Checking output file...")
    if os.path.exists(OUT_PATH):
        try:
            out_df = pd.read_csv(OUT_PATH)
            print(f"   ✓ Forecast file exists: {OUT_PATH}")
            print(f"   ✓ Output rows: {len(out_df)}")
            
            if len(out_df) >= 30:
                print(f"   ✓ Forecast file has sufficient rows (≥30 for 90-day horizon)")
            else:
                print(f"   ⚠️  Forecast file has fewer rows than expected: {len(out_df)} < 30")
            
            # Check columns
            if 'date' in out_df.columns and 'yhat' in out_df.columns:
                print(f"   ✓ Required columns present (date, yhat)")
            else:
                print(f"   ⚠️  Missing required columns")
                
        except Exception as e:
            print(f"   ❌ Failed to read output file: {e}")
            return False
    else:
        print(f"   ❌ Forecast output file not found: {OUT_PATH}")
        return False
    
    # Summary
    print(f"\n{'=' * 60}")
    print("✅ Test PASSED")
    print(f"{'=' * 60}")
    print(f"\nSummary:")
    print(f"  - Sample product: {sample_product}")
    print(f"  - Forecast path: {OUT_PATH}")
    print(f"  - Ensemble RMSE: {metrics.get('ensemble_rmse', 'N/A')}")
    print(f"  - Ensemble MAE: {metrics.get('ensemble_mae', 'N/A')}")
    print(f"  - Forecast rows: {len(out_df)}")
    
    return True

if __name__ == "__main__":
    success = test_flow()
    sys.exit(0 if success else 1)

