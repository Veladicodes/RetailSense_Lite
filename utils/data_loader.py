
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class RetailDataLoader:
    """
    Retail Data Loader for Market Dataset.
    
    Features:
    - Load raw CSV data
    - Handle missing values & date parsing
    - Validate dataset quality
    - Create time-based, expiry, and pricing features
    - Extract product-level or store-level time series
    - Save processed datasets for later modeling
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None

    def load_data(self):
        """Load and perform initial data cleaning"""
        print("📥 Loading retail data...")
        self.df = pd.read_csv(self.file_path)

        # Convert date columns
        self.df['week_start'] = pd.to_datetime(self.df['week_start'], errors='coerce')
        self.df['week_end'] = pd.to_datetime(self.df['week_end'], errors='coerce')

        if 'expiry_date' in self.df.columns:
            self.df['expiry_date'] = pd.to_datetime(self.df['expiry_date'], errors='coerce')

        print(f"✅ Data loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        return self.df

    def validate_data(self):
        """Validate dataset for missing values, negatives, and anomalies"""
        print("🔎 Validating dataset...")

        missing = self.df.isnull().sum()
        if missing.sum() > 0:
            print("⚠️ Missing values detected:\n", missing[missing > 0])
        else:
            print("✅ No missing values")

        if (self.df['sales_qty'] < 0).any():
            print("⚠️ Warning: Negative sales values detected!")

        if (self.df['price'] < 0).any():
            print("⚠️ Warning: Negative prices detected!")

        if 'stock_on_hand' in self.df.columns and (self.df['stock_on_hand'] < 0).any():
            print("⚠️ Warning: Negative stock values detected!")

        print("✅ Data validation complete")
        return True

    def create_time_features(self):
        """Create time-based and expiry-based features for modeling"""
        print("⏳ Creating time features...")

        # Basic time features
        self.df['year'] = self.df['week_start'].dt.year
        self.df['month'] = self.df['week_start'].dt.month
        self.df['week_of_year'] = self.df['week_start'].dt.isocalendar().week.astype(int)
        self.df['quarter'] = self.df['week_start'].dt.quarter
        self.df['is_month_end'] = self.df['week_end'].dt.is_month_end.astype(int)

        # Days until expiry (if applicable)
        if 'expiry_date' in self.df.columns:
            self.df['days_to_expiry'] = (self.df['expiry_date'] - self.df['week_start']).dt.days
            self.df['days_to_expiry'] = self.df['days_to_expiry'].fillna(365)
            self.df['near_expiry'] = (self.df['days_to_expiry'] <= 7).astype(int)

        # Price and availability features
        self.df['price_per_unit'] = self.df['price'] / self.df['sales_qty'].replace(0, 1)
        self.df['stock_sales_ratio'] = self.df['stock_on_hand'] / self.df['sales_qty'].replace(0, 1)

        print("✅ Time features created")
        return self.df

    def get_product_series(self, product_id, store_id=None):
        """
        Get time series for specific product.
        If store_id is provided -> return product-store series.
        If not -> aggregate across all stores.
        """
        print(f"📊 Extracting series for product {product_id} {'(store ' + str(store_id) + ')' if store_id else '(all stores)'}")

        if store_id:
            series_data = self.df[
                (self.df['product_id'] == product_id) & 
                (self.df['store_id'] == store_id)
            ].copy()
        else:
            # Aggregate across all stores
            series_data = self.df[self.df['product_id'] == product_id].copy()
            series_data = series_data.groupby('week_start').agg({
                'sales_qty': 'sum',
                'stock_on_hand': 'sum',
                'price': 'mean',
                'promotion': 'max',
                'holiday_flag': 'max'
            }).reset_index()

        series_data = series_data.sort_values('week_start')
        print(f"✅ Series extracted: {series_data.shape[0]} records")
        return series_data

    def save_processed_data(self, out_path="data/processed/market_processed.csv"):
        """Save the cleaned & feature-enriched dataset"""
        self.df.to_csv(out_path, index=False)
        print(f"💾 Processed data saved at {out_path}")


# Example usage (for testing directly)
if __name__ == "__main__":
    loader = RetailDataLoader("data/market.csv")
    df = loader.load_data()
    loader.validate_data()
    df = loader.create_time_features()

    # Print basic info
    print("\n📈 Dataset Info:")
    print(f"Date range: {df['week_start'].min()} to {df['week_start'].max()}")
    print(f"Unique products: {df['product_id'].nunique()}")
    print(f"Unique stores: {df['store_id'].nunique()}")
    print(f"Categories: {df['category'].unique()}")

    loader.save_processed_data()
