# models/feature_engineering.py
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ✅ Explicit export so `from models.feature_engineering import FeatureEngineering` works
__all__ = ["FeatureEngineering"]

class FeatureEngineering:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        self.df = None
        print("✅ FeatureEngineering initialized")

    def load_data(self, file_path):
        """Load the cleaned data"""
        print("📊 Loading data for feature engineering...")
        self.df = pd.read_csv(file_path)
        self.df['week_start'] = pd.to_datetime(self.df['week_start'])
        self.df['week_end'] = pd.to_datetime(self.df['week_end'])
        print(f"✅ Data loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        return self.df

    def create_time_features(self):
        print("🔄 Creating time-based features...")
        self.df['year'] = self.df['week_start'].dt.year.astype('Int64').fillna(0).astype(int)
        self.df['month'] = self.df['week_start'].dt.month.astype('Int64').fillna(0).astype(int)
        self.df['quarter'] = self.df['week_start'].dt.quarter.astype('Int64').fillna(0).astype(int)
        self.df['week_of_year'] = self.df['week_start'].dt.isocalendar().week.astype('Int64').fillna(0).astype(int)
        self.df['day_of_year'] = self.df['week_start'].dt.dayofyear.astype('Int64').fillna(0).astype(int)

        # cyclical encoding
        self.df['month_sin'] = np.sin(2 * np.pi * self.df['month'] / 12)
        self.df['month_cos'] = np.cos(2 * np.pi * self.df['month'] / 12)
        self.df['quarter_sin'] = np.sin(2 * np.pi * self.df['quarter'] / 4)
        self.df['quarter_cos'] = np.cos(2 * np.pi * self.df['quarter'] / 4)

        self.df['is_year_end'] = (self.df['month'].isin([11, 12])).astype(int)
        self.df['is_year_start'] = (self.df['month'].isin([1, 2])).astype(int)
        self.df['is_mid_year'] = (self.df['month'].isin([6, 7])).astype(int)

        delta_days = (self.df['week_start'] - self.df['week_start'].min()).dt.days
        self.df['weeks_from_start'] = delta_days.fillna(0).astype(int) // 7

        time_features = [
            'year','month','quarter','week_of_year','day_of_year',
            'month_sin','month_cos','quarter_sin','quarter_cos',
            'is_year_end','is_year_start','is_mid_year','weeks_from_start'
        ]
        print(f"✅ Created {len(time_features)} time-based features")
        return time_features

    def create_lag_features(self, target_col='sales_qty', lags=[1, 2, 4, 8]):
        print(f"🔄 Creating lag features for {target_col}...")
        self.df = self.df.sort_values(['product_name','week_start'])
        lag_features = []
        for product in self.df['product_name'].unique():
            mask = self.df['product_name'] == product
            product_data = self.df.loc[mask, target_col]
            for lag in lags:
                col = f'{target_col}_lag_{lag}'
                self.df.loc[mask, col] = product_data.shift(lag)
                lag_features.append(col)
        for col in lag_features:
            self.df[col].fillna(self.df[col].median(), inplace=True)
        print(f"✅ Created {len(lag_features)} lag features")
        return lag_features

    def create_rolling_features(self, target_col='sales_qty', windows=[3, 6, 12]):
        print(f"🔄 Creating rolling features for {target_col}...")
        self.df = self.df.sort_values(['product_name','week_start'])
        rolling_features = []
        for product in self.df['product_name'].unique():
            mask = self.df['product_name'] == product
            series = self.df.loc[mask, target_col]
            for w in windows:
                mean_col = f'{target_col}_ma_{w}'
                std_col = f'{target_col}_std_{w}'
                min_col = f'{target_col}_min_{w}'
                max_col = f'{target_col}_max_{w}'
                self.df.loc[mask, mean_col] = series.rolling(w).mean()
                self.df.loc[mask, std_col] = series.rolling(w).std()
                self.df.loc[mask, min_col] = series.rolling(w).min()
                self.df.loc[mask, max_col] = series.rolling(w).max()
                rolling_features.extend([mean_col, std_col, min_col, max_col])
        for col in rolling_features:
            self.df[col].fillna(self.df[col].median(), inplace=True)
        print(f"✅ Created {len(rolling_features)} rolling features")
        return rolling_features

    def create_price_features(self):
        print("🔄 Creating price-related features...")
        self.df = self.df.sort_values(['product_name','week_start'])
        for product in self.df['product_name'].unique():
            mask = self.df['product_name'] == product
            self.df.loc[mask,'price_change'] = self.df.loc[mask,'price'].pct_change()
            self.df.loc[mask,'price_lag_1'] = self.df.loc[mask,'price'].shift(1)
            self.df.loc[mask,'price_volatility_3'] = self.df.loc[mask,'price'].rolling(3).std()
            self.df.loc[mask,'price_volatility_6'] = self.df.loc[mask,'price'].rolling(6).std()
        self.df['price_relative_to_avg'] = self.df['price'] / self.df.groupby('product_name')['price'].transform('mean')
        self.df['price_relative_to_category'] = self.df['price'] / self.df.groupby('category')['price'].transform('mean')
        price_features = [
            'price_change','price_lag_1','price_volatility_3','price_volatility_6',
            'price_relative_to_avg','price_relative_to_category'
        ]
        for col in price_features:
            self.df[col].fillna(self.df[col].median(), inplace=True)
        print(f"✅ Created {len(price_features)} price-related features")
        return price_features

    def create_inventory_features(self):
        print("🔄 Creating inventory features...")
        self.df['stock_sales_ratio'] = self.df['stock_on_hand'] / (self.df['sales_qty'] + 1)
        self.df['sales_stock_ratio'] = self.df['sales_qty'] / (self.df['stock_on_hand'] + 1)
        self.df = self.df.sort_values(['product_name','week_start'])
        for product in self.df['product_name'].unique():
            mask = self.df['product_name'] == product
            self.df.loc[mask,'avg_stock_4w'] = self.df.loc[mask,'stock_on_hand'].rolling(4).mean()
            self.df.loc[mask,'stock_change'] = self.df.loc[mask,'stock_on_hand'].diff()
        stock_median = self.df['stock_on_hand'].median()
        self.df['low_stock'] = (self.df['stock_on_hand'] < stock_median*0.5).astype(int)
        self.df['high_stock'] = (self.df['stock_on_hand'] > stock_median*2).astype(int)
        inventory_features = [
            'stock_sales_ratio','sales_stock_ratio','avg_stock_4w',
            'stock_change','low_stock','high_stock'
        ]
        for col in inventory_features:
            self.df[col].fillna(self.df[col].median(), inplace=True)
        print(f"✅ Created {len(inventory_features)} inventory features")
        return inventory_features

    def create_categorical_features(self):
        print("🔄 Encoding categorical features...")
        categorical_cols = ['category','product_name','season','weather']
        encoded_features = []
        for col in categorical_cols:
            if col in self.df.columns:
                le = LabelEncoder()
                enc_col = f'{col}_encoded'
                self.df[enc_col] = le.fit_transform(self.df[col].astype(str))
                self.label_encoders[col] = le
                encoded_features.append(enc_col)
        for col in ['category','product_name']:
            if col in self.df.columns:
                freq_col = f'{col}_frequency'
                freq_map = self.df[col].value_counts().to_dict()
                self.df[freq_col] = self.df[col].map(freq_map)
                encoded_features.append(freq_col)
        print(f"✅ Created {len(encoded_features)} categorical features")
        return encoded_features

    def create_interaction_features(self):
        print("🔄 Creating interaction features...")
        self.df['price_promotion_interaction'] = self.df['price'] * self.df['promotion']
        self.df['holiday_season_interaction'] = self.df['holiday_flag'] * self.df.get('season_encoded',0)
        self.df['stock_availability_interaction'] = self.df['stock_on_hand'] * self.df['availability']
        self.df['price_stock_interaction'] = self.df['price'] * self.df['stock_on_hand']
        interaction_features = [
            'price_promotion_interaction','holiday_season_interaction',
            'stock_availability_interaction','price_stock_interaction'
        ]
        print(f"✅ Created {len(interaction_features)} interaction features")
        return interaction_features

    def run_complete_feature_engineering(self):
        print("\n🚀 Starting Complete Feature Engineering Pipeline")
        print("="*60)
        all_features = []
        all_features += self.create_time_features()
        all_features += self.create_lag_features()
        all_features += self.create_rolling_features()
        all_features += self.create_price_features()
        all_features += self.create_inventory_features()
        all_features += self.create_categorical_features()
        all_features += self.create_interaction_features()
        self.feature_names = all_features
        print(f"\n✅ Feature Engineering Complete!")
        print(f"📊 Total Features Created: {len(all_features)}")
        print(f"📈 Dataset Shape: {self.df.shape}")
        return self.df, all_features

    def save_engineered_data(self, output_path='data/processed/engineered_features.csv'):
        """Save the dataset with all engineered features"""
        # Ensure output_path is absolute and normalize path separators
        if not os.path.isabs(output_path):
            # Make it absolute relative to current working directory
            output_path = os.path.abspath(output_path)
        
        # Ensure directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Validate that dataframe exists and is not empty
        if self.df is None or self.df.empty:
            raise ValueError("Cannot save: No data available. Run feature engineering first.")
        
        print(f"💾 Saving engineered dataset to {output_path}...")
        try:
            self.df.to_csv(output_path, index=False)
            print(f"✅ Successfully saved {len(self.df)} rows to {output_path}")
        except Exception as e:
            print(f"❌ Error saving to {output_path}: {e}")
            raise

        # Save feature metadata to same directory as the CSV
        feature_metadata = {
            'total_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'dataset_shape': list(self.df.shape),
            'timestamp': pd.Timestamp.now().isoformat()
        }
        import json
        metadata_path = os.path.join(output_dir, 'feature_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(feature_metadata, f, indent=2)
        print(f"✅ Engineered dataset and metadata saved successfully!")
        print(f"📁 CSV: {output_path}")
        print(f"📁 Metadata: {metadata_path}")
        return output_path

# ✅ Example usage
if __name__ == "__main__":
    fe = FeatureEngineering()
    data_path = "data/processed/final-retail_data_cleaned1.csv"
    fe.load_data(data_path)
    engineered_df, features = fe.run_complete_feature_engineering()
    fe.save_engineered_data("data/processed/data_with_all_features.csv")
    print("\n" + "="*60)
    print("🎉 FEATURE ENGINEERING COMPLETED!")
    print("="*60)
    print(f"📁 Output CSV: data/processed/data_with_all_features.csv")
    print(f"📁 Metadata: feature_metadata.json")
    print(f"📊 Features created: {len(features)}")
