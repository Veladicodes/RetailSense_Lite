import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import joblib
import warnings
warnings.filterwarnings("ignore")


class AnomalyDetection:
    def __init__(self):
        self.df = None
        self.scaler = StandardScaler()
        self.pca_vis = PCA(n_components=2)      # For 2D visualization
        self.pca_full = PCA(n_components=0.95)  # For dimensionality reduction
        self.iforest = None
        self.ocsvm = None

    def load_data(self, file_path):
        """Load dataset for anomaly detection"""
        print("📊 Loading data for anomaly detection...")
        self.df = pd.read_csv(file_path)

        if 'week_start' in self.df.columns:
            self.df['week_start'] = pd.to_datetime(self.df['week_start'], errors="coerce")

        print(f"✅ Data loaded: {self.df.shape[0]} records, {self.df.shape[1]} columns")
        return self.df

    def prepare_features(self):
        """Prepare numerical features"""
        print("🔄 Preparing features for anomaly detection...")

        X = self.df.select_dtypes(include=[np.number]).copy()
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.dropna(axis=1, how='all')
        X = X.fillna(X.median())

        feature_names = list(X.columns)
        X_scaled = self.scaler.fit_transform(X)
        X_pca = self.pca_vis.fit_transform(X_scaled)

        print(f"✅ Features prepared: {X.shape[1]} numeric features → {X_pca.shape[1]} PCA components")
        return X, X_scaled, X_pca, feature_names

    def run_models(self, X_scaled, X_pca):
        """Train Isolation Forest and One-Class SVM"""
        print("🔄 Training Isolation Forest...")
        self.iforest = IsolationForest(contamination=0.05, random_state=42)
        iforest_labels = self.iforest.fit_predict(X_pca)
        self.df['is_anomaly_iforest'] = (iforest_labels == -1).astype(int)
        self.df['iforest_scores'] = self.iforest.decision_function(X_pca)

        print("🔄 Training One-Class SVM...")
        self.ocsvm = OneClassSVM(nu=0.05, kernel="rbf", gamma="scale")
        ocsvm_labels = self.ocsvm.fit_predict(X_pca)
        self.df['is_anomaly_ocsvm'] = (ocsvm_labels == -1).astype(int)

        print("✅ Models trained and anomaly flags added")

    def save_models(self, output_dir=r"F:\RetailSense_Lite\outputs"):
        """Save trained anomaly models"""
        os.makedirs(output_dir, exist_ok=True)
        if self.iforest:
            joblib.dump(self.iforest, os.path.join(output_dir, "isolation_forest_model.pkl"))
        if self.ocsvm:
            joblib.dump(self.ocsvm, os.path.join(output_dir, "ocsvm_model.pkl"))
        print(f"✅ Models saved to {output_dir}")

    def visualize_dashboard(self, output_dir=r"F:\RetailSense_Lite\outputs"):
        """Dashboard-style anomaly visualization"""
        print("📈 Generating anomaly dashboard...")
        os.makedirs(output_dir, exist_ok=True)

        fig, axs = plt.subplots(3, 2, figsize=(15, 12))
        axs = axs.flatten()

        if 'sales_qty' in self.df.columns:
            axs[0].scatter(self.df['week_start'], self.df['sales_qty'],
                           c=self.df['is_anomaly_iforest'], cmap='coolwarm', s=10)
            axs[0].set_title("Sales Anomalies Over Time - Isolation Forest")

            axs[1].scatter(self.df['week_start'], self.df['sales_qty'],
                           c=self.df['is_anomaly_ocsvm'], cmap='coolwarm', s=10)
            axs[1].set_title("Sales Anomalies Over Time - One-Class SVM")

        axs[2].hist(self.df['iforest_scores'], bins=40, edgecolor='black')
        axs[2].set_title("Isolation Forest - Anomaly Score Distribution")

        if 'product_name' in self.df.columns:
            top_products = self.df[self.df['is_anomaly_iforest'] == 1]['product_name'].value_counts().head(5)
            top_products.plot(kind='bar', ax=axs[3])
            axs[3].set_title("Top Products with Most Anomalies")

        if 'season' in self.df.columns:
            season_counts = self.df[self.df['is_anomaly_iforest'] == 1]['season'].value_counts()
            axs[4].pie(season_counts, labels=season_counts.index, autopct='%1.1f%%')
            axs[4].set_title("Seasonal Distribution of Anomalies")

        if 'price' in self.df.columns and 'sales_qty' in self.df.columns:
            axs[5].scatter(self.df['price'], self.df['sales_qty'],
                           c=self.df['is_anomaly_iforest'], cmap='coolwarm', alpha=0.6)
            axs[5].set_title("Price vs Sales with Anomalies")

        plt.tight_layout()

        # Save dashboard
        plot_path = os.path.join(output_dir, "anomaly_dashboard.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"✅ Anomaly dashboard saved to {plot_path}")


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(BASE_DIR, "data", "processed", "data_with_all_features.csv")

    detector = AnomalyDetection()
    detector.load_data(data_path)
    X, X_scaled, X_pca, features = detector.prepare_features()
    detector.run_models(X_scaled, X_pca)
    detector.save_models()
    detector.visualize_dashboard()
    print("🎉 Anomaly detection completed successfully")
