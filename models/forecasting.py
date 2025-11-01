# models/forecasting.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except Exception:
    lgb = None
    LGB_AVAILABLE = False
import joblib
import warnings
import os
warnings.filterwarnings('ignore')


class AdvancedForecasting:
    def __init__(self):
        self.xgb_model = None
        self.lgb_model = None
        self.feature_cols = []
        self.results = {}
        self.df = None
        # ✅ Always point to engineered dataset
        self.default_path = os.path.join("data", "processed", "data_with_all_features.csv")

    def load_and_prepare_data(self, file_path=None):
        """Load engineered dataset"""
        if file_path is None:
            file_path = self.default_path

        print(f"📊 Loading engineered dataset for advanced forecasting from: {file_path}")
        self.df = pd.read_csv(file_path)
        self.df['week_start'] = pd.to_datetime(self.df['week_start'])
        self.df['week_end'] = pd.to_datetime(self.df['week_end'])
        print(f"✅ Data loaded: {self.df.shape[0]} records, {self.df.shape[1]} columns")
        return self.df

    def prepare_features(self):
        """Prepare feature matrix (numeric only)"""
        base_exclude = [
            'sales_qty', 'week_start', 'week_end',
            'product_name', 'category', 'season', 'weather'
        ]

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.feature_cols = [c for c in numeric_cols if c not in base_exclude]

        X = self.df[self.feature_cols].copy()
        y = self.df['sales_qty'].copy()
        X.fillna(X.median(), inplace=True)

        print(f"✅ Features prepared: {len(self.feature_cols)} numeric features")
        return X, y

    def train_models(self):
        """Train XGBoost and LightGBM models"""
        print("🚀 Training advanced forecasting models...")
        X, y = self.prepare_features()

        n = len(X)
        if n < 5:
            raise ValueError(f"Insufficient samples for advanced models: {n} < 5. Provide more data for the selected product.")
        split_idx = max(1, int(n * 0.8))
        if split_idx >= n:
            split_idx = n - 1
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        print(f"📊 Training set: {len(X_train)}, Test set: {len(X_test)}")

        # ---- XGBoost (Production-grade with early stopping) ----
        print("🔄 Training XGBoost with early stopping...")
        self.xgb_model = xgb.XGBRegressor(
            n_estimators=1000,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_weight=3,
            gamma=0.1,
            random_state=42,
            n_jobs=-1,
            early_stopping_rounds=50
        )
        self.xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        xgb_pred = self.xgb_model.predict(X_test)
        xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
        xgb_mae = mean_absolute_error(y_test, xgb_pred)
        xgb_mape = np.mean(np.abs((y_test - xgb_pred) / y_test)) * 100

        print(f"✅ XGBoost - RMSE: {xgb_rmse:.2f}, MAE: {xgb_mae:.2f}, MAPE: {xgb_mape:.1f}% (stopped at {self.xgb_model.best_iteration} iterations)")

        # ---- LightGBM (Production-grade with early stopping) ----
        lgb_pred = None
        if LGB_AVAILABLE:
            try:
                print("🔄 Training LightGBM with early stopping...")
                try:
                    print(f"ℹ️ LightGBM version: {getattr(lgb, '__version__', 'unknown')}")
                except Exception:
                    pass
                self.lgb_model = lgb.LGBMRegressor(
                    n_estimators=1000,
                    num_leaves=64,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    min_child_samples=20,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1
                )
                self.lgb_model.fit(
                    X_train, y_train,
                    eval_set=[(X_test, y_test)],
                    eval_metric='rmse',
                    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
                )

                lgb_pred = self.lgb_model.predict(X_test)
                lgb_rmse = np.sqrt(mean_squared_error(y_test, lgb_pred))
                lgb_mae = mean_absolute_error(y_test, lgb_pred)
                lgb_mape = np.mean(np.abs((y_test - lgb_pred) / y_test)) * 100

                print(f"✅ LightGBM - RMSE: {lgb_rmse:.2f}, MAE: {lgb_mae:.2f}, MAPE: {lgb_mape:.1f}% (stopped at {self.lgb_model.best_iteration_} iterations)")
            except Exception as e:
                print(f"⚠️ LightGBM training failed, skipping. Reason: {e}")
        else:
            print("⚠️ LightGBM not installed. Skipping LightGBM training.")

        self.results = {
            'X_test': X_test,
            'y_test': y_test,
            'xgb_pred': xgb_pred,
            'xgb_metrics': {'RMSE': xgb_rmse, 'MAE': xgb_mae, 'MAPE': xgb_mape},
        }
        if lgb_pred is not None:
            self.results['lgb_pred'] = lgb_pred
            self.results['lgb_metrics'] = {'RMSE': lgb_rmse, 'MAE': lgb_mae, 'MAPE': lgb_mape}

        return self.results

    def plot_feature_importance(self):
        if not self.xgb_model and not self.lgb_model:
            print("❌ Models not trained yet!")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # XGBoost
        xgb_importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.xgb_model.feature_importances_
        }).sort_values('importance', ascending=True)

        ax1.barh(xgb_importance.tail(15)['feature'], xgb_importance.tail(15)['importance'], color='skyblue')
        ax1.set_title('XGBoost - Top 15 Features')

        # LightGBM
        if self.lgb_model is not None:
            lgb_importance = pd.DataFrame({
                'feature': self.feature_cols,
                'importance': self.lgb_model.feature_importances_
            }).sort_values('importance', ascending=True)

            ax2.barh(lgb_importance.tail(15)['feature'], lgb_importance.tail(15)['importance'], color='lightcoral')
            ax2.set_title('LightGBM - Top 15 Features')
        else:
            ax2.axis('off')
            ax2.set_title('LightGBM not available')

        plt.tight_layout()
        plt.savefig('forecasting_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_predictions(self):
        if not self.results:
            print("❌ No results available!")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        y_test = self.results['y_test']
        xgb_pred = self.results['xgb_pred']
        lgb_pred = self.results.get('lgb_pred')

        axes[0, 0].plot(y_test.values[:100], label='Actual')
        axes[0, 0].plot(xgb_pred[:100], label='XGBoost')
        axes[0, 0].legend(); axes[0, 0].set_title("XGBoost Predictions (First 100)")

        axes[0, 1].plot(y_test.values[:100], label='Actual')
        if lgb_pred is not None:
            axes[0, 1].plot(lgb_pred[:100], label='LightGBM')
            axes[0, 1].legend(); axes[0, 1].set_title("LightGBM Predictions (First 100)")
        else:
            axes[0, 1].text(0.5, 0.5, 'LightGBM not available', ha='center', va='center')
            axes[0, 1].set_title("LightGBM Predictions (N/A)")

        axes[1, 0].scatter(y_test, xgb_pred, alpha=0.5)
        axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        axes[1, 0].set_title("XGBoost Actual vs Predicted")

        if lgb_pred is not None:
            axes[1, 1].scatter(y_test, lgb_pred, alpha=0.5, color='green')
            axes[1, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            axes[1, 1].set_title("LightGBM Actual vs Predicted")
        else:
            axes[1, 1].axis('off')
            axes[1, 1].set_title("LightGBM (N/A)")

        plt.tight_layout()
        plt.savefig('forecasting_predictions.png', dpi=300, bbox_inches='tight')
        plt.show()

    def save_models(self):
        if self.xgb_model:
            joblib.dump(self.xgb_model, 'xgboost_model.pkl')
            print("✅ XGBoost model saved")

        if self.lgb_model:
            joblib.dump(self.lgb_model, 'lightgbm_model.pkl')
            print("✅ LightGBM model saved")

        if self.results:
            data = {
                'actual': self.results['y_test'],
                'xgb_pred': self.results['xgb_pred'],
            }
            if 'lgb_pred' in self.results:
                data['lgb_pred'] = self.results['lgb_pred']
            results_df = pd.DataFrame(data)
            results_df.to_csv('forecasting_results.csv', index=False)
            print("✅ Forecasting results saved to forecasting_results.csv")


# Example usage
if __name__ == "__main__":
    forecaster = AdvancedForecasting()
    forecaster.load_and_prepare_data()   # ✅ Always picks data_with_all_features.csv
    forecaster.train_models()
    forecaster.plot_feature_importance()
    forecaster.plot_predictions()
    forecaster.save_models()
    print("🎉 Advanced forecasting completed!")
