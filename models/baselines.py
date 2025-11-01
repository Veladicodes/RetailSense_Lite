# models/baselines.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
import traceback
warnings.filterwarnings('ignore')


class BaselineModels:
    def __init__(self):
        self.arima_model = None
        self.prophet_model = None
        self.results = {}
        self.df = None

    def load_data(self, file_path="data/processed/data_with_all_features.csv"):
        """Load the dataset (engineered features if available)"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"❌ Dataset not found at {file_path}")

        print(f"📂 Loading dataset from: {file_path}")   # ✅ Explicit print of input file
        self.df = pd.read_csv(file_path)
        self.df['week_start'] = pd.to_datetime(self.df['week_start'])

        # Ensure required cols exist (for Prophet regressors)
        for col in ['holiday_flag', 'promotion']:
            if col not in self.df.columns:
                self.df[col] = 0

        print(f"✅ Data loaded: {self.df.shape[0]} records, {self.df.shape[1]} columns")
        return self.df

    def train_arima(self, product_name=None, horizon=4):
        """Train ARIMA model and forecast"""
        print("🔄 Training ARIMA Model...")

        if product_name is None:
            product_name = self.df['product_name'].value_counts().index[0]

        if product_name not in self.df['product_name'].values:
            print(f"❌ Product {product_name} not found in dataset")
            return None

        product_data = self.df[self.df['product_name'] == product_name].copy()
        product_data = product_data.sort_values('week_start')
        ts_data = product_data.set_index('week_start')['sales_qty']

        try:
            self.arima_model = ARIMA(ts_data, order=(1, 1, 1))
            arima_fitted = self.arima_model.fit()
        except Exception:
            print("⚠️ ARIMA(1,1,1) failed, trying fallback ARIMA(1,0,0)...")
            try:
                self.arima_model = ARIMA(ts_data, order=(1, 0, 0))
                arima_fitted = self.arima_model.fit()
            except Exception as e:
                print("❌ ARIMA training failed completely:")
                traceback.print_exc()
                self.results['arima'] = {"error": str(e)}
                return None

        forecast = arima_fitted.forecast(steps=horizon)

        # Evaluate
        mae, rmse = None, None
        if len(ts_data) > horizon:
            true_vals = ts_data[-horizon:]
            pred_vals = arima_fitted.predict(start=len(ts_data) - horizon, end=len(ts_data) - 1)
            mae = mean_absolute_error(true_vals, pred_vals)
            mse = mean_squared_error(true_vals, pred_vals)
            rmse = np.sqrt(mse)

        self.results['arima'] = {
            'model': arima_fitted,
            'forecast': forecast,
            'product': product_name,
            'aic': arima_fitted.aic,
            'mae': mae,
            'rmse': rmse
        }

        print(f"✅ ARIMA trained for {product_name}")
        print(f"   AIC: {arima_fitted.aic:.2f}")
        print(f"   Next {horizon} weeks forecast: {forecast.values.round(2)}")

        return forecast

    def train_prophet(self, product_name=None, horizon=4):
        """Train Prophet model and forecast"""
        print("🔄 Training Prophet Model...")

        if product_name is None:
            product_name = self.df['product_name'].value_counts().index[0]

        if product_name not in self.df['product_name'].values:
            print(f"❌ Product {product_name} not found in dataset")
            return None

        product_data = self.df[self.df['product_name'] == product_name].copy()
        product_data = product_data.sort_values('week_start')

        try:
            prophet_data = product_data[['week_start', 'sales_qty']].copy()
            prophet_data.columns = ['ds', 'y']
            prophet_data['holiday'] = product_data['holiday_flag'].values
            prophet_data['promotion'] = product_data['promotion'].values

            self.prophet_model = Prophet(
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=True
            )
            self.prophet_model.add_regressor('holiday')
            self.prophet_model.add_regressor('promotion')
            self.prophet_model.fit(prophet_data)

            future = self.prophet_model.make_future_dataframe(periods=horizon, freq='W')
            future['holiday'] = 0
            future['promotion'] = 0
            forecast = self.prophet_model.predict(future)

            self.results['prophet'] = {
                'model': self.prophet_model,
                'forecast': forecast.tail(horizon)['yhat'].values,
                'product': product_name,
                'forecast_df': forecast
            }

            print(f"✅ Prophet trained for {product_name}")
            print(f"   Next {horizon} weeks forecast: {forecast.tail(horizon)['yhat'].values.round(2)}")

            return forecast.tail(horizon)['yhat'].values

        except Exception as e:
            print("❌ Prophet training failed:")
            traceback.print_exc()
            self.results['prophet'] = {"error": str(e)}
            return None

    def plot_forecasts(self):
        """Plot ARIMA & Prophet forecasts"""
        if not self.results:
            print("❌ No models trained yet!")
            return

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # ARIMA plot
        if self.results.get('arima') is not None and "forecast" in self.results['arima']:
            product_name = self.results['arima']['product']
            product_data = self.df[self.df['product_name'] == product_name].sort_values('week_start')
            product_data['smoothed'] = product_data['sales_qty'].rolling(window=4, min_periods=1).mean()

            axes[0].plot(product_data['week_start'], product_data['smoothed'],
                         label='Historical (Smoothed)', color='blue', alpha=0.7)

            last_date = product_data['week_start'].max()
            forecast_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1),
                                           periods=len(self.results['arima']['forecast']), freq='W')
            axes[0].plot(forecast_dates, self.results['arima']['forecast'],
                         'ro-', label='ARIMA Forecast', markersize=6, linewidth=2)

            axes[0].set_title(f'ARIMA Forecast - {product_name}')
            axes[0].set_xlabel('Date')
            axes[0].set_ylabel('Sales Quantity')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        else:
            axes[0].text(0.5, 0.5, "ARIMA Failed", ha="center", va="center", fontsize=14, color="red")
            axes[0].set_title("ARIMA Forecast")

        # Prophet plot
        if self.results.get('prophet') is not None and "forecast_df" in self.results['prophet']:
            forecast_df = self.results['prophet']['forecast_df']
            axes[1].plot(forecast_df['ds'], forecast_df['yhat'],
                         label='Prophet Forecast', color='orange')
            axes[1].fill_between(forecast_df['ds'],
                                 forecast_df['yhat_lower'], forecast_df['yhat_upper'],
                                 alpha=0.3, color='orange', label='Confidence Interval')

            future_mask = forecast_df['ds'] > self.df['week_start'].max()
            axes[1].plot(forecast_df[future_mask]['ds'], forecast_df[future_mask]['yhat'],
                         'ro-', markersize=6, label='Future Predictions')

            axes[1].set_title(f'Prophet Forecast - {self.results["prophet"]["product"]}')
            axes[1].set_xlabel('Date')
            axes[1].set_ylabel('Sales Quantity')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, "Prophet Failed", ha="center", va="center", fontsize=14, color="red")
            axes[1].set_title("Prophet Forecast")

        plt.tight_layout()
        plt.savefig('baseline_forecasts.png', dpi=300, bbox_inches='tight')
        plt.show()

    def save_results(self):
        """Save results to CSV"""
        results_summary = {
            'timestamp': pd.Timestamp.now(),
            'models_trained': [k for k, v in self.results.items() if v is not None],
        }

        if self.results.get('arima') is not None:
            if "forecast" in self.results['arima']:
                results_summary['arima_aic'] = self.results['arima']['aic']
                results_summary['arima_forecast'] = self.results['arima']['forecast'].tolist()
                results_summary['arima_mae'] = self.results['arima']['mae']
                results_summary['arima_rmse'] = self.results['arima']['rmse']
            else:
                results_summary['arima_error'] = self.results['arima']['error']

        if self.results.get('prophet') is not None:
            if "forecast" in self.results['prophet']:
                results_summary['prophet_forecast'] = self.results['prophet']['forecast'].tolist()
            else:
                results_summary['prophet_error'] = self.results['prophet']['error']

        pd.DataFrame([results_summary]).to_csv('baseline_results.csv', index=False)
        print("✅ Baseline results saved to 'baseline_results.csv'")


# Example usage
if __name__ == "__main__":
    baseline = BaselineModels()
    baseline.load_data("data/processed/data_with_all_features.csv")   # ✅ fixed input
    baseline.train_arima(horizon=4)
    baseline.train_prophet(horizon=4)
    baseline.plot_forecasts()
    baseline.save_results()
    print("🎉 Baseline models completed!")
