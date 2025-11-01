import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class BusinessInsightsEngine:
    def __init__(self):
        self.df = None
        self.models = {}
        self.alerts = []
    
    def load_data_and_models(self, data_path, model_paths=None):
        print("📊 Loading data and models...")
        try:
            self.df = pd.read_csv(data_path)
            self.df['week_start'] = pd.to_datetime(self.df['week_start'])
            print(f"✅ Data loaded: {self.df.shape}")
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
        
        if model_paths is None:
            model_paths = {
                'xgboost': '../models/xgboost_model.pkl',
                'lightgbm': '../models/lightgbm_model.pkl',
                'isolation_forest': '../models/isolation_forest_model.pkl'
            }
        
        for name, path in model_paths.items():
            if os.path.exists(path):
                try:
                    self.models[name] = joblib.load(path)
                    print(f"✅ Model loaded: {name}")
                except Exception as e:
                    print(f"⚠️ Failed to load {name}: {e}")
            else:
                print(f"⚠️ Model file missing: {path}")
        return True
    
    def generate_inventory_alerts(self):
        print("🚨 Generating inventory alerts...")
        alerts = []
        for product in self.df['product_name'].unique():
            product_data = self.df[self.df['product_name'] == product].sort_values('week_start')
            latest = product_data.iloc[-1]
            recent_sales = product_data['sales_qty'].tail(4).mean()
            current_stock = latest['stock_on_hand']
            if recent_sales > 0:
                days_until_stockout = current_stock / (recent_sales / 7)
            else:
                days_until_stockout = float('inf')
            
            if days_until_stockout <= 3:
                urgency = "🔴 CRITICAL"
                restock_qty = int(recent_sales * 2)
                action = f"Immediate restock: {restock_qty} units"
            elif days_until_stockout <= 7:
                urgency = "🟡 WARNING"
                restock_qty = int(recent_sales * 1.5)
                action = f"Plan restock: {restock_qty} units"
            elif current_stock > recent_sales * 8:
                urgency = "🔵 OVERSTOCK"
                restock_qty = 0
                action = "Consider promotional pricing"
            else:
                continue  # No alert
            
            alerts.append({
                'product': product,
                'urgency': urgency,
                'current_stock': current_stock,
                'days_until_stockout': days_until_stockout,
                'recommended_restock': restock_qty,
                'action': action
            })
        
        self.alerts = sorted(alerts, key=lambda x: ['🔴 CRITICAL', '🟡 WARNING', '🔵 OVERSTOCK'].index(x['urgency']))
        print(f"✅ {len(self.alerts)} inventory alerts generated")
        return self.alerts
    
    def detect_sales_anomalies(self):
        print("🔍 Detecting sales anomalies...")
        insights = []
        if 'iso_forest_anomaly' not in self.df.columns:
            print("⚠️ Missing anomaly column.")
            return insights
        
        anomalies = self.df[self.df['iso_forest_anomaly'] == -1]
        for _, row in anomalies.iterrows():
            insights.append({
                'product': row['product_name'],
                'category': row['category'],
                'week_start': row['week_start'].strftime('%Y-%m-%d'),
                'sales_qty': row['sales_qty'],
                'action': "Investigate anomaly"
            })
        print(f"✅ {len(insights)} sales anomalies identified")
        return insights
    
    def generate_executive_summary(self):
        return {
            'report_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_products': self.df['product_name'].nunique(),
            'total_alerts': len(self.alerts),
            'total_anomalies': len(self.df[self.df.get('iso_forest_anomaly', 0) == -1])
        }
    
    def create_actionable_insights_report(self):
        inventory = self.generate_inventory_alerts()
        anomalies = self.detect_sales_anomalies()
        executive_summary = self.generate_executive_summary()
        return {
            'executive_summary': executive_summary,
            'inventory_alerts': inventory[:10],
            'sales_anomalies': anomalies[:10]
        }
    
    def display_business_insights(self, report):
        print("\n--- EXECUTIVE SUMMARY ---")
        for k, v in report['executive_summary'].items():
            print(f"{k}: {v}")
        print("\n--- INVENTORY ALERTS SAMPLE ---")
        for alert in report['inventory_alerts'][:3]:
            print(alert)
        print("\n--- SALES ANOMALIES SAMPLE ---")
        for anomaly in report['sales_anomalies'][:3]:
            print(anomaly)
    
    def save_insights_to_files(self, report):
        os.makedirs('../outputs/', exist_ok=True)
        pd.DataFrame(report['inventory_alerts']).to_csv('../outputs/inventory_alerts.csv', index=False)
        pd.DataFrame(report['sales_anomalies']).to_csv('../outputs/sales_anomalies.csv', index=False)
        pd.DataFrame([report['executive_summary']]).to_csv('../outputs/executive_summary.csv', index=False)
        print("✅ Insights saved to ../outputs/")
    
    def create_insights_visualizations(self, report):
        # Simple inventory alert distribution chart
        urgencies = [a['urgency'] for a in report['inventory_alerts']]
        pd.Series(urgencies).value_counts().plot(kind='bar')
        plt.title("Inventory Alert Urgency Distribution")
        plt.ylabel('Count')
        plt.tight_layout()
        os.makedirs('../outputs/', exist_ok=True)
        plt.savefig('../outputs/inventory_alert_distribution.png', dpi=300)
        plt.show()
        print("✅ Visualization saved")

if __name__ == "__main__":
    engine = BusinessInsightsEngine()
    if engine.load_data_and_models('../data/processed/data_with_anomalies.csv'):
        report = engine.create_actionable_insights_report()
        engine.display_business_insights(report)
        engine.create_insights_visualizations(report)
        engine.save_insights_to_files(report)
