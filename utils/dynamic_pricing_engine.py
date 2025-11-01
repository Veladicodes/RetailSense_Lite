import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")


class DynamicPricingEngine:
    """
    Complete Dynamic Pricing Engine for Retail
    Enhanced version with advanced ML capabilities and integration with notebook structure
    """

    def __init__(self):
        self.elasticity_models = {}
        self.demand_models = {}
        self.pricing_rules = {
            "min_margin": 0.15,
            "max_discount": 0.40,
            "competitor_buffer": 0.05,
            "inventory_threshold_low": 0.20,
            "inventory_threshold_high": 0.80,
            "seasonal_multiplier": 1.2,
        }
        self.scaler = StandardScaler()
        self.products_df = pd.DataFrame()
        self.sales_df = pd.DataFrame()
        self.elasticity_df = pd.DataFrame()
        self.feature_columns = []

    # ------------------------------------------------------------------
    # Data Integration with Notebook
    # ------------------------------------------------------------------

    def load_data_from_dataframe(self, df, products_df=None):
        """
        Load data from the provided dataframe structure
        """
        self.sales_df = df.copy()
        
        # If products_df is not provided, create it from the sales data
        if products_df is None:
            self.products_df = self._create_products_df_from_sales()
        else:
            self.products_df = products_df.copy()
        
        # Convert date columns if they exist
        if 'week_start' in self.sales_df.columns:
            self.sales_df['date'] = pd.to_datetime(self.sales_df['week_start'])
        elif 'date' in self.sales_df.columns:
            self.sales_df['date'] = pd.to_datetime(self.sales_df['date'])
        
        print(f"Loaded {len(self.sales_df)} sales records for {self.sales_df['product_name'].nunique()} products")
        return self.products_df, self.sales_df

    def _create_products_df_from_sales(self):
        """Create products dataframe from sales data"""
        products_df = self.sales_df.groupby(['product_name', 'category']).agg({
            'price': ['mean', 'min', 'max'],
            'sales_qty': 'mean',
            'stock_on_hand': 'mean'
        }).reset_index()
        
        # Flatten column names
        products_df.columns = ['product_name', 'category', 'avg_price', 'min_price', 'max_price', 
                              'avg_demand', 'avg_stock']
        
        # Calculate cost price (assuming 30% margin on average)
        products_df['cost_price'] = products_df['avg_price'] * 0.7
        
        # Add product_id for compatibility
        products_df['product_id'] = products_df['product_name']
        
        return products_df

    def prepare_features(self, feature_columns=None):
        """
        Prepare features for modeling based on the notebook's feature engineering
        """
        if feature_columns is None:
            # Define default feature columns based on the dataset
            self.feature_columns = [
                'price', 'stock_on_hand', 'promotion', 'holiday_flag', 'disaster_flag',
                'category_encoded', 'price_relative_to_avg', 'price_relative_to_category',
                'stock_sales_ratio', 'sales_stock_ratio', 'month_sin', 'month_cos',
                'sales_lag_1', 'sales_lag_2', 'sales_ma_4', 'price_volatility_4w'
            ]
        else:
            self.feature_columns = feature_columns
        
        print(f"Using {len(self.feature_columns)} features for modeling")
        return self.feature_columns

    # ------------------------------------------------------------------
    # Enhanced Elasticity & Prediction
    # ------------------------------------------------------------------

    def calculate_price_elasticity(self, product_id=None, method='advanced'):
        """
        Calculate price elasticity for products using multiple methods
        """
        if self.sales_df.empty:
            raise ValueError("No sales data loaded. Please load data first.")
        
        if product_id:
            data = self.sales_df[self.sales_df['product_name'] == product_id].copy()
        else:
            data = self.sales_df.copy()

        elasticity_results = []

        for pid in data['product_name'].unique():
            product_data = data[data['product_name'] == pid].copy()
            if len(product_data) < 20:
                continue

            if method == 'simple':
                # Simple log-log regression
                product_data['log_price'] = np.log(product_data['price'] + 1)
                product_data['log_demand'] = np.log(product_data['sales_qty'] + 1)
                
                X = product_data[['log_price']].values
                y = product_data['log_demand'].values
                
                model = Ridge(alpha=0.1)
                model.fit(X, y)
                elasticity = model.coef_[0]
                r2 = model.score(X, y)
                
            elif method == 'advanced':
                # Advanced regression with multiple features
                X = product_data[self.feature_columns].fillna(0).values
                y = np.log(product_data['sales_qty'] + 1)
                
                # Use time series cross-validation
                tscv = TimeSeriesSplit(n_splits=3)
                model = RandomForestRegressor(n_estimators=50, random_state=42)
                
                try:
                    cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='r2')
                    r2 = cv_scores.mean()
                    
                    # Fit final model
                    model.fit(X, y)
                    
                    # Estimate elasticity by perturbing price
                    X_test = X.copy()
                    price_idx = self.feature_columns.index('price') if 'price' in self.feature_columns else None
                    
                    if price_idx is not None:
                        # Create price variations
                        price_range = np.linspace(
                            product_data['price'].min() * 0.8, 
                            product_data['price'].max() * 1.2, 
                            20
                        )
                        demand_predictions = []
                        
                        for price_val in price_range:
                            X_test[:, price_idx] = price_val
                            pred = model.predict(X_test)
                            demand_predictions.append(np.mean(np.exp(pred) - 1))
                        
                        # Calculate elasticity
                        log_prices = np.log(price_range + 1)
                        log_demand = np.log(np.array(demand_predictions) + 1)
                        elasticity = np.polyfit(log_prices, log_demand, 1)[0]
                    else:
                        elasticity = -1.0  # Default value
                        
                except Exception as e:
                    print(f"Error modeling {pid}: {e}")
                    continue
                    
            elasticity_results.append({
                'product_id': pid,
                'elasticity': elasticity,
                'r2_score': r2,
                'method': method
            })
            self.elasticity_models[pid] = model

        self.elasticity_df = pd.DataFrame(elasticity_results)
        return self.elasticity_df

    def predict_demand(self, product_id, features_dict):
        """
        Predict demand using the trained model or fallback to elasticity formula
        """
        if product_id in self.elasticity_models:
            model = self.elasticity_models[product_id]
            
            # Prepare feature vector in correct order
            feature_vector = []
            for col in self.feature_columns:
                if col in features_dict:
                    feature_vector.append(features_dict[col])
                else:
                    feature_vector.append(0)  # Default value for missing features
            
            # Make prediction
            log_demand_pred = model.predict([feature_vector])[0]
            demand_pred = np.exp(log_demand_pred) - 1
            return max(0, demand_pred)
        
        # Fallback using base elasticity
        prod_data = self.products_df[self.products_df['product_name'] == product_id]
        if not prod_data.empty:
            avg_price = prod_data['avg_price'].iloc[0]
            avg_demand = prod_data['avg_demand'].iloc[0]
            
            # Get elasticity or use default
            if not self.elasticity_df.empty and product_id in self.elasticity_df['product_id'].values:
                elasticity = self.elasticity_df[self.elasticity_df['product_id'] == product_id]['elasticity'].iloc[0]
            else:
                elasticity = -1.5  # Default elasticity
                
            current_price = features_dict.get('price', avg_price)
            demand = avg_demand * (current_price / avg_price) ** elasticity
            return max(0, demand)
        
        return 0.0

    # ------------------------------------------------------------------
    # Optimization & Recommendations
    # ------------------------------------------------------------------

    def calculate_revenue_at_price(self, product_id, price, features_dict):
        """Calculate revenue, profit, and margin at a given price"""
        # Update price in features
        features_dict['price'] = price
        
        demand = self.predict_demand(product_id, features_dict)
        if demand == 0:
            return None

        # Get cost price
        prod_data = self.products_df[self.products_df['product_name'] == product_id]
        if prod_data.empty:
            return None
            
        cost_price = prod_data['cost_price'].iloc[0]
        revenue = price * demand
        profit = revenue - (cost_price * demand)
        
        return {
            'price': price,
            'demand': demand,
            'revenue': revenue,
            'profit': profit,
            'margin': (price - cost_price) / price if price > 0 else 0,
        }

    def optimize_price(self, product_id, features_dict, strategy='profit_max'):
        """Find optimal price with multiple strategy options"""
        prod_data = self.products_df[self.products_df['product_name'] == product_id]
        if prod_data.empty:
            return None

        cost_price = prod_data['cost_price'].iloc[0]
        avg_price = prod_data['avg_price'].iloc[0]
        
        # Price bounds
        min_price = cost_price * (1 + self.pricing_rules['min_margin'])
        max_price = avg_price * (1 + self.pricing_rules['max_discount'])
        
        # Consider competitor price if available
        if 'competitor_price' in features_dict and features_dict['competitor_price'] > 0:
            buffer = features_dict['competitor_price'] * self.pricing_rules['competitor_buffer']
            max_price = min(max_price, features_dict['competitor_price'] + buffer)
        
        # Consider inventory level for strategy
        inventory_level = features_dict.get('inventory_level', 0.5)
        if inventory_level < self.pricing_rules['inventory_threshold_low']:
            strategy = 'margin_max'  # Premium pricing when stock is low
        elif inventory_level > self.pricing_rules['inventory_threshold_high']:
            strategy = 'demand_max'  # Clearance pricing when stock is high

        # Test price points
        price_points = np.linspace(min_price, max_price, 50)
        results = []
        
        for price in price_points:
            res = self.calculate_revenue_at_price(product_id, price, features_dict.copy())
            if res:
                results.append(res)
                
        if not results:
            return None

        # Select optimal based on strategy
        if strategy == 'profit_max':
            optimal_result = max(results, key=lambda x: x['profit'])
            strategy_name = "Profit Maximization"
        elif strategy == 'revenue_max':
            optimal_result = max(results, key=lambda x: x['revenue'])
            strategy_name = "Revenue Maximization"
        elif strategy == 'margin_max':
            optimal_result = max(results, key=lambda x: x['margin'])
            strategy_name = "Margin Maximization"
        elif strategy == 'demand_max':
            optimal_result = max(results, key=lambda x: x['demand'])
            strategy_name = "Demand Maximization"
        else:
            optimal_result = max(results, key=lambda x: x['profit'])
            strategy_name = "Profit Maximization"

        optimal_result.update({
            'strategy': strategy_name, 
            'product_id': product_id,
            'category': prod_data['category'].iloc[0]
        })
        
        return optimal_result

    def get_pricing_recommendations(self, current_date=None, strategy='profit_max'):
        """Generate pricing recommendations for all products"""
        if current_date is None:
            current_date = pd.Timestamp.now()
            
        # Get the latest data for each product
        latest_data = self.sales_df.sort_values('date').groupby('product_name').tail(1)
        
        recommendations = []
        for _, row in latest_data.iterrows():
            product_id = row['product_name']
            
            # Create features dictionary
            features_dict = {}
            for col in self.feature_columns:
                if col in row:
                    features_dict[col] = row[col]
                else:
                    # Set default values for missing features
                    if col == 'price':
                        features_dict[col] = row['price']
                    elif col == 'stock_on_hand':
                        features_dict[col] = row.get('stock_on_hand', 100)
                    else:
                        features_dict[col] = 0
            
            # Add time-based features
            features_dict['month_sin'] = np.sin(2 * np.pi * current_date.month / 12)
            features_dict['month_cos'] = np.cos(2 * np.pi * current_date.month / 12)
            
            # Get optimization result
            optimal = self.optimize_price(product_id, features_dict, strategy)
            
            if optimal:
                current_price = row['price']
                price_change = optimal['price'] - current_price
                price_change_pct = (price_change / current_price) * 100
                
                optimal.update({
                    'current_price': current_price,
                    'price_change': price_change,
                    'price_change_pct': price_change_pct,
                    'inventory_level': row.get('stock_on_hand', 100) / row.get('avg_stock_4w', 200) if 'avg_stock_4w' in row else 0.5,
                })
                recommendations.append(optimal)
                
        return pd.DataFrame(recommendations)

    def save_recommendations(self, recommendations, path='pricing_recommendations.csv'):
        """Save recommendations to CSV file"""
        if recommendations is not None and not recommendations.empty:
            recommendations.to_csv(path, index=False)
            print(f"Saved {len(recommendations)} recommendations to {path}")
            return True
        return False

    # ------------------------------------------------------------------
    # Dashboard and Analysis
    # ------------------------------------------------------------------

    def create_dashboard_data(self):
        """Prepare data for visualization dashboard"""
        recs = self.get_pricing_recommendations()
        if recs.empty:
            return None
            
        summary = {
            'total_products': len(recs),
            'price_increases': len(recs[recs['price_change'] > 0]),
            'price_decreases': len(recs[recs['price_change'] < 0]),
            'avg_margin': recs['margin'].mean(),
            'total_projected_revenue': recs['revenue'].sum(),
            'total_projected_profit': recs['profit'].sum(),
        }
        
        category_analysis = recs.groupby('category').agg({
            'price_change_pct': 'mean', 
            'margin': 'mean', 
            'revenue': 'sum', 
            'profit': 'sum'
        }).reset_index()
        
        strategy_dist = recs['strategy'].value_counts().to_dict()
        
        return {
            'recommendations': recs,
            'summary': summary,
            'category_analysis': category_analysis,
            'strategy_distribution': strategy_dist
        }


# ----------------------------------------------------------------------
# Visualization functions
# ----------------------------------------------------------------------

def create_pricing_visualizations(engine):
    """Create comprehensive visualizations for the pricing engine"""
    dashboard_data = engine.create_dashboard_data()
    if not dashboard_data:
        return None

    recs = dashboard_data['recommendations']
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Price Change Distribution', 
            'Margin by Category', 
            'Strategy Distribution', 
            'Price vs Demand Curve'
        ),
        specs=[[{"type": "histogram"}, {"type": "box"}], 
               [{"type": "pie"}, {"type": "scatter"}]]
    )
    
    # Price change distribution
    fig.add_trace(
        go.Histogram(x=recs['price_change_pct'], name='Price Change %', nbinsx=20),
        row=1, col=1
    )
    
    # Margin by category
    for category in recs['category'].unique():
        fig.add_trace(
            go.Box(y=recs[recs['category'] == category]['margin'], name=category),
            row=1, col=2
        )
    
    # Strategy distribution
    fig.add_trace(
        go.Pie(
            labels=list(dashboard_data['strategy_distribution'].keys()), 
            values=list(dashboard_data['strategy_distribution'].values())
        ),
        row=2, col=1
    )
    
    # Price vs demand curve for a sample product
    if not recs.empty:
        sample_product = recs.iloc[0]['product_id']
        price_range = np.linspace(
            recs[recs['product_id'] == sample_product]['current_price'].iloc[0] * 0.5,
            recs[recs['product_id'] == sample_product]['current_price'].iloc[0] * 1.5,
            20
        )
        
        # Create features for the sample product
        sample_features = {}
        for col in engine.feature_columns:
            sample_features[col] = recs[recs['product_id'] == sample_product].iloc[0].get(col, 0)
        
        demand_curve = []
        for price in price_range:
            sample_features['price'] = price
            demand = engine.predict_demand(sample_product, sample_features)
            demand_curve.append