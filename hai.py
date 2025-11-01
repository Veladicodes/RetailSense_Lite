import pandas as pd
from utils.advanced_forecasting import train_ensemble

# Load your processed dataset
df = pd.read_csv(r"F:\RetailSense_Lite\data\processed\cleaned_data.csv")
print("Columns:", df.columns.tolist())

# Select one product for forecasting
selected_product = df["product_name"].unique()[0]

# Filter that product's data
ts = df[df["product_name"] == selected_product][["week_start", "sales_qty"]]

# Rename to required columns for model
ts = ts.rename(columns={"week_start": "date", "sales_qty": "sales_qty"})

print("Selected product:", selected_product)
print("Shape of ts:", ts.shape)
print(ts.head())

# Run the forecast for 90 days (≈ 3 months)
result = train_ensemble(ts, horizon=90, fast_mode=True, debug=True)

print("Metrics:", result.metrics)
print("Details:", result.details)
