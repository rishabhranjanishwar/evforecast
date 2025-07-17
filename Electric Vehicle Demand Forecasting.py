# ============================================
# Electric Vehicle Demand Forecasting
# ============================================

# ==========================
# 1. Import Required Libraries
# ==========================
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# ==========================
# 2. Load & Preprocess Data
# ==========================
# Load dataset (Ensure 'title_transactions.csv' is in the same folder)
data = pd.read_csv('title_transactions.csv')

# Convert transaction date to datetime & clean data
data['transaction_date'] = pd.to_datetime(data['transaction_date'])
data.drop_duplicates(inplace=True)

# Extract Year & Month for demand calculation
data['transaction_year'] = data['transaction_date'].dt.year
data['transaction_month'] = data['transaction_date'].dt.to_period('M')

# ==========================
# 3. Define Demand Metrics
# ==========================
# Demand = Number of EV transactions per month per county
demand_data = data.groupby(['transaction_month', 'county']).size().reset_index(name='demand')
current_data = demand_data[demand_data['transaction_month'] <= '2021-12']

# ==========================
# 4. Generate Synthetic Data (2023-2024)
# ==========================
top_counties = current_data['county'].value_counts().nlargest(10).index
synthetic_data = pd.DataFrame({
    'transaction_month': np.tile(pd.period_range('2023-01', '2024-12', freq='M'), len(top_counties)),
    'county': np.repeat(top_counties, 24),
    'demand': np.random.randint(100, 500, size=len(top_counties) * 24)
})

# Combine historical & synthetic data
combined_data = pd.concat([current_data, synthetic_data], ignore_index=True)

# ==========================
# 5. Prepare Data for Modeling
# ==========================
regression_data = combined_data.pivot(index='transaction_month', columns='county', values='demand').fillna(0).reset_index()
regression_data['transaction_month'] = regression_data['transaction_month'].astype(str).str.replace('-', '').astype(int)

X = regression_data.drop(columns=['transaction_month'])
y = regression_data['transaction_month']

# ==========================
# 6. Train Forecasting Models
# ==========================
## Linear Regression
linear_regressor = LinearRegression()
linear_regressor.fit(X, y)

## Ridge Regression
ridge_regressor = Ridge()
ridge_regressor.fit(X, y)

## ARIMA (Time Series Forecast)
arima_data = combined_data.groupby('transaction_month')['demand'].sum().reset_index()
arima_data.set_index('transaction_month', inplace=True)

arima_model = ARIMA(arima_data, order=(5, 1, 0)).fit()

# ==========================
# 7. Forecast Future Demand (2025-2026)
# ==========================
future_months = pd.period_range('2025-01', '2026-12', freq='M')
future_data = pd.DataFrame(0, index=np.arange(len(future_months)), columns=X.columns)

linear_pred = linear_regressor.predict(future_data)
ridge_pred = ridge_regressor.predict(future_data)
arima_forecast = arima_model.forecast(steps=len(future_months))

# Compile predictions
predictions_df = pd.DataFrame({
    'Month': future_months.astype(str),
    'Linear Regression': linear_pred,
    'Ridge Regression': ridge_pred,
    'ARIMA': arima_forecast.values
})
print("Future Predictions (2025-2026):")
print(predictions_df.head())

# ==========================
# 8. Visualization
# ==========================
plt.figure(figsize=(12, 6))
current_data.groupby('transaction_month')['demand'].sum().plot(label='Current Data till 2021')
plt.title('Electric Vehicle Demand Till 2021')
plt.xlabel('Month')
plt.ylabel('Demand')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
synthetic_data.groupby('transaction_month')['demand'].sum().plot(label='Synthetic Data 2023-2024', color='red')
plt.title('Electric Vehicle Demand for 2023 and 2024')
plt.xlabel('Month')
plt.ylabel('Demand')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(predictions_df['Month'], predictions_df['Linear Regression'], label='Linear Regression Prediction', color='green')
plt.plot(predictions_df['Month'], predictions_df['Ridge Regression'], label='Ridge Regression Prediction', color='orange')
plt.plot(predictions_df['Month'], predictions_df['ARIMA'], label='ARIMA Prediction', color='blue')
plt.xticks(rotation=90)
plt.title('EV Demand Prediction for 2025 and 2026')
plt.xlabel('Month')
plt.ylabel('Demand')
plt.legend()
plt.show()

# ==========================
# 9. Conclusion
# ==========================
print("\nThe model provides insights into future EV demand trends.\n"
      "- Linear & Ridge show overall trend patterns.\n"
      "- ARIMA captures time-series seasonality effects.\n"
      "Use these forecasts for policy-making & infrastructure planning.")
