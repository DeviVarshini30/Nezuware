import yfinance as yf
import pandas as pd

# Download historical stock data for a particular stock, e.g., Apple (AAPL)
stock_data = yf.download('AAPL', start='2010-01-01', end='2023-12-31')
print(stock_data.head())

# Preprocessing: Handling missing values
stock_data = stock_data.dropna()

# Feature Engineering: Creating additional features
stock_data['Open-Close'] = stock_data['Open'] - stock_data['Close']
stock_data['High-Low'] = stock_data['High'] - stock_data['Low']

# Target variable: Close price
X = stock_data[['Open', 'High', 'Low', 'Volume', 'Open-Close', 'High-Low']]
y = stock_data['Close']


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model initialization and training
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')



import matplotlib.pyplot as plt

plt.figure(figsize=(14, 7))
plt.plot(y_test.index, y_test, label='Actual Price')
plt.plot(y_test.index, y_pred, label='Predicted Price')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title('Actual vs Predicted Stock Price')
plt.show()


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Step 1: Load historical stock data
# stock_data = pd.read_csv('"C:/Users/deviv/Downloads/archive (6)/historical_stock_data.csv"')
# # Incorrect path with double quotes
# # stock_data = pd.read_csv('"C:/Users/deviv/Downloads/archive (6)/historical_stock_data.csv"')

# # Corrected path without double quotes
stock_data = pd.read_csv('C:/Users/deviv/Downloads/archive (6)/historical_stock_data.csv')


# Step 2: Data preprocessing and feature engineering
# Assuming 'Date', 'Open', 'High', 'Low', 'Close', 'Volume' are columns in stock_data
# Example: Calculate additional features like 'Open-Close' and 'High-Low'
stock_data['Open-Close'] = stock_data['Open'] - stock_data['Close']
stock_data['High-Low'] = stock_data['High'] - stock_data['Low']

# Step 3: Define features (X) and target variable (y)
X = stock_data[['Open', 'High', 'Low', 'Volume', 'Open-Close', 'High-Low']]
y = stock_data['Close']

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Model selection and training
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Model evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Step 7: Visualization of results
plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test.values, label='Actual Close Price', color='blue')
plt.plot(y_test.index, y_pred, label='Predicted Close Price', color='red')
plt.title('Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()
