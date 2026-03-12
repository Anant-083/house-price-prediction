# House Price Prediction
# Import libraries
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load dataset
print("Loading data...")
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['Price'] = housing.target

# Explore data
print("\nFirst 5 rows of data:")
print(df.head())
print("\nShape of data:", df.shape)
print("\nBasic statistics:")
print(df.describe())

# Prepare data
X = df.drop('Price', axis=1)
y = df['Price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# Train model
print("\nTraining model...")
model = LinearRegression()
model.fit(X_train, y_train)

# Test model
y_pred = model.predict(X_test)

# Results
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print("\n--- MODEL RESULTS ---")
print(f"R2 Score: {r2:.2f}")
print(f"MSE: {mse:.2f}")
print(f"Accuracy: {r2*100:.1f}%")

# Plot results
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.tight_layout()
plt.savefig('results.png')
print("\nGraph saved as results.png")
print("Done!")