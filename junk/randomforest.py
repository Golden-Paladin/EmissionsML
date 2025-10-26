import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd

# Load and prepare data
data = []
with open('../epa_ghgrp_2021_2023_aggregate.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        data.append({
            'state': row['state'],
            'reporting_year': int(row['reporting_year']),
            'facility_name': row['facility_name'],
            'industry_sector': row['industry_sector'],
            'total_emissions': float(row['total_ghg_emissions_tonnes'])
        })

# Convert to DataFrame for easier processing
df = pd.DataFrame(data)

# Prepare features
# Encode categorical variables
le_state = LabelEncoder()
le_sector = LabelEncoder()
le_facility = LabelEncoder()

df['state_encoded'] = le_state.fit_transform(df['state'])
df['sector_encoded'] = le_sector.fit_transform(df['industry_sector'])
df['facility_encoded'] = le_facility.fit_transform(df['facility_name'])

# Select features for the model
X = df[['reporting_year', 'state_encoded', 'sector_encoded', 'facility_encoded']]
y = df['total_emissions']

# Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")
print()

# Train the model
# Using Random Forest Regressor (good for tabular data)
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)

print("Training model...")
model.fit(X_train, y_train)
print("Model trained!")
print()

# Make predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Evaluate the model
print("=== Model Performance ===")
print("\nTraining Set:")
print(f"R² Score: {r2_score(y_train, y_pred_train):.4f}")
print(f"MAE: {mean_absolute_error(y_train, y_pred_train):,.2f} tonnes")
print(f"RMSE: {np.sqrt(mean_squared_error(y_train, y_pred_train)):,.2f} tonnes")

print("\nTest Set:")
print(f"R² Score: {r2_score(y_test, y_pred_test):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred_test):,.2f} tonnes")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_test)):,.2f} tonnes")

# Feature importance
print("\n=== Feature Importance ===")
feature_names = ['reporting_year', 'state', 'industry_sector', 'facility_name']
importances = model.feature_importances_
for name, importance in sorted(zip(feature_names, importances),
                               key=lambda x: x[1], reverse=True):
    print(f"{name}: {importance:.4f}")

# Example prediction for a new data point
print("\n=== Example Prediction ===")
# Predict for first test sample
sample_idx = 0
sample = X_test.iloc[sample_idx:sample_idx + 1]
prediction = model.predict(sample)[0]
actual = y_test.iloc[sample_idx]

print(f"Predicted emissions: {prediction:,.2f} tonnes")
print(f"Actual emissions: {actual:,.2f} tonnes")
print(f"Error: {abs(prediction - actual):,.2f} tonnes")


# Function to make predictions for new data
def predict_emissions(year, state, industry_sector, facility_name):
    """
    Predict emissions for a company given input attributes
    """
    try:
        state_enc = le_state.transform([state])[0]
        sector_enc = le_sector.transform([industry_sector])[0]
        facility_enc = le_facility.transform([facility_name])[0]

        input_data = np.array([[year, state_enc, sector_enc, facility_enc]])
        prediction = model.predict(input_data)[0]

        return prediction
    except ValueError as e:
        return f"Error: Unknown value encountered. {str(e)}"


# Example usage of prediction function
print("\n=== Custom Prediction Example ===")
print("To predict for new data, use:")
print("predict_emissions(year=2024, state='CA', industry_sector='Power Plants', facility_name='Some Facility')")



