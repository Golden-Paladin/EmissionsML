# model_base_no_facility_random.py

import csv
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ===============================
# Load Data
# ===============================
print("Loading data...")
data = []
with open('../epa_ghgrp_2021_2023_aggregate.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        data.append({
            'state': row['state'],
            'reporting_year': int(row['reporting_year']),
            'industry_sector': row['industry_sector'],
            'total_emissions': float(row['total_ghg_emissions_tonnes'])
        })

df = pd.DataFrame(data)
print(f"Total records: {len(df):,}")

# ===============================
# Encode categorical features
# ===============================
le_state = LabelEncoder()
le_sector = LabelEncoder()

df['state_encoded'] = le_state.fit_transform(df['state'])
df['sector_encoded'] = le_sector.fit_transform(df['industry_sector'])

# ===============================
# Features and Target
# ===============================
X = df[['reporting_year', 'state_encoded', 'sector_encoded']]
y = df['total_emissions']

# ===============================
# Use a different random seed each run
# ===============================
seed = int(time.time())
print(f"\nUsing random seed: {seed}")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=25,
    random_state=seed,  # randomizes forest each run
    n_jobs=-1
)

print("\nTraining Base Model (no facility)...")
rf.fit(X_train, y_train)

# ===============================
# Evaluate Model
# ===============================
y_pred_train = rf.predict(X_train)
y_pred_test = rf.predict(X_test)

print("\n=== Base Model Performance ===")
print(f"Training R²: {r2_score(y_train, y_pred_train):.4f}")
print(f"Testing  R²: {r2_score(y_test, y_pred_test):.4f}")
print(f"Testing  MAE: {mean_absolute_error(y_test, y_pred_test):,.2f} tonnes")
print(f"Testing  RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_test)):,.2f} tonnes")
