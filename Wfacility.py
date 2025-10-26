import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

data = []

with open('epa_ghgrp_2021_2023_aggregate.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        data.append({
            'state': row['state'],
            'reporting_year': int(row['reporting_year']),
            'industry_sector': row['industry_sector'],
            'facility_name': row['facility_name'],
            'total_emissions': float(row['total_ghg_emissions_tonnes'])
        })

df = pd.DataFrame(data)

# add the labels for each attribute we are using
le_state = LabelEncoder()
le_sector = LabelEncoder()
le_year = LabelEncoder()
le_facility = LabelEncoder()

df['state_encoded'] = le_state.fit_transform(df['state'])
df['sector_encoded'] = le_sector.fit_transform(df['industry_sector'])
df['year_encoded'] = le_year.fit_transform(df['reporting_year'])
df['facility_encoded'] = le_facility.fit_transform(df['facility_name'])


attributes = df[['reporting_year', 'state_encoded', 'sector_encoded', 'facility_encoded']]
target = df['total_emissions']


attributes_train, attributes_test, target_train, target_test = train_test_split(attributes, target, test_size=0.2, random_state=42)

rf = RandomForestRegressor(
    n_estimators = 100, # number of trees created
    max_depth = 19, # max depth of tree
    # min_samples_split= 5, # min sample to split a node
    # min_samples_leaf= 3, # min sample per leaf
    random_state = 42, #
)

rf.fit(attributes_train, target_train)

target_pred_train = rf.predict(attributes_train)
target_pred_test = rf.predict(attributes_test)

print("\n=== Facility Model Performance ===")
print(f"Training R²: {r2_score(target_train, target_pred_train):.4f}")
delta = r2_score(target_train, target_pred_train) - r2_score(target_test, target_pred_test)
print(f'Delta R²: {delta:.2f}')
print(f"Testing  R²: {r2_score(target_test, target_pred_test):.4f}")
print(f"Testing  MAE: {mean_absolute_error(target_test, target_pred_test):,.2f} tonnes")


import joblib
joblib.dump(rf, "rf_model.pkl")

