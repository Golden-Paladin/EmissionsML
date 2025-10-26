import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd

# store the values state, reporting year, facility name, industry sector, total, emissions
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


df = pd.DataFrame(data)

# encode the state, industry sector and facility
# takes the catogorical labels into number for the random forest
labelState = LabelEncoder()
labelSector = LabelEncoder()
le_facility = LabelEncoder()

df['state_encoded'] = labelState.fit_transform(df['state'])
df['sector_encoded'] = labelSector.fit_transform(df['industry_sector'])
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
        state_enc = labelState.transform([state])[0]
        sector_enc = labelSector.transform([industry_sector])[0]
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

# ===== VISUALIZATION =====
import matplotlib.pyplot as plt

print("\n=== Generating Performance Visualizations ===")

# 1. Actual vs Predicted (Test Set)
plt.figure(figsize=(10, 8))
plt.scatter(y_test, y_pred_test, alpha=0.5, s=20)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Emissions (tonnes)', fontsize=12)
plt.ylabel('Predicted Emissions (tonnes)', fontsize=12)
plt.title('Actual vs Predicted - Test Set', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('1_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
print("Saved: 1_actual_vs_predicted.png")
plt.show()

# 2. Residual Plot (Test Set)
plt.figure(figsize=(10, 8))
residuals = y_test - y_pred_test
plt.scatter(y_pred_test, residuals, alpha=0.5, s=20)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Predicted Emissions (tonnes)', fontsize=12)
plt.ylabel('Residuals (tonnes)', fontsize=12)
plt.title('Residual Plot - Test Set', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('2_residual_plot.png', dpi=300, bbox_inches='tight')
print("Saved: 2_residual_plot.png")
plt.show()

# 3. Feature Importance
plt.figure(figsize=(10, 8))
feature_names = ['Reporting Year', 'State', 'Industry Sector', 'Facility Name']
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
plt.ylabel('Importance', fontsize=12)
plt.title('Feature Importance', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('3_feature_importance.png', dpi=300, bbox_inches='tight')
print("Saved: 3_feature_importance.png")
plt.show()

# 4. Distribution of Errors (Test Set)
plt.figure(figsize=(10, 8))
errors = np.abs(y_test - y_pred_test)
plt.hist(errors, bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Absolute Error (tonnes)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Distribution of Prediction Errors', fontsize=14, fontweight='bold')
plt.axvline(np.mean(errors), color='r', linestyle='--', lw=2, label=f'Mean: {np.mean(errors):,.0f}')
plt.axvline(np.median(errors), color='g', linestyle='--', lw=2, label=f'Median: {np.median(errors):,.0f}')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('4_error_distribution.png', dpi=300, bbox_inches='tight')
print("Saved: 4_error_distribution.png")
plt.show()

# 5. Prediction Error by Year
plt.figure(figsize=(10, 8))
test_df = pd.DataFrame({
    'year': X_test['reporting_year'],
    'error': np.abs(y_test - y_pred_test)
})
year_errors = test_df.groupby('year')['error'].mean()
plt.bar(year_errors.index, year_errors.values, edgecolor='black', alpha=0.7)
plt.xlabel('Reporting Year', fontsize=12)
plt.ylabel('Mean Absolute Error (tonnes)', fontsize=12)
plt.title('Average Prediction Error by Year', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('5_error_by_year.png', dpi=300, bbox_inches='tight')
print("Saved: 5_error_by_year.png")
plt.show()

# 6. R² Comparison (Train vs Test)
plt.figure(figsize=(10, 8))
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
datasets = ['Training Set', 'Test Set']
r2_scores = [r2_train, r2_test]
colors = ['#2ecc71', '#3498db']
bars = plt.bar(datasets, r2_scores, color=colors, edgecolor='black', alpha=0.7)
plt.ylabel('R² Score', fontsize=12)
plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
plt.ylim([0, 1])
plt.grid(True, alpha=0.3, axis='y')
# Add value labels on bars
for bar, score in zip(bars, r2_scores):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height,
             f'{score:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
plt.tight_layout()
plt.savefig('6_r2_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: 6_r2_comparison.png")
plt.show()

print("\n✓ All plots generated successfully!")