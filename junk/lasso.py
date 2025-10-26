# model_with_facility_random.py

import csv
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ===============================
# Configuration
# ===============================
CONFIG = {
    'data_file': 'epa_ghgrp_2021_2023_aggregate.csv',
    'test_size': 0.2,
    'n_estimators': 200,
    'max_depth': 25,
    'n_jobs': -1
}


# ===============================
# Load Data
# ===============================
def load_data(filename):
    """Load data from CSV file."""
    print("Loading data...")
    data = []
    with open(filename, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                data.append({
                    'state': row['state'],
                    'reporting_year': int(row['reporting_year']),
                    'industry_sector': row['industry_sector'],
                    'facility_name': row['facility_name'],
                    'total_emissions': float(row['total_ghg_emissions_tonnes'])
                })
            except (ValueError, KeyError) as e:
                # Skip rows with invalid data
                continue

    df = pd.DataFrame(data)
    print(f"Total records: {len(df):,}")

    # Data summary
    print(f"\nData Summary:")
    print(f"  Years: {df['reporting_year'].min()} - {df['reporting_year'].max()}")
    print(f"  Unique facilities: {df['facility_name'].nunique():,}")
    print(f"  Unique sectors: {df['industry_sector'].nunique()}")
    print(f"  Unique states: {df['state'].nunique()}")
    print(f"  Emissions range: {df['total_emissions'].min():,.0f} - {df['total_emissions'].max():,.0f} tonnes")

    return df


# ===============================
# Encode categorical features
# ===============================
def encode_features(df):
    """Encode categorical features using LabelEncoder."""
    print("\nEncoding categorical features...")

    le_state = LabelEncoder()
    le_sector = LabelEncoder()
    le_facility = LabelEncoder()

    df['state_encoded'] = le_state.fit_transform(df['state'])
    df['sector_encoded'] = le_sector.fit_transform(df['industry_sector'])
    df['facility_encoded'] = le_facility.fit_transform(df['facility_name'])

    print(f"  States encoded: {len(le_state.classes_)}")
    print(f"  Sectors encoded: {len(le_sector.classes_)}")
    print(f"  Facilities encoded: {len(le_facility.classes_):,}")

    return df, le_state, le_sector, le_facility


# ===============================
# Train Model
# ===============================
def train_model(X_train, y_train, seed, config):
    """Train Random Forest model."""
    print(f"\nTraining Random Forest Model...")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Training samples: {len(X_train):,}")

    rf = RandomForestRegressor(
        n_estimators=config['n_estimators'],
        max_depth=config['max_depth'],
        random_state=seed,
        n_jobs=config['n_jobs'],
        verbose=1
    )

    start_time = time.time()
    rf.fit(X_train, y_train)
    training_time = time.time() - start_time

    print(f"  Training completed in {training_time:.2f} seconds")

    return rf


# ===============================
# Evaluate Model
# ===============================
def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Evaluate model performance."""
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    # Calculate MAPE
    test_mape = np.mean(np.abs((y_test - y_pred_test) / (y_test + 1e-10))) * 100

    # Accuracy within thresholds
    within_10 = np.sum(np.abs((y_test - y_pred_test) / (y_test + 1e-10)) < 0.1) / len(y_test) * 100
    within_25 = np.sum(np.abs((y_test - y_pred_test) / (y_test + 1e-10)) < 0.25) / len(y_test) * 100
    within_50 = np.sum(np.abs((y_test - y_pred_test) / (y_test + 1e-10)) < 0.5) / len(y_test) * 100

    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE")
    print("=" * 60)
    print(f"Training R²:  {train_r2:.4f}")
    print(f"Testing  R²:  {test_r2:.4f}")
    print(f"Testing  MAE: {test_mae:,.2f} tonnes")
    print(f"Testing  RMSE: {test_rmse:,.2f} tonnes")
    print(f"Testing  MAPE: {test_mape:.2f}%")
    print("\nPrediction Accuracy:")
    print(f"  Within 10%: {within_10:.1f}% of predictions")
    print(f"  Within 25%: {within_25:.1f}% of predictions")
    print(f"  Within 50%: {within_50:.1f}% of predictions")
    print("=" * 60)

    return {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'test_mape': test_mape
    }


# ===============================
# Feature Importance
# ===============================
def analyze_feature_importance(model, feature_names):
    """Analyze and display feature importance."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE")
    print("=" * 60)
    for i, idx in enumerate(indices[:10]):
        print(f"{i + 1}. {feature_names[idx]:30s} {importances[idx]:.4f}")
    print("=" * 60)


# ===============================
# Prediction Functions
# ===============================
def predict_known_facility(model, year, state, sector, facility,
                           le_state, le_sector, le_facility):
    """Predict emissions for a known facility."""
    try:
        state_enc = le_state.transform([state])[0]
        sector_enc = le_sector.transform([sector])[0]
        facility_enc = le_facility.transform([facility])[0]

        X_input = np.array([[year, state_enc, sector_enc, facility_enc]])
        prediction = model.predict(X_input)[0]

        print(f"\nPrediction for KNOWN facility:")
        print(f"  Facility: {facility}")
        print(f"  Year: {year}, State: {state}, Sector: {sector}")
        print(f"  Estimated emissions: {prediction:,.2f} tonnes")

        return prediction
    except ValueError as e:
        print(f"Error: {e}")
        print("Make sure all inputs exist in the training data.")
        return None


def predict_unknown_facility(model, year, state, sector,
                             le_state, le_sector, le_facility):
    """Predict emissions for an unknown facility."""
    try:
        # Add 'Unknown_Facility' to encoder if not present
        if "Unknown_Facility" not in le_facility.classes_:
            le_facility.classes_ = np.append(le_facility.classes_, "Unknown_Facility")

        state_enc = le_state.transform([state])[0]
        sector_enc = le_sector.transform([sector])[0]
        facility_enc = le_facility.transform(["Unknown_Facility"])[0]

        X_input = np.array([[year, state_enc, sector_enc, facility_enc]])
        prediction = model.predict(X_input)[0]

        print(f"\nPrediction for UNKNOWN facility:")
        print(f"  Year: {year}, State: {state}, Sector: {sector}")
        print(f"  Estimated emissions: {prediction:,.2f} tonnes")
        print(f"  (Generic estimate based on sector/state/year)")

        return prediction
    except ValueError as e:
        print(f"Error: {e}")
        print("Make sure state and sector exist in the training data.")
        return None


# ===============================
# Main Execution
# ===============================
def main():
    print("=" * 60)
    print("GHG EMISSIONS PREDICTION MODEL")
    print("Random Forest with Facility Data")
    print("=" * 60)

    # Load data
    df = load_data(CONFIG['data_file'])

    # Encode features
    df, le_state, le_sector, le_facility = encode_features(df)

    # Prepare features and target
    X = df[['reporting_year', 'state_encoded', 'sector_encoded', 'facility_encoded']]
    y = df['total_emissions']

    # Use different random seed each run
    seed = int(time.time())
    print(f"\n{'=' * 60}")
    print(f"Using random seed: {seed}")
    print(f"{'=' * 60}")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=CONFIG['test_size'],
        random_state=seed
    )

    print(f"\nData Split:")
    print(f"  Training: {len(X_train):,} samples ({len(X_train) / len(X) * 100:.1f}%)")
    print(f"  Testing:  {len(X_test):,} samples ({len(X_test) / len(X) * 100:.1f}%)")

    # Train model
    model = train_model(X_train, y_train, seed, CONFIG)

    # Evaluate model
    metrics = evaluate_model(model, X_train, y_train, X_test, y_test)

    # Feature importance
    feature_names = ['reporting_year', 'state', 'sector', 'facility']
    analyze_feature_importance(model, feature_names)

    # ===============================
    # Test Predictions
    # ===============================
    print("\n" + "=" * 60)
    print("SAMPLE PREDICTIONS")
    print("=" * 60)

    # Get sample data from test set
    sample_idx = X_test.index[0]
    sample_year = df.loc[sample_idx, 'reporting_year']
    sample_state = df.loc[sample_idx, 'state']
    sample_sector = df.loc[sample_idx, 'industry_sector']
    sample_facility = df.loc[sample_idx, 'facility_name']
    actual_emissions = df.loc[sample_idx, 'total_emissions']

    # Prediction 1: Known facility
    print("\n1. Known Facility Test:")
    pred1 = predict_known_facility(
        model, sample_year, sample_state, sample_sector, sample_facility,
        le_state, le_sector, le_facility
    )
    if pred1:
        print(f"  Actual emissions: {actual_emissions:,.2f} tonnes")
        print(
            f"  Error: {abs(pred1 - actual_emissions):,.2f} tonnes ({abs(pred1 - actual_emissions) / actual_emissions * 100:.1f}%)")

    # Prediction 2: Unknown facility
    print("\n2. Unknown Facility Test:")
    pred2 = predict_unknown_facility(
        model, 2022, 'CA', 'Power Plants',
        le_state, le_sector, le_facility
    )

    # ===============================
    # Missing facility error test
    # ===============================
    print("\n" + "=" * 60)
    print("ERROR HANDLING TEST")
    print("=" * 60)
    print("\nAttempting prediction without facility feature:")
    sample = X_test.iloc[0][['reporting_year', 'state_encoded', 'sector_encoded']]
    try:
        model.predict([sample])
        print("  Warning: Model accepted incomplete input!")
    except ValueError as e:
        print("  Expected error caught:")
        print(f"  {str(e)[:100]}...")
        print("  ✓ Model properly requires facility feature")

    # Return objects for interactive use
    return model, le_state, le_sector, le_facility, df, metrics


# ===============================
# Run Script
# ===============================
if __name__ == "__main__":
    model, le_state, le_sector, le_facility, df, metrics = main()

    print("\n" + "=" * 60)
    print("MODEL READY FOR PREDICTIONS")
    print("=" * 60)
    print("\nAvailable functions:")
    print("  predict_known_facility(model, year, state, sector, facility,")
    print("                         le_state, le_sector, le_facility)")
    print("\n  predict_unknown_facility(model, year, state, sector,")
    print("                           le_state, le_sector, le_facility)")

    print("\n" + "=" * 60)
    print("SAMPLE FACILITIES")
    print("=" * 60)
    sample_facilities = df[['facility_name', 'industry_sector', 'state']].drop_duplicates().head(5)
    for idx, row in sample_facilities.iterrows():
        print(f"  {row['facility_name'][:50]:50s} | {row['industry_sector']:30s} | {row['state']}")

    print("\n" + "=" * 60)
    print("AVAILABLE SECTORS")
    print("=" * 60)
    sectors = sorted(df['industry_sector'].unique())
    for sector in sectors:
        print(f"  - {sector}")

    print("\n" + "=" * 60)
    print("AVAILABLE STATES")
    print("=" * 60)
    states = sorted(df['state'].unique())
    print(f"  {', '.join(states)}")