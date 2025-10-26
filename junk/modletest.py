import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate

# --- Load data ---
df = pd.read_csv('../epa_ghgrp_2021_2023_aggregate.csv')
df['reporting_year'] = df['reporting_year'].astype(int)
df['total_ghg_emissions_tonnes'] = df['total_ghg_emissions_tonnes'].astype(float)

# --- Encode categorical features ---
le_state = LabelEncoder()
le_sector = LabelEncoder()

df['state_encoded'] = le_state.fit_transform(df['state'])
df['sector_encoded'] = le_sector.fit_transform(df['industry_sector'])

# --- Features and target ---
X = df[['reporting_year', 'state_encoded', 'sector_encoded']]
y = df['total_ghg_emissions_tonnes']

# Optional: log-transform target to reduce scale impact
y_log = np.log1p(y)

# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)

# --- Scale numeric features ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Neural network ---
input_layer = Input(shape=(X_train_scaled.shape[1],))
x = Dense(64, activation='relu')(input_layer)
x = Dense(32, activation='relu')(x)
output_layer = Dense(1, activation='linear')(x)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# --- Train ---
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=200,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# --- Predict ---
y_pred_log = model.predict(X_test_scaled)
y_pred = np.expm1(y_pred_log)
y_test_orig = np.expm1(y_test)

# --- Evaluate ---
mae = mean_absolute_error(y_test_orig, y_pred)
r2 = r2_score(y_test_orig, y_pred)

print("\n=== Neural Network Performance ===")
print(f"Testing R²: {r2:.4f}")
print(f"Testing MAE: {mae:,.2f} tonnes")

# --- Prediction function ---
def predict_emissions(year, state, industry_sector):
    try:
        state_enc = le_state.transform([state])[0]
        sector_enc = le_sector.transform([industry_sector])[0]
        input_data = np.array([[year, state_enc, sector_enc]])
        input_scaled = scaler.transform(input_data)
        pred_log = model.predict(input_scaled)[0][0]
        pred = np.expm1(pred_log)
        return pred
    except ValueError as e:
        return f"Error: {str(e)}"

# --- Example prediction ---
sample_idx = 0
sample_year = X_test.iloc[sample_idx]['reporting_year']
sample_state = le_state.inverse_transform([X_test.iloc[sample_idx]['state_encoded']])[0]
sample_sector = le_sector.inverse_transform([X_test.iloc[sample_idx]['sector_encoded']])[0]

pred_value = predict_emissions(sample_year, sample_state, sample_sector)
actual_value = y_test_orig.iloc[sample_idx]

print("\n=== Example Prediction ===")
print(f"Input: Year={sample_year}, State={sample_state}, Sector={sample_sector}")
print(f"Predicted Output: {pred_value:,.2f} tonnes")
print(f"Actual Output: {actual_value:,.2f} tonnes")
print(f"Error: {abs(pred_value - actual_value):,.2f} tonnes")
