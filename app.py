from flask import Flask, render_template, request
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load CSV to get label encoders
df = pd.read_csv('epa_ghgrp_2021_2023_aggregate.csv')

# Create LabelEncoders like in training
le_state = LabelEncoder()
le_sector = LabelEncoder()
le_year = LabelEncoder()
le_facility = LabelEncoder()

le_state.fit(df['state'])
le_sector.fit(df['industry_sector'])
le_year.fit(df['reporting_year'])
le_facility.fit(df['facility_name'])

# Load the trained Random Forest model
rf_model = joblib.load('rf_model.pkl')

@app.route("/", methods=['GET', 'POST'])
def index():
    prediction = None
    state_val = ""
    sector_val = ""
    year_val = ""

    if request.method == 'POST':
        state_val = request.form.get('state', '')
        sector_val = request.form.get('sector', '')
        year_val = request.form.get('year', '')

        if not state_val or not sector_val or not year_val:
            prediction = "Please fill out all fields."
            return render_template('index.html', prediction=prediction, state_val=state_val, sector_val=sector_val, year_val=year_val)

        try:
            year_int = int(year_val)
            state_enc = le_state.transform([state_val])[0]
            sector_enc = le_sector.transform([sector_val])[0]
            year_enc = le_year.transform([year_int])[0]
        except ValueError:
            prediction = "Invalid input. Please enter valid state, sector, and year."
            return render_template('index.html', prediction=prediction, state_val=state_val, sector_val=sector_val, year_val=year_val)
        except Exception:
            prediction = "Error encoding inputs. Check your spelling of state and sector."
            return render_template('index.html', prediction=prediction, state_val=state_val, sector_val=sector_val, year_val=year_val)

        # Default facility
        facility_enc = le_facility.transform([df['facility_name'].iloc[0]])[0]
        input_features = [[year_int, state_enc, sector_enc, facility_enc]]
        pred_emission = rf_model.predict(input_features)[0]
        prediction = f"Predicted Emissions: {pred_emission:,.2f} tonnes"

    return render_template('index.html', prediction=prediction, state_val=state_val, sector_val=sector_val, year_val=year_val)



if __name__ == "__main__":
    app.run(debug=True)
