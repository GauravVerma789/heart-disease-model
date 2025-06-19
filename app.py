from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open('heart_disease_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect and convert form inputs

        # Categorical mappings
        sex = 1 if request.form['sex'].lower() == 'male' else 0

        cp_map = {
            'typical': 0,
            'atypical': 1,
            'non-anginal': 2,
            'asymptomatic': 3
        }
        cp = cp_map.get(request.form['cp'].lower(), 0)

        thal_map = {
            'normal': 1,
            'fixed': 2,
            'reversible': 3
        }
        thal = thal_map.get(request.form['thal'].lower(), 1)

        restecg_map = {
            'normal': 0,
            'abnormal': 1,
            'hypertrophy': 2
        }
        restecg = restecg_map.get(request.form['restecg'].lower(), 0)

        slope_map = {
            'up': 0,
            'flat': 1,
            'down': 2
        }
        slope = slope_map.get(request.form['slope'].lower(), 0)

        # Continuous/numeric inputs
        age = float(request.form['age'])
        trestbps = float(request.form['trestbps'])
        chol = float(request.form['chol'])
        fbs = float(request.form['fbs'])
        thalch = float(request.form['thalch'])
        exang = float(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        ca = float(request.form['ca'])

        # Final feature list
        features = [age, sex, cp, trestbps, chol, fbs, restecg,
                    thalch, exang, oldpeak, slope, ca, thal]

        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)[0]

        result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"
        return render_template('result.html', prediction_text=result)

    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
