import numpy as np
import pandas as pd
import pickle
from flask import Flask, request, render_template
import os

# Create Flask app
app = Flask(__name__)

# Load model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scale = pickle.load(open('scale.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def predict():
    try:
        # Get input values from form
        input_features = [float(x) for x in request.form.values()]

        # Define correct column names in correct order
        feature_names = [
        'holiday', 'temp', 'rain', 'snow', 'weather',
        'day', 'month', 'year', 'hours', 'minutes', 'seconds'
        ]


        # Convert input to DataFrame
        input_df = pd.DataFrame([input_features], columns=feature_names)

        # Apply scaler
        scaled_df = scale.transform(input_df)

        # Predict
        prediction = model.predict(scaled_df)
        output = round(prediction[0], 2)

        return render_template("result.html", prediction_text=f"{output}")

    except Exception as e:
       return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, port=port)

