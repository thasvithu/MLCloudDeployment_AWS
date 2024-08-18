# app.py

from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the model and scaler
model = pickle.load(open('admission_model.pickle', 'rb'))
scaler = pickle.load(open('scaler.pickle', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Extract features from the form
    gre = float(request.form['gre'])
    toefl = float(request.form['toefl'])
    university_rating = float(request.form['university_rating'])
    sop = float(request.form['sop'])
    lor = float(request.form['lor'])
    cgpa = float(request.form['cgpa'])
    research = float(request.form['research'])

    # Create a feature array from the user inputs
    features = np.array([[gre, toefl, university_rating, sop, lor, cgpa, research]])

    # Standardize the features using the loaded scaler
    standardized_features = scaler.transform(features)

    # Make prediction
    prediction = model.predict(standardized_features)

    # Convert to percentage
    output = round(prediction[0] * 100, 2)

    return render_template('result.html', prediction=output)


if __name__ == "__main__":
    app.run(debug=True)
