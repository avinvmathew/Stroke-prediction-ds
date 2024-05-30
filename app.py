from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('final_model.sav', 'rb'))

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define the route to handle form submission and make prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extract form data
        age = float(request.form['age'])
        gender = request.form['gender']
        hypertension = int(request.form['hypertension'])
        heart_disease = int(request.form['heart_disease'])
        ever_married = request.form['ever_married']
        work_type = request.form['work_type']
        Residence_type = request.form['Residence_type']
        avg_glucose_level = float(request.form['avg_glucose_level'])
        bmi = float(request.form['bmi'])
        smoking_status = request.form['smoking_status']

        # Create a DataFrame for the input data
        input_data = pd.DataFrame([[gender, hypertension, heart_disease, ever_married, work_type, Residence_type, smoking_status, age, avg_glucose_level, bmi]], 
                                  columns=['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'smoking_status', 'age', 'avg_glucose_level', 'bmi'])

        # Apply the same encoding and scaling as the training data
        le = LabelEncoder()
        for col in ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']:
            input_data[col] = le.fit_transform(input_data[col])

        # Scale the numerical data
        input_data = pd.DataFrame(scale_data(input_data))

        # Make prediction
        prediction = model.predict(input_data)[0]
        result = 'Stroke' if prediction == 1 else 'No Stroke'
        
        return render_template('index.html', result=result)

def scale_data(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    scaled_data = pd.DataFrame(scaled_data, columns=data.columns)
    return scaled_data

if __name__ == '__main__':
    app.run(debug=True)
