import os

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, redirect, flash, send_file
from sklearn.preprocessing import MinMaxScaler
from werkzeug.utils import secure_filename
import pickle

app = Flask(__name__) #Initialize the flask App

model = pickle.load(open('Naveen.pkl', 'rb'))
scaler = MinMaxScaler()

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/abstract')
def abstract():
    return render_template('abstract.html')

@app.route('/future')
def future():
    return render_template('future.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/preview', methods=["POST"])
def preview():
    if request.method == 'POST':
        dataset = request.files['datasetfile']
        df = pd.read_csv(dataset, encoding='unicode_escape')
        
        if 'Id' in df.columns:
            df.set_index('Id', inplace=True)
            
        # Features expected by the machine learning model
        required_cols = ['DAY_OF_MONTH', 'DAY_OF_WEEK', 'OP_CARRIER_AIRLINE_ID', 
                         'ORIGIN_AIRPORT_ID', 'DEST_AIRPORT_ID', 'DEP_TIME', 'ARR_TIME', 
                         'DEP_DEL15', 'DIVERTED', 'DISTANCE']
        
        try:
            # Check if all required columns exist in the uploaded CSV
            if all(col in df.columns for col in required_cols):
                X = df[required_cols].copy()
                
                # Convert time format from HH:MM to HHMM if the column is stored as strings
                for col in ['DEP_TIME', 'ARR_TIME']:
                    if X[col].dtype == object:
                        X[col] = X[col].astype(str).str.replace(':', '', regex=False).astype(float)
                        
                X = X.fillna(0) # Handle any missing values
                predictions = model.predict(X.values)
                
                # Append the predictions to the original dataframe
                df['Prediction'] = ["Delayed" if p == 1 else "On Time" for p in predictions]
            
            # Save the processed dataset so the user can download it
            os.makedirs('static/downloads', exist_ok=True)
            df.to_csv('static/downloads/predictions.csv')

            # Only show the top 100 rows in the preview to prevent the browser from freezing
            return render_template("preview.html", df_view=df.head(100))
            
        except Exception as e:
            return f"An error occurred while processing the dataset: {e}"

@app.route('/download_predictions')
def download_predictions():
    return send_file('static/downloads/predictions.csv', as_attachment=True)

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/test.html', methods=['GET', 'POST'])
def single():
    return render_template('test.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = []
        # Get form values
        for value in request.form.values():
            # Convert time format HH:MM to HHMM
            if ":" in value:
                hour, minute = value.split(":")
                value = int(hour) * 100 + int(minute)
            features.append(float(value))

        # Convert into numpy array
        final_features = [np.array(features)]

        # Prediction
        prediction = model.predict(final_features)

        # Extract Departure Delay input to give a smarter explanation
        dep_del15 = float(request.form.get('DEP_DEL15', 0))

        # Output
        if prediction[0] == 1:
            output = "Flight Will Be Delayed"
            if dep_del15 == 1:
                explanation = "Explanation: The flight already has a departure delay (15+ mins), which historically is the strongest indicator of a late arrival."
            else:
                explanation = "Explanation: Based on historical patterns, factors such as the airline, route congestion, or time of day indicate a high probability of delay."
        else:
            output = "Flight Will Arrive On Time"
            if dep_del15 == 1:
                explanation = "Explanation: Despite a delayed departure, historical data suggests the flight can make up time in the air and arrive as scheduled."
            else:
                explanation = "Explanation: With an on-time departure and favorable historical performance for this route, the flight is expected to arrive on time."

        return render_template(
            'test.html',
            prediction_text=output,
            explanation_text=explanation
        )
    except Exception as e:
        return render_template(
            'test.html',
            prediction_text=f"Error : {e}"
        )

@app.route('/chart/')
@app.route('/chart/<prediction_text>')
def hello(prediction_text=None):
    return render_template('chart.html', prediction=prediction_text)

@app.route('/existing')
def existing():
    return render_template('existing.html')

# DASHBOARD ROUTE
@app.route('/dashboard')
def dashboard():
    # Read datasets
    df1 = pd.read_csv('Jan_2019.csv')
    df2 = pd.read_csv('Jan_2020.csv')

    # Combine datasets
    df = pd.concat([df1, df2])

    # Remove unnamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # KPI DATA
    total_flights = len(df)
    total_delayed = len(df[df['ARR_DEL15'] == 1])
    delay_percentage = round(
        df['ARR_DEL15'].mean() * 100,
        2
    )

    best_airline = (
        df.groupby('OP_UNIQUE_CARRIER')['ARR_DEL15']
        .mean()
        .sort_values()
        .index[0]
    )

    worst_airport = (
        df.groupby('ORIGIN')['ARR_DEL15']
        .mean()
        .sort_values(ascending=False)
        .index[0]
    )

    # PREPARE DATA FOR INTERACTIVE CHART
    trend = df.groupby('DAY_OF_MONTH')['ARR_DEL15'].sum()
    labels = trend.index.tolist()
    values = trend.values.tolist()

    # Airline Chart Data (Top 10 Delayed Airlines)
    top_airlines = df.groupby('OP_UNIQUE_CARRIER')['ARR_DEL15'].sum().sort_values(ascending=False).head(10)
    airline_labels = top_airlines.index.tolist()
    airline_values = top_airlines.values.tolist()

    # Airport Chart Data (Top 10 Worst Airports)
    top_airports = df.groupby('ORIGIN')['ARR_DEL15'].sum().sort_values(ascending=False).head(10)
    airport_labels = top_airports.index.tolist()
    airport_values = top_airports.values.tolist()

    return render_template(
        'dashboard.html',
        total_flights=total_flights,
        total_delayed=total_delayed,
        delay_percentage=delay_percentage,
        best_airline=best_airline,
        worst_airport=worst_airport,
        labels=labels,
        values=values,
        airline_labels=airline_labels,
        airline_values=airline_values,
        airport_labels=airport_labels,
        airport_values=airport_values
    )

if __name__ == '__main__':
    app.run(debug=True)