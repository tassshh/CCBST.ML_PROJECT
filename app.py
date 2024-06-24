import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import Ridge
from flask import Flask, request, jsonify, render_template

app=Flask(__name__)

# upload pickle models from the folder
ridge_model = pickle.load(open('models/ridge_project.pkl', 'rb'))
scaler_model = pickle.load(open('models/scaler_project.pkl', 'rb'))


# open home page

@app.route('/')
def home_page():
    return render_template('index.html')

#prediction page
@app.route('/predict', methods=['POST', 'GET'])
def predict_datapoint():
    if request.method == 'POST':
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain=float(request.form.get('Rain'))
        FFMC=float(request.form.get('FFMC'))
        DMC=float(request.form.get('DMC'))
        ISI=float(request.form.get('ISI'))
        Classes=float(request.form.get('Classes'))
        Region=float(request.form.get('Region'))

        new_data=scaler_model.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])

        result=ridge_model.predict(new_data)

        return render_template('index.html', Result=result[0])
   
    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5000)