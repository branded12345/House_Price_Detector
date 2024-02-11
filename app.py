from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

#Firstly, we will import ridge and scaler pickle file
ridge_model= pickle.load(open('models/ridge.pkl', 'rb'))
standard_Scaler= pickle.load(open('models/scaler.pkl', 'rb'))

#Route for Home Page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods= ['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        crim= float(request.form.get('crim'))
        zn= float(request.form.get('zn'))
        indus= float(request.form.get('indus'))
        chas= float(request.form.get('chas'))
        nox= float(request.form.get('nox'))
        rm= float(request.form.get('rm'))
        age= float(request.form.get('age'))
        dis= float(request.form.get('dis'))
        rad= float(request.form.get('rad'))
        tax= float(request.form.get('tax'))
        ptratio= float(request.form.get('ptratio'))
        b= float(request.form.get('b'))
        lstat= float(request.form.get('lstat'))

        new_data_scaled= standard_Scaler.transform([[crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat]])
        result= ridge_model.predict(new_data_scaled)

        return render_template('home.html', result= result[0])

    else:
        return render_template('home.html')


if __name__=="__main__":
    app.run(host="0.0.0.0")
