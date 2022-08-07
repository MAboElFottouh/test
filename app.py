import numpy as np
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
import pickle
from app import app
from app import routes

app = Flask(__name__)
model = pickle.load(open('E:/Downloads/Flask/model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    print(float(x) for x in request.form.values())
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction
    print(output)

    return render_template('index.html', prediction_text='{}'.format(output))



app.run(debug=True)
