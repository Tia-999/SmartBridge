from flask import Flask, render_template, request
import numpy as np
import pickle
import pandas as pd


from flask import Flask, request, render_template

import pickle
app = Flask(__name__)
model = pickle.load(open(r'thyroid_model.pkl', 'rb'))


@app.route('/')
def helloworld():
    return render_template("home.html")

@app.route('/predict', methods = ['POST'])
def predict():
    goitre = request.form['goitre']
    if(goitre == 'goitre_t'):
        goitre_t = 1
    else:
        goitre_t = 0
    tumor = request.form['tumor']
    if (tumor == 'tumor_t'):
        tumor_t = 1
    else:
        tumor_t = 0
        sex = request.form['sex']
    if (sex == "sex_M"):
        sex_M = 1
    else:
        sex_M = 0
    sick = request.form['sick']
    if (sick == 'sick_t'):
        sick_t = 1
    else:
        sick_t = 0
    pregnant = request.form['pregnant']
    if (pregnant == 'pregnant_t'):
        pregnant_t = 1
    else:
        pregnant_t = 0
    thyroid_surgery = request.form['thyroid_surgery']
    if (thyroid_surgery == 'thyroid_surgery_t'):
        thyroid_surgery_t = 1
    else:
        thyroid_surgery_t = 0
    TSH = float(request.form["TSH"])
    T3 = float(request.form["T3"])
    TT4 = float(request.form["TT4"])
    T4U = float(request.form["T4U"])
    FTI = float(request.form["FTI"])
    sex = request.form['sex']
    if (sex == "sex_M"):
        sex_M = 1
    else:
        sex_M = 0
    sick = request.form['sick']
    if (sick == 'sick_t'):
        sick_t = 1
    else:
        sick_t = 0
    pregnant = request.form['pregnant']
    if (pregnant == 'pregnant_t'):
        pregnant_t = 1
    else:
        pregnant_t = 0
    thyroid_surgery = request.form['thyroid_surgery']
    if (thyroid_surgery == 'thyroid_surgery_t'):
        thyroid_surgery_t = 1
    else:
        thyroid_surgery_t = 0
    goitre = request.form['goitre']
    prediction = model.predict([[TSH,
                                 T3,
                                 TT4,
                                 T4U,
                                 FTI,
                                 sex_M,
                                 sick_t,
                                 pregnant_t,
                                 thyroid_surgery_t,
                                 goitre_t,
                                 tumor_t]])
    output = prediction[0]
    if output == 0:
        print('Thyroid Classification Result : HYPERTHYROID')
        return render_template('home.html', y='YOU HAVE: HYPERTHYROID')
    elif output == 1:
        print('Thyroid Classification Result : HYPOTHYROID')
        return render_template('home.html', y='YOU HAVE: HYPOTHYROID')
    elif output == 2:
        print('Thyroid Classification Result : NEGATIVE')
        return render_template('home.html', y='YOU ARE: NEGATIVE')
    else:
        print('Thyroid Classification Result : SICK')
        return render_template('home.html', y='YOU ARE : SICK')

if __name__ == '__main__':
    app.run(debug=False)
