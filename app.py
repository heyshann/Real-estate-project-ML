
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle

data=pd.read_csv('location.csv')
X = pickle.load(open("list.pkl", 'rb'))
model=pickle.load(open('banglore_home_prices_model.pickle',"rb"))

def predict_price(location,sqft,bath,bhk):    
    loc_index = np.where(X.columns==location)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return model.predict([x])[0]

app = Flask(__name__)



@app.route('/')
def index():

    locations=sorted(data['location'].unique())    

    return render_template('index.html',locations=locations)


@app.route('/predict',methods=['POST'])
def predict():
    location=request.form.get("location")
    sqft=request.form.get("total_sqft")
    bath=request.form.get("bath")
    bhk=request.form.get("bhk")

    results=abs(predict_price(location,sqft,bath,bhk))
    res = "{:.2f}".format(results)
    return render_template('predict.html',result=res)

if __name__=="__main__":
    app.run(debug=True)