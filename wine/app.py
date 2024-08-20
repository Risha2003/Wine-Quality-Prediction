from flask import Flask, render_template,request
import pickle
import numpy as np
#initialize the app

app=Flask(__name__)

@app.route('/',methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        fixed_acidity=float(request.form['fixed_acidity'])
        volatile_acidity=float(request.form['volatile_acidity'])
        citric_acid=float(request.form['citric_acid'])
        residual_sugar=float(request.form['residual_sugar'])
        chlorides=float(request.form['chlorides'])
        free_sulphur_dioxide=float(request.form['free_sulphur_dioxide'])
        total_sulphur_dioxide=float(request.form['total_sulphur_dioxide'])
        density=float(request.form['density'])
        pH=float(request.form['pH'])
        sulphates=float(request.form['sulphates'])
        alcohol=float(request.form['alcohol'])
        dummy_feature = float(request.form['dummy_feature'])
        #load the pickle file
        filename='F:\wine.html\wine\pipe.pkl'
        rfc=pickle.load(open(filename,'rb'))
        red=np.array([[dummy_feature,fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,free_sulphur_dioxide, total_sulphur_dioxide,density,pH,sulphates,alcohol]])
        my_prediction=rfc.predict(red)
        #get the result template
        print(my_prediction)
        return render_template('result.html',prediction=my_prediction)
if __name__ == '__main__':
    app.run(debug=True)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    #http://127.0.0.1:5000/