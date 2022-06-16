
import numpy as np
import pickle
import pandas as pd
from flask import Flask, request
import flasgger
from flasgger import Swagger
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
Swagger(app)

# loading the model
pickle_in = open("model.pkl","rb")
log_reg = pickle.load(pickle_in)

pickle_in = open("scaler.pkl","rb")
sc = pickle.load(pickle_in)


@app.route('/')
def hello():
    return "Mid Term Test!!"


@app.route('/predict_test', methods=["POST"])
def predict_test_class():
    
    """
    Prediction for the Cancer dataset
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: file
        in: formData
        type: file
        required: true
    responses:
        200:
            description: The output values
        
    """
    
    df_test = pd.read_csv(request.files.get("file"))
    
    # Introducing feature scaling

    df_test = sc.transform(df_test)
    prediction = log_reg.predict(df_test)
    
    # Used this line of code for debugging the output of the prediction
    print(prediction)
    return " The Predicated Class for the TestFile is"+ str(list(prediction))


if __name__=='__main__':
    app.run()