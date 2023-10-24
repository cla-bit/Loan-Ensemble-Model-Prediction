import os
import numpy as np
import joblib
from django.conf import settings


model_file_dir = None
scaler_file_dir = None

try:
    model_file_dir = os.path.join(settings.PICKLES_DIR_PATH, 'loan_prediction_model.pkl')
    with open(model_file_dir, 'rb') as model_file:
        model = joblib.load(model_file)
except FileNotFoundError as e:
    print(f"Error: {e}")

try:
    scaler_file_dir = os.path.join(settings.PICKLES_DIR_PATH, 'loan_scaler.pkl')
    with open(scaler_file_dir, 'rb') as scaler_file:
        scaler = joblib.load(scaler_file)
except FileNotFoundError as e:
    print(f"Error: {e}")

# Load the model and scaler
# model = joblib.load(model_file_dir)


def predict_loan_approval(data):
    # Preprocess the input data (data should be a dictionary)
    print(data)
    input_data = [
        # data['Gender'],
        # data['Married'],
        # data['Dependents'],
        # data['Education'],
        # data['Self_Employed'],
        data['ApplicantIncome'],
        # data['CoApplicantIncome'],
        # data['LoanAmount'],
        data['Loan_Amount_Term'],
        data['Credit_History'],
        data['Property_Area'],
    ]

    try:
        input_data = [int(x) for x in input_data]
        input_data = np.array(input_data).reshape(1, -1)
        input_data = scaler.fit_transform(input_data)
        prediction = model.predict(input_data)
        return prediction
    except Exception as e:
        print(f"Error: {e}")
        return

