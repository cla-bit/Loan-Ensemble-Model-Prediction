from django.conf import settings

import joblib
import numpy as np


# load the trained model
try:
    model = joblib.load(settings.MODEL_PICKLE_PATH)
    scaler = joblib.load(settings.SCALER_PICKLE_PATH)
except FileNotFoundError:
    raise FileNotFoundError("Both Scaler and Model files are not found")


def predict_loan_approval(data):
    # Preprocess the input data (data should be a dictionary)
    user_data = [
        data['Gender'],
        data['Married'],
        data['Dependents'],
        data['Education'],
        data['Self_Employed'],
        data['ApplicantIncome'],
        data['CoApplicantIncome'],
        data['LoanAmount'],
        data['Loan_Amount_Term'],
        data['Credit_History'],
        data['Property_Area'],
    ]

    try:
        input_data = [int(x) for x in user_data]
        input_data = np.array(input_data).reshape(1, -1)
        input_data = scaler.fit_transform(input_data)
        prediction = model.predict(input_data)
        return prediction
    except Exception as e:
        print(f"Error: {e}")
        return

