import os
import numpy as np
import joblib
from django.conf import settings


# load the trained model
model = joblib.load(os.path.join(settings.PICKLES_DIR_PATH, 'loan_prediction_model.pkl'))
scaler = joblib.load(os.path.join(settings.PICKLES_DIR_PATH, 'loan_scaler.pkl'))


def predict_loan_approval(data):
    # Preprocess the input data (data should be a dictionary)
    print(data)
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

