from django.shortcuts import render
from .forms import LoadApprovalForm
from .loan_predictor import predict_loan_approval
import numpy as np
from django.conf import settings
import os
import joblib

# Create your views here.


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


def loan_prediction(request):
    if request.method == 'POST':
        form = LoadApprovalForm(request.POST)
        if form.is_valid():
            # Get user input as a dictionary
            user_input = form.cleaned_data
            print(user_input.keys())
            print(user_input.values())
            user_input_int = [int(x) for x in user_input.values()]
            user_input_array = [np.array(user_input_int)]
            prediction = model.predict(user_input_array)
            print(prediction[0])

            # Call the prediction function
            # prediction = predict_loan_approval(user_input)
            print(f"Prediction: {prediction[0]}")
            # return render(request, 'result.html', {'prediction': prediction})
    else:
        form = LoadApprovalForm()

    return render(request, 'loan-form.html', {'form': form})

