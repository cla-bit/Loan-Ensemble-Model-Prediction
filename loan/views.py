import os
import joblib
from django.conf import settings
from django.shortcuts import render, redirect
from django.contrib.messages import success, error
from .forms import LoadApprovalForm
from .loan_predictor import predict_loan_approval


# Create your views here.

# load the trained model
model = joblib.load(os.path.join(settings.PICKLES_DIR_PATH, 'loan_prediction_model.pkl'))
scaler = joblib.load(os.path.join(settings.PICKLES_DIR_PATH, 'loan_scaler.pkl'))


def loan_prediction(request):
    MESSAGE = ''
    if request.method == 'POST':
        form = LoadApprovalForm(request.POST)
        if form.is_valid():
            # Get user input as a dictionary
            user_input = form.cleaned_data
            # Call the prediction function
            prediction = predict_loan_approval(user_input)
            # Check the prediction and set message based on the result
            if prediction[0] == 1:
                success(request, 'Your request for a Loan has been Approved!')
                MESSAGE = 'Your request for a Loan has been Approved! You can go to any of our ' \
                          'Branch to apply for a loan with ease!'
            else:
                error(request, 'Your request for a Loan has been Rejected!')
                MESSAGE = 'Your request for a Loan has been Rejected!'
            # Reset the form
            form = LoadApprovalForm()
    else:
        form = LoadApprovalForm()

    return render(request, 'loan-form.html', {'form': form, 'MESSAGE': MESSAGE})

