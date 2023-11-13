from django.shortcuts import render, redirect
from django.contrib.messages import success, error

from .forms import LoadApprovalForm
from .loan_predictor import predict_loan_approval

# Create your views here.


def loan_prediction(request):
    if request.method == 'POST':
        form = LoadApprovalForm(request.POST)
        if form.is_valid():
            # Get user input as a dictionary
            user_input = form.cleaned_data
            # Call the prediction function
            prediction = predict_loan_approval(user_input)
            # Check the prediction and set message based on the result
            if prediction[0] == 1:
                success(request, 'Your request for a Loan has been Approved! You can go to any of our'
                                 ' Branch to apply for a loan with ease!')
            else:
                error(request, 'Your request for a Loan has been Rejected!')
            return redirect('loan:loan_prediction')
    else:
        form = LoadApprovalForm()

    return render(request, 'loan-form.html', {'form': form})
