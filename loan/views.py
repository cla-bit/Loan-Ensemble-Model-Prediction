import os
import joblib
from django.conf import settings
from django.shortcuts import render
from .forms import LoadApprovalForm

# Create your views here.

# load the trained model
model = joblib.load(os.path.join(settings.PICKLES_DIR_PATH, 'loan_prediction_model.pkl'))


def loan_prediction(request):
    if request.method == 'POST':
        form = LoadApprovalForm(request.POST)
        if form.is_valid():
            # Get user input as a dictionary
            user_input = form.cleaned_data
            print(user_input.keys())
            print(user_input.values())
            user_input_data = [
                user_input['Gender'],
                user_input['Married'],
                user_input['Dependents'],
                user_input['Education'],
                user_input['Self_Employed'],
                user_input['ApplicantIncome'],
                user_input['CoApplicantIncome'],
                user_input['LoanAmount'],
                user_input['Loan_Amount_Term'],
                user_input['Credit_History'],
                user_input['Property_Area'],
            ]
            # user_input_int = [int(x) for x in user_input.values()]
            # user_input_array = [np.array(user_input_int)]
            prediction = model.predict([user_input_data])[0]
            print(prediction)

            # Call the prediction function
            # prediction = predict_loan_approval(user_input)
            # print(f"Prediction: {prediction[0]}")
            # return render(request, 'result.html', {'prediction': prediction})
    else:
        form = LoadApprovalForm()

    return render(request, 'loan-form.html', {'form': form})

