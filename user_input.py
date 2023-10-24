import os
import joblib
from django.conf import settings
import pandas as pd
import pickle


# model_file_dir = os.path.join(settings.PICKLES_DIR_PATH, 'loan_prediction_model.pkl')
model_file_dir = os.path.join('pickles_dir', 'loan_prediction_model.pkl')
# scaler_file_dir = os.path.join(settings.PICKLES_DIR_PATH, 'loan_scaler.pkl')
scaler_file_dir = os.path.join('pickles_dir', 'loan_scaler.pkl')
# Load the loan predictive model
# model = joblib.load('loan_predictive_model.pkl')
# scaler_model = joblib.load('')
with open(model_file_dir, 'rb') as model_file:
    loaded_model = pickle.load(model_file)
    print(loaded_model)

with open(scaler_file_dir, 'rb') as scaler_file:
    loaded_scaler = pickle.load(scaler_file)
    print(loaded_scaler)


def get_input(prompt):
    while True:
        user_input = input(prompt)
        if user_input:
            return user_input


def get_numeric_input(prompt):
    while True:
        user_input = input(prompt)
        if user_input.isnumeric():
            return int(user_input)
        else:
            print("Please enter a valid numeric value.")


def get_yes_no_input(prompt):
    while True:
        user_input = input(prompt).strip().lower()
        if user_input in ['yes', 'y']:
            return 'Y'
        elif user_input in ['no', 'n']:
            return 'N'
        else:
            print("Please enter 'yes' or 'no'.")


def main():
    print("Welcome to the Loan Application Simulator!")

    # Collect user input for loan application features
    gender = get_numeric_input("Gender (Male/Female): ")
    married = get_numeric_input("Married (Yes/No): ")
    dependents = get_numeric_input("Number of Dependents: ")
    education = get_numeric_input("Education (Graduate/Not Graduate): ")
    self_employed = get_numeric_input("Self Employed (Yes/No): ")
    applicant_income = get_numeric_input("Applicant Income: ")
    coapplicant_income = get_numeric_input("Coapplicant Income: ")
    loan_amount = get_numeric_input("Loan Amount: ")
    loan_amount_term = get_numeric_input("Loan Amount Term (in months): ")
    credit_history = get_numeric_input("Credit History (1 for Yes, 0 for No): ")
    property_area = get_numeric_input("Property Area (Urban/Rural/SemiUrban): ")

    # Prepare user input for prediction
    user_input_data = pd.DataFrame({
        'Gender': [gender],
        'Married': [married],
        'Dependents': [dependents],
        'Education': [education],
        'Self_Employed': [self_employed],
        'ApplicantIncome': [applicant_income],
        'CoApplicantIncome': [coapplicant_income],
        'LoanAmount': [loan_amount],
        'Loan_Amount_Term': [loan_amount_term],
        'Credit_History': [credit_history],
        'Property_Area': [property_area]
    })

    # Label encoding
    user_input_data['Gender'] = user_input_data['Gender'].map({'Male': 0, 'Female': 1}).astype(int)
    user_input_data['Married'] = user_input_data['Married'].map({'No': 0, 'Yes': 1}).astype(int)
    user_input_data['Education'] = user_input_data['Education'].map({'Not Graduate': 0, 'Graduate': 1}).astype(int)
    user_input_data['Self_Employed'] = user_input_data['Self_Employed'].map({'No': 0, 'Yes': 1}).astype(int)
    user_input_data['Property_Area'] = user_input_data['Property_Area'].map({'Urban': 0, 'Rural': 1, 'SemiUrban': 2}).astype(int)

    # Scale the user input data
    # scaler = StandardScaler()
    # user_input_data = scaler.fit_transform(user_input_data)
    user_input_data = loaded_scaler.fit_transform(user_input_data)

    # Make a loan approval prediction
    # prediction = model.predict(user_input_data)
    prediction = loaded_model.predict(user_input_data)

    if prediction[0] == 1:
        print("Congratulations! Your loan application is approved.")
    else:
        print("Sorry, your loan application is not approved.")


if __name__ == "__main__":
    main()
