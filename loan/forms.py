from django import forms


class LoadApprovalForm(forms.Form):
    Gender = forms.ChoiceField(choices=[(0, 'Male'), (1, 'Female')], label='What is your Gender?')
    Married = forms.ChoiceField(choices=[(0, 'No'), (1, 'Yes')], label='Are you Married?')
    Dependents = forms.IntegerField(label='How many Dependents do you live with?')
    Education = forms.ChoiceField(choices=[(0, 'Not Graduate'), (1, 'Graduate')], label='What is your level of Education?')
    Self_Employed = forms.ChoiceField(choices=[(0, 'No'), (1, 'Yes')], label='Are you Self-Employed?')
    ApplicantIncome = forms.IntegerField(label='What is your monthly Income?')
    CoApplicantIncome = forms.IntegerField(label='What is your monthly Co-Applicant Income?')
    LoanAmount = forms.FloatField(label='How much Loan do you want to apply?')
    Loan_Amount_Term = forms.FloatField(label='What is your Loan Term?')
    Credit_History = forms.ChoiceField(choices=[(0, 'No'), (1, 'Yes')], label='Do you have Credit History?')
    Property_Area = forms.ChoiceField(choices=[(0, 'Urban'), (1, 'Rural'), (2, 'SemiUrban')], label='What is your Property Area?')

    # def clean(self):
    #     cleaned_data = super().clean()
    #     applicant_income = cleaned_data.get('ApplicantIncome', 0)
    #     co_applicant_income = cleaned_data.get('CoApplicantIncome', 0)
    #     total_income = applicant_income + co_applicant_income
    #     cleaned_data['Total_Income'] = total_income
    #     return cleaned_data

