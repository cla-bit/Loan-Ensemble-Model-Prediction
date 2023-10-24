from django import forms


class LoadApprovalForm(forms.Form):
    Gender = forms.ChoiceField(choices=[(0, 'Male'), (1, 'Female')])
    Married = forms.ChoiceField(choices=[(0, 'No'), (1, 'Yes')])
    Dependents = forms.IntegerField()
    Education = forms.ChoiceField(choices=[(0, 'Not Graduate'), (1, 'Graduate')])
    Self_Employed = forms.ChoiceField(choices=[(0, 'No'), (1, 'Yes')])
    ApplicantIncome = forms.IntegerField()
    CoApplicantIncome = forms.IntegerField()
    LoanAmount = forms.FloatField()
    Loan_Amount_Term = forms.FloatField()
    Credit_History = forms.ChoiceField(choices=[(0, 'No'), (1, 'Yes')])
    Property_Area = forms.ChoiceField(choices=[(0, 'Rural'), (1, 'SemiUrban'), (2, 'Urban')])

    # def clean(self):
    #     cleaned_data = super().clean()
    #     applicant_income = cleaned_data.get('ApplicantIncome', 0)
    #     co_applicant_income = cleaned_data.get('CoApplicantIncome', 0)
    #     total_income = applicant_income + co_applicant_income
    #     cleaned_data['Total_Income'] = total_income
    #     return cleaned_data

