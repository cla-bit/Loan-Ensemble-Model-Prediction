from django.urls import path
from .views import loan_prediction


app_name = 'loan'

urlpatterns = [
    path('', loan_prediction, name='loan_prediction'),
]
