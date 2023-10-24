import os
import pickle
from loan_prediction_model import meta_ensemble, scaler
from django.conf import settings
import joblib


model_file_dir = os.path.join(settings.PICKLES_DIR_PATH, 'loan_prediction_model.pkl')
scaler_file_dir = os.path.join(settings.PICKLES_DIR_PATH, 'loan_scaler.pkl')

with open(model_file_dir, 'wb') as model_file:
    # pickle.dump(meta_ensemble, model_file)
    joblib.dump(meta_ensemble, model_file)

with open(scaler_file_dir, 'wb') as scaler_file:
    # pickle.dump(scaler, scaler_file)
    joblib.dump(scaler, scaler_file)

# with open(model_file_dir, 'rb') as model_file:
#     loaded_model = pickle.load(model_file)
#     print(loaded_model)
#
# with open(scaler_file_dir, 'rb') as model_file:
#     loaded_model = pickle.load(model_file)
#     print(loaded_model)
