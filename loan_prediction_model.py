import os
import joblib
import pandas as pd
from django.conf import settings
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Data Path and loadin gof the dataset path
DATA_PATH = None
try:
    DATA_PATH = os.path.join(settings.DATASET_DIR_PATH, 'train.csv')
except FileNotFoundError as e:
    print(f"Error: {e}")

load_dataset = pd.read_csv(DATA_PATH)


# Remove the 'Loan_ID' column
load_dataset.drop('Loan_ID', axis=1, inplace=True)

# Filling up the missing values in the columns and rows affected
load_dataset['Gender'] = load_dataset['Gender'].fillna(load_dataset['Gender'].mode()[0])
load_dataset['Married'] = load_dataset['Married'].fillna(load_dataset['Married'].mode()[0])
load_dataset['Dependents'] = load_dataset['Dependents'].str.replace(r"\+", "", regex=True)
load_dataset['Dependents'] = pd.to_numeric(load_dataset['Dependents'])
load_dataset['Dependents'] = load_dataset['Dependents'].fillna(load_dataset['Dependents'].mode()[0])
load_dataset['Self_Employed'] = load_dataset['Self_Employed'].fillna(load_dataset['Self_Employed'].mode()[0])
load_dataset['LoanAmount'] = load_dataset['LoanAmount'].fillna(load_dataset['LoanAmount'].mean())
load_dataset['Loan_Amount_Term'] = load_dataset['Loan_Amount_Term'].fillna(load_dataset['Loan_Amount_Term'].mean())
load_dataset['Credit_History'] = load_dataset['Credit_History'].fillna(load_dataset['Credit_History'].mean())

# label encoding
load_dataset['Gender'] = load_dataset['Gender'].map({'Male': 0, 'Female': 1}).astype(int)
load_dataset['Married'] = load_dataset['Married'].map({'No': 0, 'Yes': 1}).astype(int)
load_dataset['Dependents'] = load_dataset['Dependents'].astype(int)
load_dataset['Education'] = load_dataset['Education'].map({'Not Graduate': 0, 'Graduate': 1}).astype(int)
load_dataset['Self_Employed'] = load_dataset['Self_Employed'].map({'No': 0, 'Yes': 1}).astype(int)
load_dataset['ApplicantIncome'] = load_dataset['ApplicantIncome'].astype(int)
load_dataset['CoApplicantIncome'] = load_dataset['CoApplicantIncome'].astype(int)
load_dataset['Credit_History'] = load_dataset['Credit_History'].astype(int)
load_dataset['Property_Area'] = load_dataset['Property_Area'].map({'Urban': 0, 'Rural': 1, 'SemiUrban': 2}).astype(int)
load_dataset['Loan_Status'] = load_dataset['Loan_Status'].map({'N': 0, 'Y': 1}).astype(int)

print(load_dataset.columns)
print(load_dataset.corr())

scaler = StandardScaler()

# split into features and target
X = load_dataset.drop('Loan_Status', axis=1)
X = scaler.fit_transform(X)

Y = load_dataset['Loan_Status']

# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=4)

# Train the individual models

# Random Forest Classifier Model
random_forest_classifier = RandomForestClassifier(n_estimators=100)
random_forest_classifier.fit(X_train, y_train)
random_forest_classifier_prediction = random_forest_classifier.predict(X_test)

# Adaboost Classifier Model
adaboost_classifier = AdaBoostClassifier(n_estimators=100)
adaboost_classifier.fit(X_train, y_train)
adaboost_classifier_prediction = adaboost_classifier.predict(X_test)

# Gradient Boosting Classifier Model
gradient_boosting_classifier = GradientBoostingClassifier(n_estimators=100)
gradient_boosting_classifier.fit(X_train, y_train)
gradient_boosting_classifier_prediction = gradient_boosting_classifier.predict(X_test)

# Logistic Regression Model
logistic_regression = LogisticRegression()
logistic_regression.fit(X_train, y_train)
logistic_regression_prediction = logistic_regression.predict(X_test)


# Create a meta-ensemble model (Voting Classifier)
meta_ensemble = VotingClassifier(estimators=[
    ('rf', random_forest_classifier),
    ('gb', gradient_boosting_classifier),
    ('lr', logistic_regression),
    ('ab', adaboost_classifier)
], voting='soft')

# Train the meta-ensemble on the predictions of the base models
meta_ensemble.fit(
    X_train,
    y_train
)

# Make predictions with the meta-ensemble
meta_predictions = meta_ensemble.predict(X_test)


# Evaluate the ensemble model's accuracy
ensemble_accuracy = accuracy_score(y_test, meta_predictions)
f1_ensemble_accuracy = f1_score(y_test, meta_predictions)
precision_score_accuracy = precision_score(y_test, meta_predictions)

print("Ensemble Model Accuracy:", ensemble_accuracy)
print("Ensemble Model F1 Score:", f1_ensemble_accuracy)
print("Ensemble Model Precision Score:", precision_score_accuracy)

# Save the model and scaler
scaler_filename = 'loan_scaler.pkl'
model_filename = 'loan_prediction_model.pkl'

scaler_filename_dir = os.path.join(settings.PICKLES_DIR_PATH, scaler_filename)
model_filename_dir = os.path.join(settings.PICKLES_DIR_PATH, model_filename)

joblib.dump(scaler, scaler_filename_dir)
joblib.dump(meta_ensemble, model_filename_dir)

print(f"Scaler saved to {scaler_filename_dir} as {scaler_filename}")
print(f"Model saved to {model_filename_dir} as {model_filename}")
