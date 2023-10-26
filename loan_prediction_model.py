import os
from imblearn.over_sampling import SMOTE
import joblib
import pandas as pd
from django.conf import settings
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB

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

# scale the data
scaler = StandardScaler()

# split into features and target
X = load_dataset.drop('Loan_Status', axis=1)
X = scaler.fit_transform(X)

Y = load_dataset['Loan_Status']

# handling class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, Y)

# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=4)

# Hyperparameter tuning for individual models
# Example for RandomForestClassifier
rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
ada_params = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 1.0]
}
grd_params = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 1.0],
    'max_depth': [3, 5, 7]
}
log_params = {
    'C': [0.1, 1, 10]
}
svm_params = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}
# Create a dictionary of classifiers and their respective parameter grids
rand_classifiers = {
    'Random Forest': (RandomForestClassifier(), rf_params),
    'AdaBoost': (AdaBoostClassifier(), ada_params),
    'Gradient Boosting': (GradientBoostingClassifier(), grd_params),
    'Logistic Regression': (LogisticRegression(), log_params),
    'SVM': (SVC(), svm_params)
}

rand_best_classifiers = {}

# Loop through each classifier and perform RandomizedSearchCV
for name, (classifier, param_dist) in rand_classifiers.items():
    random_search = RandomizedSearchCV(classifier, param_distributions=param_dist, n_iter=10, cv=5, n_jobs=-1,
                                       random_state=42)
    random_search.fit(X_train, y_train)
    rand_best_classifiers[name] = random_search.best_estimator_


# Train the individual models
random_forest_classifier = rand_best_classifiers['Random Forest']
adaboost_classifier = rand_best_classifiers['AdaBoost']
gradient_boosting_classifier = rand_best_classifiers['Gradient Boosting']
logistic_regression = rand_best_classifiers['Logistic Regression']
naive_bayes_classifier = GaussianNB()
svm_classifier = rand_best_classifiers['SVM']

random_forest_classifier.fit(X_train, y_train)
adaboost_classifier.fit(X_train, y_train)
gradient_boosting_classifier.fit(X_train, y_train)
logistic_regression.fit(X_train, y_train)
naive_bayes_classifier.fit(X_train, y_train)
svm_classifier.fit(X_train, y_train)

# Make predictions using individual models
random_forest_classifier_prediction = random_forest_classifier.predict(X_test)
adaboost_classifier_prediction = adaboost_classifier.predict(X_test)
gradient_boosting_classifier_prediction = gradient_boosting_classifier.predict(X_test)
logistic_regression_prediction = logistic_regression.predict(X_test)
naive_bayes_classifier_prediction = naive_bayes_classifier.predict(X_test)
svm_classifier_prediction = svm_classifier.predict(X_test)

# analyze the models for accuracy, f1 score, precision
random_forest_accuracy = accuracy_score(y_test, random_forest_classifier_prediction)
adaboost_accuracy = accuracy_score(y_test, adaboost_classifier_prediction)
gradient_boosting_accuracy = accuracy_score(y_test, gradient_boosting_classifier_prediction)
logistic_regression_accuracy = accuracy_score(y_test, logistic_regression_prediction)
naive_bayes_accuracy = accuracy_score(y_test, naive_bayes_classifier_prediction)
svm_accuracy = accuracy_score(y_test, svm_classifier_prediction)

random_forest_f1_score = f1_score(y_test, random_forest_classifier_prediction)
adaboost_f1_score = f1_score(y_test, adaboost_classifier_prediction)
gradient_boosting_f1_score = f1_score(y_test, gradient_boosting_classifier_prediction)
logistic_regression_f1_score = f1_score(y_test, logistic_regression_prediction)
naive_bayes_f1_score = f1_score(y_test, naive_bayes_classifier_prediction)
svm_f1_score = f1_score(y_test, svm_classifier_prediction)

random_forest_precision_score = precision_score(y_test, random_forest_classifier_prediction)
adaboost_precision_score = precision_score(y_test, adaboost_classifier_prediction)
gradient_boosting_precision_score = precision_score(y_test, gradient_boosting_classifier_prediction)
logistic_regression_precision_score = precision_score(y_test, logistic_regression_prediction)
naive_bayes_precision_score = precision_score(y_test, naive_bayes_classifier_prediction)
svm_precision_score = precision_score(y_test, svm_classifier_prediction)

# Cross-validation
classifiers = {'Random Forest': random_forest_classifier, 'Adaboost': adaboost_classifier,
               'Gradient Boosting': gradient_boosting_classifier, 'Logistic Regression': logistic_regression,
               'Naive Bayes': naive_bayes_classifier, 'SVM': svm_classifier}
accuracies_mean_score = {}
improved_classifiers = {}
best_accuracy = 0
for name, classifier in classifiers.items():
    scores = cross_val_score(classifier, X_train, y_train, cv=10, scoring='accuracy')
    mean_score = scores.mean()

    accuracies_mean_score[name] = mean_score

    if mean_score > best_accuracy:
        best_accuracy = mean_score
        improved_classifiers = {name: classifier}
    elif mean_score == best_accuracy:
        improved_classifiers[name] = classifier

# Print the accuracies and improved classifiers
print("Classifier Accuracies:")
for name, accuracy in accuracies_mean_score.items():
    print(f"{name}: {accuracy:.3f}")

# Create a meta-ensemble model (Voting Classifier)
meta_ensemble = VotingClassifier(estimators=[
    (name, classifier) for name, classifier in improved_classifiers.items()], voting='hard')

# Train the meta-ensemble on the predictions of the base models
meta_ensemble.fit(X_train, y_train)

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
