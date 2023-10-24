# Loan-Ensemble-Model-Prediction: Django App

<p> The "Loan-Ensemble-Model-Prediction" Djnago App is a robust and sophisticated Django web application designed to assist financial institutions, lenders, and borrowers in making informed decisions regarding loan applications and repayment strategies. This app leverages the power of ensemble machine learning techniques to enhance the accuracy of loan repayment predictions, ultimately contributing to more prudent financial decisions.</p>

<h4> Key Features </h4>
<ul>
	<li>Loan Application: Borrowers can submit loan applications by providing essential personal and financial information. The app validates and processes these applications, ensuring data accuracy.</li>
	<li>Data Collection and Integration: The application integrates with various data sources, including credit bureaus, to collect comprehensive borrower data. This data is used to assess creditworthiness.</li>
	<li>Ensemble Algorithm: The heart of the application lies in its ensemble algorithm, which combines multiple predictive models such as Random Forest, Gradient Boosting, and Logistic Regression, AdaBoost and Voting Classifier. This ensemble technique enhances the reliability of loan repayment predictions</li>
</ul>

<h3>How to run this app</h3>
This project is to be run using Dokcer. In case Dokcer is not available, the project can be done using the following steps:

<ol>
	<li>Create and activate a virtual environment and install the requirements:
		<code>pip install requirements.txt</code>
	</li>
	<li>Run the loan_prediction_model.py:
		<code>python manage.py shell < loan_prediction_model.py</code>
	</li>
	<li>Run the django runserver command:
		<code>python manage.py runserver</code>
	</li>
</0l>
