CatBoost Loan Prediction

This project demonstrates a machine learning pipeline for predicting loan approval status using the CatBoost classifier. The process includes data preprocessing, exploratory data analysis (EDA), and model training.

Project Structure

    Catboost_loan_prediction(1).ipynb: A Jupyter Notebook containing the full code for this project.

Key Features

    Data Preprocessing: The notebook handles data cleaning, including dropping unnecessary columns (loan_id) and cleaning whitespace from column names.

    Feature Engineering: Categorical features like education, self_employed, and loan_status are converted to the category data type.

    Exploratory Data Analysis (EDA): The notebook includes basic EDA to understand the dataset, checking for null values and duplicates.

    Modeling: The project uses the CatBoostClassifier for prediction. While predicting without hnadling class imbalance for one of the model and another that uses imblearn.over_sampling.SMOTE to handle potential class imbalance and sklearn.model_selection.GridSearchCV for hyperparameter tuning.

    Accuracy: The modeles provided accuracy of abouit 96 percent across F1-Score,recall and precision

Requirements

To run this notebook, you will need the following libraries:

    pandas

    numpy

    matplotlib

    catboost

    imblearn

    sklearn

You can install these dependencies using pip:

!pip install catboost
!pip install imblearn
!pip install scikit-learn

Dataset

The model is trained on a dataset from a CSV file named loan_approval_dataset.csv. This dataset is sourced from a kaggle dataset(https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset( and contains information about various factors influencing loan approval, such as income, assets, and CIBIL score.
