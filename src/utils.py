# utils.py
import os
import sys
import numpy as np 
import pandas as pd
import dill
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import accuracy_score
from src.exception import CustomException
import sys
from src.logger import logging
from sklearn.metrics import classification_report

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        logging.info('Exception Occurred in load_object function')
        raise CustomException(e, sys)

def evaluate_models(xtrain, ytrain, xtest, ytest, models, problem_type="regression"):
    """
    Evaluate multiple models on training and test data.
    
    Args:
        xtrain, ytrain, xtest, ytest: np.arrays
        models: dict of {model_name: model_object}
        problem_type: "regression" or "classification"
    
    Returns:
        dict of {model_name: test_score}
    """
    try:
        report = {}
        for name, model in models.items():
            model.fit(xtrain, ytrain)
            y_test_pred = model.predict(xtest)

            if problem_type == "regression":
                score = r2_score(ytest, y_test_pred)
            elif problem_type == "classification":
                score = accuracy_score(ytest, y_test_pred)
            else:
                raise ValueError("problem_type must be 'regression' or 'classification'")

            report[name] = score

        return report
    except Exception as e:
        logging.info('Exception occurred during model evaluation')
        raise CustomException(e, sys)
    
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score, classification_report

def model_metrics(true, predicted, problem_type="regression"):
    """
    Evaluates model performance for regression or classification.
    
    Args:
        true: true target values
        predicted: predicted target values
        problem_type: "regression" or "classification"
    
    Returns:
        Regression: mae, rmse, r2
        Classification: accuracy, classification_report string
    """
    try:
        if problem_type == "regression":
            mae = mean_absolute_error(true, predicted)
            mse = mean_squared_error(true, predicted)
            rmse = np.sqrt(mse)
            r2_square = r2_score(true, predicted)
            return mae, rmse, r2_square
        
        elif problem_type == "classification":
            acc = accuracy_score(true, predicted)
            report = classification_report(true, predicted)
            return acc, report
        
        else:
            raise ValueError("problem_type must be 'regression' or 'classification'")
    
    except Exception as e:
        logging.info('Exception Occurred while evaluating metric')
        raise CustomException(e, sys)


def print_evaluated_results(xtrain, ytrain, xtest, ytest, model, problem_type="regression"):
    """
    Prints evaluation results for regression or classification models.
    
    Args:
        xtrain, ytrain, xtest, ytest: np.arrays
        model: trained model object
        problem_type: "regression" or "classification"
    """
    try:
        ytrain_pred = model.predict(xtrain)
        ytest_pred = model.predict(xtest)

        if problem_type == "regression":
            # Evaluate regression
            model_train_mae, model_train_rmse, model_train_r2 = model_metrics(ytrain, ytrain_pred)
            model_test_mae, model_test_rmse, model_test_r2 = model_metrics(ytest, ytest_pred)

            print('Model performance for Training set (Regression)')
            print("- Root Mean Squared Error: {:.4f}".format(model_train_rmse))
            print("- Mean Absolute Error: {:.4f}".format(model_train_mae))
            print("- R2 Score: {:.4f}".format(model_train_r2))
            print('----------------------------------')
            print('Model performance for Test set (Regression)')
            print("- Root Mean Squared Error: {:.4f}".format(model_test_rmse))
            print("- Mean Absolute Error: {:.4f}".format(model_test_mae))
            print("- R2 Score: {:.4f}".format(model_test_r2))

        elif problem_type == "classification":
            # Evaluate classification
            train_acc = accuracy_score(ytrain, ytrain_pred)
            test_acc = accuracy_score(ytest, ytest_pred)

            print('Model performance for Training set (Classification)')
            print(f"- Accuracy: {train_acc:.4f}")
            print("- Classification Report:\n", classification_report(ytrain, ytrain_pred))
            print('----------------------------------')
            print('Model performance for Test set (Classification)')
            print(f"- Accuracy: {test_acc:.4f}")
            print("- Classification Report:\n", classification_report(ytest, ytest_pred))

        else:
            raise ValueError("problem_type must be 'regression' or 'classification'")

    except Exception as e:
        logging.info('Exception occurred during printing of evaluated results')
        raise CustomException(e, sys)