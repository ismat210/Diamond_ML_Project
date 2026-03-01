# ===========================
# Basic Imports
# ===========================
import numpy as np
import pandas as pd
import os
import sys
from dataclasses import dataclass

# ===========================
# Model Imports
# ===========================
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor,
    VotingRegressor
)
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import r2_score

from catboost import CatBoostRegressor
from xgboost import XGBRegressor

# ===========================
# Project Imports
# ===========================
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_models
from src.utils import print_evaluated_results
from src.utils import model_metrics


# ===========================
# Config Class
# ===========================
@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')


# ===========================
# Model Trainer Class
# ===========================
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initate_model_training(self, train_array, test_array):
        try:
            logging.info("Splitting train & test arrays")

            xtrain, ytrain, xtest, ytest = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # =====================================
            # Base Models
            # =====================================
            models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "KNN": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoost": CatBoostRegressor(verbose=False),
                "GradientBoosting": GradientBoostingRegressor(),
                "AdaBoost": AdaBoostRegressor()
            }

            # =====================================
            # Evaluate Models
            # =====================================
            model_report = evaluate_models(xtrain, ytrain, xtest, ytest, models)

            print("\nModel Performance Report:")
            print(model_report)
            print("=" * 80)

            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]

            print(f"Best Model: {best_model_name}")
            print(f"Best R2 Score: {best_model_score}")
            print("=" * 80)

            # Do NOT stop training even if score is low
            if best_model_score < 0.6:
                print("⚠ Warning: Best model R2 score < 0.6, continuing anyway.")

            # =====================================
            # Hyperparameter Tuning - CatBoost
            # =====================================
            print("Tuning CatBoost...")

            cbr = CatBoostRegressor(verbose=False)

            param_dist = {
                'depth': [4,5,6,7,8,9,10],
                'learning_rate': [0.01,0.02,0.03,0.04],
                'iterations': [300,400,500,600]
            }

            rscv = RandomizedSearchCV(
                cbr,
                param_dist,
                scoring='r2',
                cv=5,
                n_jobs=-1
            )

            rscv.fit(xtrain, ytrain)

            best_cbr = rscv.best_estimator_

            print("Best CatBoost Params:", rscv.best_params_)
            print("Best CatBoost Score:", rscv.best_score_)
            print("=" * 80)

            # =====================================
            # Hyperparameter Tuning - KNN
            # =====================================
            print("Tuning KNN...")

            knn = KNeighborsRegressor()
            param_grid = {'n_neighbors': list(range(2, 31))}

            grid = GridSearchCV(knn, param_grid, cv=5, scoring='r2', n_jobs=-1)
            grid.fit(xtrain, ytrain)

            best_knn = grid.best_estimator_

            print("Best KNN Params:", grid.best_params_)
            print("Best KNN Score:", grid.best_score_)
            print("=" * 80)

            # =====================================
            # Final Ensemble Model
            # =====================================
            print("Training Final Voting Regressor...")

            final_model = VotingRegressor(
                estimators=[
                    ('cbr', best_cbr),
                    ('xgb', XGBRegressor()),
                    ('knn', best_knn)
                ],
                weights=[3, 2, 1]
            )

            final_model.fit(xtrain, ytrain)

            print("\nFinal Model Evaluation:")
            print_evaluated_results(xtrain, ytrain, xtest, ytest, final_model)

            # =====================================
            # Save Model
            # =====================================
            os.makedirs(
                os.path.dirname(self.model_trainer_config.trained_model_file_path),
                exist_ok=True
            )

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=final_model
            )

            print("\n✅ Model saved successfully at artifacts/model.pkl")

            # =====================================
            # Final Metrics
            # =====================================
            ytest_pred = final_model.predict(xtest)
            mae, rmse, r2 = model_metrics(ytest, ytest_pred)

            print("\nFinal Test Metrics:")
            print(f"MAE  : {mae}")
            print(f"RMSE : {rmse}")
            print(f"R2   : {r2}")

            logging.info("Model Training Completed Successfully")

            return mae, rmse, r2

        except Exception as e:
            logging.error("Error occurred in Model Training")
            raise CustomException(e, sys)