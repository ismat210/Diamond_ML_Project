# src/components/model_trainer.py
# ===========================
# Model Trainer for Regression
# ===========================
import os
import sys
from dataclasses import dataclass
import logging
import numpy as np

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor,
    VotingRegressor
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import r2_score

from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models, print_evaluated_results, model_metrics


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'regression_model.pkl')


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info("Splitting train & test arrays")
            xtrain, ytrain = train_array[:, :-1], train_array[:, -1]
            xtest, ytest = test_array[:, :-1], test_array[:, -1]

            # ===========================
            # Base Models
            # ===========================
            models = {
                "Linear Regression": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "KNN": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(random_state=42),
                "XGBRegressor": XGBRegressor(),
                "CatBoost": CatBoostRegressor(verbose=False),
                "GradientBoosting": GradientBoostingRegressor(),
                "AdaBoost": AdaBoostRegressor()
            }

            # ===========================
            # Evaluate Models
            # ===========================
            model_report = evaluate_models(xtrain, ytrain, xtest, ytest, models, problem_type="regression")

            print("\nRegression Model Performance Report:")
            print(model_report)
            print("="*80)

            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            print(f"Best Model: {best_model_name}, R2 Score: {best_model_score:.4f}")

            # ===========================
            # Hyperparameter Tuning - CatBoost
            # ===========================
            print("Tuning CatBoost Regressor...")
            cbr = CatBoostRegressor(verbose=False)
            param_dist = {
                'depth': [4, 5, 6, 7],
                'learning_rate': [0.01, 0.02, 0.03],
                'iterations': [300, 400, 500]
            }
            rscv = RandomizedSearchCV(cbr, param_dist, scoring='r2', cv=3, n_jobs=-1, random_state=42)
            rscv.fit(xtrain, ytrain)
            best_cbr = rscv.best_estimator_
            print("Best CatBoost Params:", rscv.best_params_)
            print("Best CatBoost Score:", rscv.best_score_)

            # ===========================
            # Hyperparameter Tuning - KNN
            # ===========================
            print("Tuning KNN Regressor...")
            knn = KNeighborsRegressor()
            param_grid = {'n_neighbors': list(range(2, 21))}
            grid = GridSearchCV(knn, param_grid, cv=3, scoring='r2', n_jobs=-1)
            grid.fit(xtrain, ytrain)
            best_knn = grid.best_estimator_
            print("Best KNN Params:", grid.best_params_)
            print("Best KNN Score:", grid.best_score_)

            # ===========================
            # Voting Regressor Ensemble
            # ===========================
            print("Training Final Voting Regressor...")
            final_model = VotingRegressor(
                estimators=[('cbr', best_cbr), ('xgb', XGBRegressor()), ('knn', best_knn)],
                weights=[3, 2, 1]
            )
            final_model.fit(xtrain, ytrain)

            # ===========================
            # Evaluation
            # ===========================
            print("\nFinal Model Evaluation:")
            print_evaluated_results(xtrain, ytrain, xtest, ytest, final_model, problem_type="regression")

            # ===========================
            # Save Model
            # ===========================
            os.makedirs(os.path.dirname(self.config.trained_model_file_path), exist_ok=True)
            save_object(self.config.trained_model_file_path, final_model)
            print(f"\n✅ Regression Model saved at {self.config.trained_model_file_path}")

            # ===========================
            # Return Metrics
            # ===========================
            ytest_pred = final_model.predict(xtest)
            mae, rmse, r2 = model_metrics(ytest, ytest_pred)
            logging.info("Regression Model Training Completed Successfully")
            return mae, rmse, r2

        except Exception as e:
            logging.error("Error occurred in Regression Model Training")
            raise CustomException(e, sys)