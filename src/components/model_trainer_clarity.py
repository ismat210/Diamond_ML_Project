import os
import sys
from dataclasses import dataclass
import numpy as np

# ===========================
# Model Imports
# ===========================
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    VotingClassifier
)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import accuracy_score

# ===========================
# Project Imports
# ===========================
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models, print_evaluated_results

# ===========================
# Config Class
# ===========================
@dataclass
class ModelTrainerClarityConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'clarity_model.pkl')

# ===========================
# Model Trainer Class
# ===========================
class ModelTrainerClarity:
    def __init__(self):
        self.model_trainer_config = ModelTrainerClarityConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting train & test arrays")

            xtrain, ytrain, xtest, ytest = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # ===========================
            # Base Models
            # ===========================
            models = {
                "Logistic Regression": LogisticRegression(max_iter=500),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(random_state=42),
                "KNN": KNeighborsClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "AdaBoost": AdaBoostClassifier()
            }

            # ===========================
            # Evaluate Models
            # ===========================
            model_report = evaluate_models(
                xtrain, ytrain, xtest, ytest, models, problem_type="classification"
            )

            print("\nClarity Model Performance Report:")
            print(model_report)
            print("=" * 80)

            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]

            print(f"Best Model: {best_model_name}")
            print(f"Best Accuracy: {best_model_score}")
            print("=" * 80)

            # ===========================
            # Hyperparameter Tuning - RandomForest
            # ===========================
            print("Tuning RandomForest...")
            rf = RandomForestClassifier(random_state=42)
            param_dist = {
                "n_estimators": [100, 200],
                "max_depth": [10, None],
                "min_samples_split": [2, 5]
            }
            rscv = RandomizedSearchCV(
                rf, param_dist, scoring="accuracy", cv=3, n_jobs=-1, random_state=42
            )
            rscv.fit(xtrain, ytrain)
            best_rf = rscv.best_estimator_
            print("Best RandomForest Params:", rscv.best_params_)
            print("Best RandomForest Score:", rscv.best_score_)
            print("=" * 80)

            # ===========================
            # Hyperparameter Tuning - KNN
            # ===========================
            print("Tuning KNN...")
            knn = KNeighborsClassifier()
            param_grid = {"n_neighbors": [3, 5, 7, 9, 11]}
            grid = GridSearchCV(knn, param_grid, cv=3, scoring="accuracy", n_jobs=-1)
            grid.fit(xtrain, ytrain)
            best_knn = grid.best_estimator_
            print("Best KNN Params:", grid.best_params_)
            print("Best KNN Score:", grid.best_score_)
            print("=" * 80)

            # ===========================
            # Final Ensemble Model
            # ===========================
            print("Training Final Voting Classifier...")
            final_model = VotingClassifier(
                estimators=[
                    ("rf", best_rf),
                    ("knn", best_knn),
                    ("gb", GradientBoostingClassifier())
                ],
                voting="hard"
            )
            final_model.fit(xtrain, ytrain)

            # ===========================
            # Final Evaluation
            # ===========================
            print_evaluated_results(
                xtrain, ytrain, xtest, ytest, final_model, problem_type="classification"
            )
            accuracy = accuracy_score(ytest, final_model.predict(xtest))
            print("\nFinal Clarity Model Accuracy:", accuracy)

            # ===========================
            # Save Model
            # ===========================
            os.makedirs(
                os.path.dirname(self.model_trainer_config.trained_model_file_path),
                exist_ok=True
            )
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=final_model
            )
            print("\n✅ Clarity Model saved successfully at artifacts/clarity_model.pkl")
            logging.info("Clarity Model Training Completed Successfully")

            return accuracy

        except Exception as e:
            logging.error("Error occurred in Clarity Model Training")
            raise CustomException(e, sys)