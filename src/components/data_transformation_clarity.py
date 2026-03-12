import os
import sys
import logging
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

from src.exception import CustomException
from src.utils import save_object


class DataTransformationClarityConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "clarity_preprocessor.pkl")
    target_encoder_path = os.path.join('artifacts', "clarity_target_encoder.pkl")


class DataTransformationClarity:
    def __init__(self):
        self.data_transformation_config = DataTransformationClarityConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = ["carat", "depth", "table", "x", "y", "z"]
            categorical_columns = ["cut", "color"]

            num_pipeline = Pipeline(
                steps=[("scaler", StandardScaler())]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("encoder", OrdinalEncoder()),
                    ("scaler", StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):

        try:
            # ==========================
            # Read Data
            # ==========================
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and Test data loaded")

            # ==========================
            # Preprocessing Object
            # ==========================
            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "clarity"
            drop_columns = ["id", target_column_name]

            # ==========================
            # Split Features and Target
            # ==========================
            input_feature_train_df = train_df.drop(columns=drop_columns)
            input_feature_test_df = test_df.drop(columns=drop_columns)

            target_feature_train_df = train_df[target_column_name]
            target_feature_test_df = test_df[target_column_name]

            # ==========================
            # Encode Target Variable
            # ==========================
            label_encoder = LabelEncoder()

            target_feature_train_arr = label_encoder.fit_transform(target_feature_train_df)
            target_feature_test_arr = label_encoder.transform(target_feature_test_df)

            # Save encoder
            save_object(
                file_path=self.data_transformation_config.target_encoder_path,
                obj=label_encoder
            )

            # ==========================
            # Apply Feature Transformation
            # ==========================
            logging.info("Applying preprocessing object")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # ==========================
            # Combine Features + Target
            # ==========================
            train_arr = np.c_[input_feature_train_arr, target_feature_train_arr]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_arr]

            # ==========================
            # Save Preprocessor
            # ==========================
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info("Preprocessor and encoder saved")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.info("Exception occurred in initiate_data_transformation")
            raise CustomException(e, sys)