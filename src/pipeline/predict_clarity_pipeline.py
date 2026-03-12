
# src/pipeline/predict_clarity_pipeline.py
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

# ===========================
# CustomClarityData Class
# ===========================
class CustomClarityData:
    """
    Class to gather input data for clarity prediction and convert to DataFrame.
    """
    def __init__(self, carat, depth, table, x, y, z, cut, color):
        self.carat = carat
        self.depth = depth
        self.table = table
        self.x = x
        self.y = y
        self.z = z
        self.cut = cut
        self.color = color

    def get_data_as_dataframe(self):
        """
        Convert input features into a pandas DataFrame.
        """
        try:
            data = {
                "carat": [self.carat],
                "depth": [self.depth],
                "table": [self.table],
                "x": [self.x],
                "y": [self.y],
                "z": [self.z],
                "cut": [self.cut],
                "color": [self.color]
            }
            df = pd.DataFrame(data)
            logging.info("Clarity input data converted to DataFrame")
            return df
        except Exception as e:
            logging.error("Exception occurred while converting clarity input to DataFrame")
            raise CustomException(e, sys)


# ===========================
# PredictClarityPipeline Class
# ===========================
class PredictClarityPipeline:
    """
    Clarity prediction pipeline.
    """
    def __init__(self):
        pass

    def predict(self, features: pd.DataFrame):
        """
        Predict clarity using the pre-trained model.

        Args:
            features: pandas DataFrame of input features

        Returns:
            np.array of predicted clarity labels
        """
        try:
            preprocessor_path = "artifacts/clarity_preprocessor.pkl"
            model_path = "artifacts/clarity_model.pkl"

            preprocessor = load_object(file_path=preprocessor_path)
            model = load_object(file_path=model_path)

            # Transform features using preprocessor
            data_scaled = preprocessor.transform(features)

            # Predict clarity
            pred = model.predict(data_scaled)

            return pred

        except Exception as e:
            logging.error("Exception occurred in clarity prediction pipeline")
            raise CustomException(e, sys)