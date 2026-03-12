# data_ingestion.py
import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig
from src.components.data_transformation_clarity import DataTransformationClarity
from src.components.model_trainer_clarity import ModelTrainerClarity

# Initialize Data Ingestion Configuration
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts','train.csv')
    test_data_path: str = os.path.join('artifacts','test.csv')
    raw_data_path: str = os.path.join('artifacts','data.csv')

# Create a class for Data Ingestion
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initate_data_ingestion(self):
        logging.info('Data ingestion method Started')
        try:
            # Get absolute path to gemstone.csv relative to this script
            current_dir = os.path.dirname(os.path.abspath(__file__))
            data_path = os.path.join(current_dir, '..', '..', 'notebook', 'gemstone.csv')

            df = pd.read_csv(data_path)
            logging.info('Dataset read as pandas Dataframe')

            # Create artifacts folder if not exists
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            # Save raw dataset
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            
            logging.info('Train Test Split Initiated')
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info('Ingestion of Data is completed')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info('Exception occured at Data Ingestion stage')
            raise CustomException(e, sys)
    
if __name__ == '__main__':
    obj = DataIngestion()
    train_data, test_data = obj.initate_data_ingestion()

    data_transformation = DataTransformationClarity()  # <-- specify target
    train_arr, test_arr, preprocessor = data_transformation.initate_data_transformation(train_data, test_data)

    modeltrainer = ModelTrainerClarity(model_type='classification')  # <-- classification model
    model = modeltrainer.initate_model_training(train_arr, test_arr)

    # Save with clarity-specific names
    import joblib, os
    os.makedirs('artifacts', exist_ok=True)
    joblib.dump(model, 'artifacts/clarity_model.pkl')
    joblib.dump(preprocessor, 'artifacts/clarity_preprocessor.pkl')