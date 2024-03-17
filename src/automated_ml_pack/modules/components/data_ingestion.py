import os
import sys
from automated_ml_pack.modules.exception import CustomException
from automated_ml_pack.modules.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    basename_tag = "*"
    train_data_path: str=os.path.join('outputs',f"{basename_tag}train.csv")
    test_data_path: str=os.path.join('outputs',f"{basename_tag}test.csv")
    raw_data_path: str=os.path.join('outputs',f"{basename_tag}data.csv")

class DataIngestion:
    def __init__(self, 
                 input_data = None,
                 test_size:float = 0.2,
                 modeling_type:str = "reg",
                 target_column_name:str ="",
                 output_base:str ="",
                 output_dir:str = "outputs"
                 ):
        
        self.ingestion_config=DataIngestionConfig()
        basename_tag = self.ingestion_config.basename_tag
        self.ingestion_config.train_data_path = self.ingestion_config.train_data_path.replace(basename_tag,f"{modeling_type}_{output_base}_")
        self.ingestion_config.test_data_path = self.ingestion_config.test_data_path.replace(basename_tag,f"{modeling_type}_{output_base}_")
        self.ingestion_config.raw_data_path = self.ingestion_config.raw_data_path.replace(basename_tag,f"{modeling_type}_{output_base}_")

        self.ingestion_config.train_data_path = self.ingestion_config.train_data_path.replace("outputs",f"{output_dir}")
        self.ingestion_config.test_data_path = self.ingestion_config.test_data_path.replace("outputs",f"{output_dir}")
        self.ingestion_config.raw_data_path = self.ingestion_config.raw_data_path.replace("outputs",f"{output_dir}")

        self.input_data = input_data
        self.test_size = test_size
        self.target_column_name = target_column_name
        self.modeling_type = modeling_type
        self.output_base = output_base
        

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df=self.input_data
            logging.info('Successfully ingested dataset!')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            df = df[df[self.target_column_name].notna()].reset_index(drop=True) # remove entries where the target value is missing
            
            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df,test_size=self.test_size,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            numerical_features = [feature for feature in df.columns if df[feature].dtype != 'O']
            categorical_features = [feature for feature in df.columns if df[feature].dtype == 'O']
            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                numerical_features,
                categorical_features

            )
        except Exception as e:
            raise CustomException(e,sys)
        
