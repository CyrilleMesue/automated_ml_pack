import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
import os
from sklearn import set_config
set_config(transform_output = "pandas")

from feature_engine.creation import RelativeFeatures
from feature_engine.creation import MathFeatures
from feature_engine.selection import RecursiveFeatureElimination
from feature_engine.selection import RecursiveFeatureAddition
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler, LabelEncoder

from automated_ml_pack.modules.exception import CustomException
from automated_ml_pack.modules.logger import logging

from automated_ml_pack.modules.utils import save_object, save_json

@dataclass
class DataTransformationConfig:
    basename_tag = "*"
    preprocessor_obj_file_path=os.path.join('outputs',f"{basename_tag}preprocessor.pkl")
    selected_columns_path = os.path.join('outputs',f"{basename_tag}selected_columns.json")

class DataTransformation:
    def __init__(self, 
                 modeling_type:str = "reg",
                 categorical_columns:list = [],
                 numerical_columns:list = [],
                 target_column_name:str = "",
                 output_base:str = "",
                 engineer_new_features:bool = False,
                 standard_scaling:bool = True,
                 feature_selection:bool = False,
                 feature_selection_method:str = "addition",
                 selectkbest_num_features:int = 32,
                 output_dir:str="outputs"
                 ):
        """
        modeling_type can be either "clf" or "reg" for classification and regression repsectively.
        """
        self.data_transformation_config=DataTransformationConfig()
        basename_tag = self.data_transformation_config.basename_tag
        self.modeling_type = modeling_type
        self.output_base = output_base
        self.data_transformation_config.preprocessor_obj_file_path = self.data_transformation_config.preprocessor_obj_file_path.replace(basename_tag,f"{modeling_type}_{output_base}_")
        self.data_transformation_config.selected_columns_path = self.data_transformation_config.selected_columns_path.replace(basename_tag,f"{modeling_type}_{output_base}_")
        self.data_transformation_config.preprocessor_obj_file_path = self.data_transformation_config.preprocessor_obj_file_path.replace("outputs",f"{output_dir}")
        self.data_transformation_config.selected_columns_path = self.data_transformation_config.selected_columns_path.replace("outputs",f"{output_dir}")
        self.engineer_new_features = engineer_new_features
        self.standard_scaling = standard_scaling
        self.feature_selection = feature_selection
        self.feature_selection_method = feature_selection_method
        self.selectkbest_num_features = selectkbest_num_features
        self.target_column_name = target_column_name
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns

        if target_column_name in self.categorical_columns:
            self.categorical_columns.remove(target_column_name)
        else:
            self.numerical_columns.remove(target_column_name)
            
            
    def get_data_transformer_object(self):
        '''
        This function si responsible for data trnasformation
        
        '''
        try:

            # set feature creation configurations
            if self.engineer_new_features:
                logging.info("Performing Feature Engineering!")
                mf_transformer = MathFeatures(variables = self.numerical_columns, func = ["sum", "mean", "median", "max", "min","std"])
                rf_num_transformer = RelativeFeatures(variables=self.numerical_columns,
                                                    reference=self.numerical_columns,
                                                    func = ["mul"])
            else:
                mf_transformer = None
                rf_num_transformer = None

            # set standard scaling configs
            if self.standard_scaling:
                logging.info("Performing Standard Scaling!")
                num_standard_scaler = StandardScaler()
                cat_standard_scaler = StandardScaler(with_mean=False)
            else:
                num_standard_scaler = None
                cat_standard_scaler = None

            # set configurations for feature selection and model training type
            feature_selection_methods = {
                "addition":RecursiveFeatureAddition(RandomForestClassifier(random_state=32), cv=3),
                "elimination": RecursiveFeatureElimination(RandomForestClassifier(random_state=32), cv=3),
                "selectkbest": SelectKBest(score_func=f_regression, k=self.selectkbest_num_features)
            }

            if self.feature_selection:
                logging.info("Performing Feature Selection!")
                feature_selector = feature_selection_methods[self.feature_selection_method]
            else:
                feature_selector = None

            

            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("math_features", mf_transformer),
                ("relative_features", rf_num_transformer),
                ("scaler",num_standard_scaler),
                ]
            )

            cat_pipeline=Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder(sparse_output=False)),
                ("scaler",cat_standard_scaler)
                ]

            )

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,self.numerical_columns),
                ("cat_pipelines",cat_pipeline,self.categorical_columns)

                ]
            )

            preprocessor_pipeline = Pipeline(
                steps = [
                    ("preprocessor", preprocessor),
                    ("feature_selection", None)
                ]
            )

            return preprocessor_pipeline
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            input_feature_train_df=train_df.drop(columns=[self.target_column_name],axis=1)
            target_feature_train_df=train_df[self.target_column_name]

            input_feature_test_df=test_df.drop(columns=[self.target_column_name],axis=1)
            target_feature_test_df=test_df[self.target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            selected_columns = list(input_feature_train_arr.columns)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            
            if self.modeling_type == "clf":
                le = LabelEncoder()
                target_feature_train = le.fit_transform(target_feature_train_df.values)
                target_feature_test = le.transform(target_feature_test_df.values)
                reverse_encode = {le.transform([value])[0]:value for value in target_feature_test_df.unique()}
            else:
                target_feature_train = target_feature_train_df.values
                target_feature_test = target_feature_test_df.values
                reverse_encode = None

            train_arr = np.c_[
                input_feature_train_arr, target_feature_train
            ]
            test_arr = np.c_[input_feature_test_arr, target_feature_test]

            logging.info(f"Saved preprocessing object.")

            preprocessing_obj_file_path = self.data_transformation_config.preprocessor_obj_file_path
            save_object(

                file_path=preprocessing_obj_file_path,
                obj=preprocessing_obj

            )
	    
    	    # save selected column
            save_json(
                self.data_transformation_config.selected_columns_path,
                selected_columns
                )
	    
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
                selected_columns,
                reverse_encode
            )
        except Exception as e:
            raise CustomException(e,sys)
