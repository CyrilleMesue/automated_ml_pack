import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.ensemble import (
    AdaBoostRegressor, AdaBoostClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    RandomForestRegressor, RandomForestClassifier
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from xgboost import XGBRegressor, XGBClassifier 
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from automated_ml_pack.modules.exception import CustomException
from automated_ml_pack.modules.logger import logging

from automated_ml_pack.modules.utils import save_object,getparams,evaluate_models, load_json, save_json

@dataclass
class ModelTrainerConfig:
    basename_tag = "*"
    trained_model_file_path=os.path.join("outputs",f"{basename_tag}model.pkl")
    model_report_path = os.path.join("outputs",f"{basename_tag}report.json")

class ModelTrainer:
    def __init__(self,
                 modeling_type = "reg",
                 output_base:str = "",
                 output_dir:str="outputs"
                 ):
        self.model_trainer_config=ModelTrainerConfig()
        self.basename_tag = self.model_trainer_config.basename_tag
        self.output_base = output_base
        self.model_trainer_config.model_report_path = self.model_trainer_config.model_report_path.replace(self.basename_tag,f"{modeling_type}_{output_base}_")
        self.model_trainer_config.model_report_path = self.model_trainer_config.model_report_path.replace("outputs",f"{output_dir}")
        self.modeling_type = modeling_type
        self.output_dir = output_dir


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            if self.modeling_type == "reg":
                models = {
                    "Random Forest": RandomForestRegressor(random_state = 32),
                    "Decision Tree": DecisionTreeRegressor(random_state = 32),
                    "Gradient Boosting": GradientBoostingRegressor(random_state = 32),
                    "Linear Regression": LinearRegression(),
                    "XGBRegressor": XGBRegressor(random_state = 32),
                    "CatBoosting Regressor": CatBoostRegressor(random_state = 32,verbose=False),
                    "AdaBoost Regressor": AdaBoostRegressor(random_state = 32),
                }
            else:
                models = {
                    "Random Forest": RandomForestClassifier(random_state = 32),
                    "Decision Tree": DecisionTreeClassifier(random_state = 32),
                    "Gradient Boosting": GradientBoostingClassifier(random_state = 32),
                    "Logistic Regression": LogisticRegression(random_state = 32),
                    "XGBClassifier": XGBClassifier(random_state = 32),
                    "CatBoosting Classifier": CatBoostClassifier(random_state = 32,verbose=False),
                    "AdaBoost Classifier": AdaBoostClassifier(random_state = 32),
                    "MLPClassifier": MLPClassifier(random_state = 32, verbose=False),
                    "SVC": SVC(random_state = 32)
                }
            load_params = getparams()
            params = load_params["model_params"][self.modeling_type]

            logging.info("Training and Evaluating Models!")
            
            model_report:dict=evaluate_models(X_train=X_train,
                                              y_train=y_train,
                                              X_test=X_test,
                                              y_test=y_test,
                                              models=models,
                                              param=params,
                                              modeling_type = self.modeling_type
                                              )
            
            ## To get best model score from dict
            best_model_score = 0
            best_model_name = ""

            for model_name in model_report:
                if self.modeling_type == "reg":
                    score = model_report[model_name]["cross_val_report"]["r2"]["mean"]
                else:
                    score = model_report[model_name]["cross_val_report"]["accuracy"]["mean"]

                if score >= best_model_score:
                    best_model_score = score
                    best_model_name = model_name
            
            best_model = models[best_model_name]

            if best_model_score<0.5:
                raise "No best model found"
            logging.info(f"Best found model on both training and testing dataset")

            # save best model
            model_trainer_obj_file_path = self.model_trainer_config.trained_model_file_path.replace(self.basename_tag,f"{self.modeling_type}_{self.output_base}_{best_model_name}_")
            model_trainer_obj_file_path = model_trainer_obj_file_path.replace("outputs",f"{self.output_dir}")

            save_object(
                file_path=model_trainer_obj_file_path,
                obj=best_model
            )

            # save reports
            model_report_file_path = self.model_trainer_config.model_report_path
            save_json(model_report_file_path, model_report)

            return best_model_name, f"Best model: {best_model_name}\nBest model cross validation score: {best_model_score}"
            
            
        except Exception as e:
            raise CustomException(e,sys)
