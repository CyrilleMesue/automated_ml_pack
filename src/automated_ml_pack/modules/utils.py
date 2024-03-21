import os
import sys

import numpy as np 
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import dill
import pickle
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, root_mean_squared_error
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_score, recall_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.metrics import classification_report

from automated_ml_pack.modules.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train,X_test,y_test,models,param, modeling_type = "reg", param_finetune = True):
    try:
        report = {}

        if modeling_type =="reg":
            kfold = KFold(n_splits=5, random_state=32, shuffle = True)
            scoring = "r2"
        else:
            kfold = StratifiedKFold(n_splits=5, random_state=32, shuffle = True)
            scoring = "accuracy"
	
        for i in tqdm(range(len(list(models)))):
            model = list(models.values())[i]
            
            if param_finetune:
                para=param[list(models.keys())[i]]

                gs = GridSearchCV(model,para,cv=kfold, scoring = scoring)
                gs.fit(X_train,y_train)
                model_best_params = gs.best_params_
                model.set_params(**model_best_params)

            model.fit(X_train,y_train)

            #get cross validation and test report
            cross_val_report = get_cross_validation_scores(model, X_train,y_train, cv= kfold, training_type = modeling_type)
            
            # make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_report = get_test_report(y_train, y_train_pred, training_type=modeling_type)

            test_report = get_test_report(y_test, y_test_pred, training_type=modeling_type)

            report[list(models.keys())[i]] = {"test_report":test_report, 
                                              "train_report":train_report, 
                                              "cross_val_report":cross_val_report}

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def get_cross_validation_scores(model, X, y, cv, training_type = "clf"):
    """
    Get cross validation scores:
        ('f1', 'precision', 'recall', 'roc_auc', "accuracy") for classification
        ('r2', 'neg_mean_squared_error', 'neg_mean_absolute_error', 'neg_root_mean_squared_error') for regression
    """

    try:
        if training_type == "clf":
            scoring = ('f1', 'precision', 'recall', 'roc_auc', "accuracy")
        elif training_type == "reg":
            scoring = ('r2', 'neg_mean_squared_error', 'neg_mean_absolute_error', 'neg_root_mean_squared_error')
        else:
            print("Provide correct value for train_type: ['clf','reg']")

        scores = cross_validate(model, X, y, cv=cv,scoring=scoring,return_train_score=False)
        score_report = {"_".join(score_name.split("_")[1:]):{"mean":scores[score_name].mean(), 
                                                         "std":scores[score_name].std(),
                                                         "all":list(scores[score_name])} for score_name in scores}
    except Exception as e:
        raise CustomException(e, sys)
    
    return score_report

def get_test_report(true, predicted, training_type='clf'):

    """
    Run Various Evaluation Metrics on data
    """

    try:
        if training_type == 'clf':
            score_report = {"f1":f1_score(true, predicted),
                            "accuracy": accuracy_score(true, predicted),
                            "roc_auc": roc_auc_score(true, predicted),
                            "precision":precision_score(true, predicted),
                            "recall":recall_score(true, predicted)
                        }
            return score_report

        elif training_type == 'reg':
            score_report = {
                            "r2": r2_score(true, predicted),
                            "neg_mean_squared_error":-mean_squared_error(true, predicted),
                            "neg_mean_absolute_error":-mean_absolute_error(true, predicted),
                            "neg_root_mean_squared_error": -mean_squared_error(true, predicted, squared=False)
            }
            return score_report
        else:
            raise ValueError("Invalid training type. Choose 'clf' or 'reg'.")
    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

def load_json(json_path):
    """
    Load json data as a dictionary
    """

    try:
        with open(json_path) as f:
            file = f.read()
        json_data = json.loads(file)

    except Exception as e:
        raise CustomException(e, sys)

    return json_data


def save_json(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        json_object = json.dumps(obj, indent=2)

        with open(file_path, "w") as file_obj:
            file_obj.write(json_object)

    except Exception as e:
        raise CustomException(e, sys)



class VisualizeResults:
    """
    This class is used to visualized the results obtained from the machine learning pack. 
    """

    def __init__(self, results_path:str = None,
                 output_dir:str = "outputs"
                ):
        self.output_dir = output_dir
        # load results data as a dictionary
        self.results_data = load_json(results_path)
    
    def prepare_error_bar_data(self, report_type:str = "cross_val_report"):
        """
        convert results to dataframe format for errorbar plotting
        """

        try:
            reshaped_results_data = {"score":[], "model":[], "metric":[]}
            for model_name in self.results_data:
                for metric in self.results_data[model_name][report_type]:
                    if report_type == "cross_val_report":
                        score = self.results_data[model_name][report_type][metric]["mean"]
                    elif report_type == "train_report":
                        score = self.results_data[model_name][report_type][metric]
                    elif report_type == "test_report":
                        score = self.results_data[model_name][report_type][metric]
                    else:
                        raise ValueError("Please enter correct value for report_type as one of ['cross_val_report', 'train_report', 'test_report']")
                    reshaped_results_data["score"].append(score)
                    reshaped_results_data["model"].append(model_name)
                    reshaped_results_data["metric"].append(metric)
            
            reshaped_results_data = pd.DataFrame(reshaped_results_data)
            reshaped_results_data = reshaped_results_data[reshaped_results_data.metric !="time"]
            return reshaped_results_data
        except Exception as e:
            raise CustomException(e, sys)

    
    def prepare_barplot_data(self, report_type:str = "cross_val_report"):
        """
        convert results to dataframe format for barplot plotting
        """
        try:
            reshaped_results_data = {name:[] for name in self.results_data}
            reshaped_results_data["metric"] = []
            report_type = report_type
            metrics = self.results_data["Random Forest"][report_type].keys()
            
            for metric in metrics:
                reshaped_results_data["metric"].append(metric)
                for model_name in self.results_data:
                    if report_type == "cross_val_report":
                        score = self.results_data[model_name][report_type][metric]["mean"]
                    elif report_type == "train_report":
                        score = self.results_data[model_name][report_type][metric]
                    elif report_type == "test_report":
                        score = self.results_data[model_name][report_type][metric]
                    else:
                        raise ValueError("Please enter correct value for report_type as one of ['cross_val_report', 'train_report', 'test_report']")

                    reshaped_results_data[model_name].append(score)
                    
            reshaped_results_data = pd.DataFrame(reshaped_results_data)
            reshaped_results_data = reshaped_results_data[reshaped_results_data.metric != "time"]
            reshaped_results_data.rename(columns={'Random Forest': 'RandomForest',
                               'Decision Tree': 'Tree',
                               'Gradient Boosting': 'GradientBoost',
                               'Logistic Regression': 'Logistic Regression',
                               'XGBClassifier': 'XGBoost',
                               'CatBoosting Classifier': 'CatBoost',
                               'AdaBoost Classifier': 'AdaBoost',
                               'MLPClassifier': 'NN',
                               "Linear Regression": "Linear Regression",
                               "XGBRegressor": 'XGBoost',
                               "CatBoosting Regressor": 'CatBoost',
                               "AdaBoost Regressor": 'AdaBoost',
                               }, inplace=True)

        except Exception as e:
            raise CustomException(e, sys)
        return reshaped_results_data

    
    def plot_errorbars(self,
                       report_type:str = "cross_val_report",
                       score_column:str= "score", 
                       hue_colum:str = "model",
                       xlabel_column:str = "metric", 
                       xlabel:str = "Evaluation Metric", 
                       ylabel:str="Score"):
        """
        receive score data and plot error bar
        """

        try:
            data = self.prepare_error_bar_data(report_type = report_type)
            # set colors
            colors = ['green', 'lime', 'magenta','orange', 'blue', 'cyan', 'red', 'black' , 'gray','brown', 'purple', 'yellow','salmon','pimk']
            
            nrows = 1
            ncols = data[xlabel_column].nunique()
            figure, axes  = plt.subplots(nrows=1 , ncols = ncols , figsize = (20, 10)) 
        
            labels = list(data[xlabel_column].unique())
            hue = list(data[hue_colum].unique())
            n_hue = data[hue_colum].nunique()
        
            ## set the labels of the X-Axis
            for i , label in enumerate(labels):
                axes[i,].set(xlabel=label)
        
            for i in range(1, ncols):
              #set horizontal axis
              #axes[i,].set_ylim((0.5, 1.00))
              axes[i,].yaxis.grid(True)
              axes[i].set_yticklabels([])    
        
            #set horizontal axis of first column plot in the figure
            axes[0,].yaxis.grid()
            
            #set the y-axis label for the first plot
            axes[0,].set_ylabel(f'{ylabel}', fontsize='large', fontweight='bold')
        
            ##set the font-size of the x-axis and y-axis labels
            for i in range(ncols):
              axes[i,].xaxis.label.set_size(24)
              if i == 0:
                axes[0,].yaxis.label.set_size(30)
                  
            x = range(1, n_hue+1)
        
            scores = {label:{"scores":[], "error_rate":[]} for label in labels}
            # create score list
            for label in labels:
                for type in hue:
                    score = data[data[xlabel_column] == label][data[hue_colum]==type].values[0,0]
                    error = 1-score
                    scores[label]["scores"].append(score)
                    scores[label]["error_rate"].append(error)
        
            # plot graphs
            for i, label in enumerate(labels):
                axes[i,].scatter(x,scores[label]["scores"] , s= 100, color = colors[:n_hue])
                #plot the error bars
                for pos, y, err, color in zip(x, scores[label]["scores"],scores[label]["error_rate"] , colors[:n_hue]):
                    axes[i,].errorbar(pos, y, err, lw = 4,capsize = 4, capthick = 2 , color = color)  
                    
            i = 0
            for pos, y, err,color in zip(x, scores[label]["scores"],scores[label]["error_rate"],colors):
                 axes[ncols-1,].errorbar(pos, y, err, lw = 4, label=hue[i], capsize = 4, capthick = 2, color = color)
                 i = i + 1
            
            ##add common x-axis label to the Figure
            figure.text(0.5, 0.04, f'{xlabel}', ha='center', fontsize = 30, weight = 'bold')
        
            # change the fontsize of x and y axis ticks
            axes[0,].tick_params(axis='y', labelsize=24)
            
            for i in range(len(labels)):
              axes[i,].tick_params(axis = 'x', labelsize = 24)
              #remove x axis ticks and labels
              axes[i, ].tick_params(right = False, labelbottom = False, bottom = False)
            plt.legend(ncol=ncols ,loc="lower center", bbox_to_anchor=(-1.9, 1.02), fontsize = 17 , handlelength=0.5, handleheight=0.5) 
            plt.savefig(f"{self.output_dir}/{report_type}_error.png", bbox_inches='tight')
            plt.savefig(f"{self.output_dir}/{report_type}_error.pdf", bbox_inches='tight', format="pdf")

        except Exception as e:
            raise CustomException(e, sys)


    def barplots(self, 
                 report_type:str = "cross_val_report", 
                 x_column:str = "metric",
                 xlabel:str = "Evaluation Metric", 
                 ylabel:str="Score"
                ):
        """
        plot bar plots from score data.
        """
        try:
            result_data = self.prepare_barplot_data(report_type = "cross_val_report")
            colors = ['green', 'lime', 'magenta','orange', 'blue', 'cyan', 'red', 'black' , 'gray','brown', 'purple', 'yellow','salmon','pimk']
        
            n_cols = len(result_data.columns)-1
            # plot grouped bar chart
            result_data.plot(x=x_column,
                             kind='bar',
                             width=0.7,
                             stacked=False,
                             color = colors[:n_cols],
                             figsize=(20, 10))
            
            plt.xlabel(f'{xlabel}', fontsize = 30 , weight = 'bold');
            plt.ylabel(f'{ylabel}', fontsize=30, weight = 'bold');
            #plt.ylim((0.50, 1.00))
            plt.xticks(fontsize= 24 , rotation = 360)
            plt.yticks(fontsize= 24)
            axes = plt.gca()
            axes.yaxis.grid()
            plt.legend(ncol=n_cols, loc="lower center", bbox_to_anchor=(0.5, 1.02), fontsize = 17,handlelength=0.6, handleheight=0.6)
            plt.savefig(f"{self.output_dir}/{report_type}_bar.png", bbox_inches='tight')
            plt.savefig(f"{self.output_dir}/{report_type}_barr.pdf", bbox_inches='tight', format="pdf")
        except Exception as e:
            raise CustomException(e, sys)

    
    def run_make_plots(self):
        """
        make visualizations of ml results and save as pdf and png.
        """
    
        options = {"cross_val_report":"Cross Validation Score", "train_report":"Training Score", "test_report":"Test Score"}
        
        try:
            for option in options:
                    self.plot_errorbars(report_type = option, ylabel = options[option])
                    self.barplots(report_type = option, ylabel = options[option])
        except Exception as e:
            raise CustomException(e, sys)


def getparams():
    params = {"model_params" : 
    {"reg" : {"Decision Tree": 
            {"criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"]},
            "Random Forest": 
            {"n_estimators": [8, 16, 32, 64, 128, 256, 512]},
            "Gradient Boosting": 
            {"n_estimators": [8, 16, 32, 64, 128, 256, 512]},
            "Linear Regression": {},
            "XGBRegressor": {
                "learning_rate": [0.1, 0.01, 0.05, 0.001],
                "n_estimators": [8, 16, 32, 64, 128, 256, 512]
                },
            "CatBoosting Regressor": 
                {
                    "depth": [6, 8, 10],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "iterations": [30, 50, 100]
                },
            "AdaBoost Regressor": 
                {
                    "learning_rate": [0.1, 0.01, 0.5, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256, 512]
                }
            },
        "clf":
            {"Decision Tree": 
                {
                    "criterion": ["gini", "entropy"]
                },
            "Random Forest": 
                {
                    "n_estimators": [8, 16, 32, 64, 128, 256, 512]
                },
            "Gradient Boosting": 
                {
                    "n_estimators": [8, 16, 32, 64, 128, 256, 512]
                },
            "Logistic Regression": {},
            "XGBClassifier": 
                {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256, 512],
                    "max_depth": [3,5,8]
                },
            "CatBoosting Classifier": 
                {
                    "depth": [6, 8, 10],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "iterations": [30, 50, 100]
                },
            "AdaBoost Classifier": 
                {
                    "learning_rate": [0.1, 0.01, 0.5, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256, 512]
                },
            "MLPClassifier":{},
            "SVC":
                {
                    "C": [0.1,1,10,100]
                }
            }

        }
    }
    return params