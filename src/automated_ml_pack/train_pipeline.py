import argparse
import pandas as pd
import numpy as np
import os, sys
from dataclasses import dataclass
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")

from automated_ml_pack.modules.logger import logging
from automated_ml_pack.modules.exception import CustomException

from automated_ml_pack.modules.components.data_transformation import DataTransformation
from automated_ml_pack.modules.components.data_ingestion import DataIngestion
from automated_ml_pack.modules.components.model_trainer import ModelTrainer
from automated_ml_pack.modules.utils import load_object, VisualizeResults

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='This tool facilitates the training of multiple machine learning models, optimizes the models, and saves the trained models. It also conducts model evaluation using diverse methods. Furthermore, the tool is capable of handling both regression and classification tasks. Additional options are described below.',
                                     usage='%(prog)s -[INPUT_FILE] [options]'
                                     )
    parser.add_argument('--input_file', required=True, type=str, help='Path to the input data in CSV/TSV format.')
    parser.add_argument('--input_type', type=str, choices=['csv', 'tsv'], default='csv', help='Type of input file format (csv or tsv)')
    parser.add_argument('--training_type', type=str, choices=['clf', 'reg'], default='clf', help='Type of training (e.g., "classification", "regression")')
    parser.add_argument('--target_column', type=str, required = True, help='Name of the target column in the input dataframe')
    parser.add_argument('--engineer_new_features', default=False, action="store_true",help='Flag to perform engineering of new features or not.')
    parser.add_argument('--output_base', type=str, default="", help='Base Name for most output files.')
    parser.add_argument('--test_size', type=float, default=0.2, help='What fraction of the dataset should be used for testing. Normally, cross validation is performed on the other percentage of the data to access the model\' generalization.')
    parser.add_argument('--no_standard_scaling', default = False, action="store_true", help='Whether or not to apply scikit-learn standard scaler on the data.')
    parser.add_argument('--feature_selection', default=False, action="store_true",help='Whether or not to perform feature selection on the dataset.')
    parser.add_argument('--feature_selection_method', type=str, default = "addition", choices = ["addition", "elimination"], help='Specify between recursive feature addition and recursive feature elimination algorithms for classification. By default, recursive feature addition is applied. For regression tasks, SelectKBest is used for feature selection.')
    parser.add_argument('--selectkbest_num_features', type=int, default = 32, help='Number of top features to select. For regression only.')
    parser.add_argument('--output_dir', type=str, default = "outputs", help='Custom Name of Output Folder.')
    parser.add_argument('--return_data', default = False, action="store_true", help='Select to include raw data, training data and test data in the output folders.')
    parser.add_argument('--no_param_finetune', default = False, action="store_true", help='If true, hyperparameter search will not be performed for each model. Otherwise, hyperparameter tunning is performed.')

    args = parser.parse_args()

    # Load dataframe from input file
    logging.info("Loading input data")
    try:
        if args.input_file:
            if args.input_type == 'csv':
                data = pd.read_csv(args.input_file)
            elif args.input_type == 'tsv':
                data = pd.read_csv(args.input_file, sep='\t')
            else:
                print("Invalid input type specified.")
                return
        else:
            raise ValueError("Please provide a valid input file path.")
            exit(1)
            
    except Exception as e:
        raise CustomException(e, sys)
    logging.info("Successfully Read the dataset as dataframe!")


    # Setting Input Configurations
    logging.info("Setting configurations based on provided parameters!")
    try:
        # Check if the target column exists in the dataframe
        if args.target_column not in data.columns:
            raise ValueError(f"Target column '{args.target_column}' not found in the input dataframe.")
            exit(1)

        # what kind of model to build
        if args.training_type =="clf":
            feature_selection_method = args.feature_selection_method
        elif args.training_type == "reg":
            feature_selection_method = "selectkbest"

        else:
            raise ValueError("--training_type argument must be either 'clf' or 'reg'!")
            exit(1)

        # create output folder
        os.makedirs(args.output_dir,exist_ok=True)

        engineer_new_features = args.engineer_new_features
        target_column = args.target_column
        training_type = args.training_type
        output_base = args.output_base
        test_size = args.test_size
        standard_scaling = not args.no_standard_scaling
        feature_selection = args.feature_selection
        selectkbest_num_features = args.selectkbest_num_features
        output_dir = args.output_dir
        return_data = args.return_data
        param_finetune = not args.no_param_finetune


    except Exception as e:
        raise CustomException(e, sys)
    logging.info("Completed setting configurations!")

    obj=DataIngestion(input_data = data, 
                      modeling_type = training_type,
                      test_size = test_size,
                      target_column_name = target_column,
                      output_base = output_base,
                      output_dir = output_dir
                      )
    train_data,test_data, numerical_columns, categorical_columns=obj.initiate_data_ingestion()
	
    logging.info("Transforming Data!")
    data_transformation=DataTransformation(numerical_columns=numerical_columns,
                                           categorical_columns = categorical_columns,
                                           target_column_name = target_column,
                                           modeling_type = training_type,
                                           engineer_new_features = engineer_new_features,
                                           standard_scaling = standard_scaling,
                                           output_base = output_base,
                                           feature_selection = feature_selection,
                                           feature_selection_method = feature_selection_method,
                                           selectkbest_num_features = selectkbest_num_features,
                                           output_dir = output_dir
                                           )
    train_arr,test_arr,_,selected_columns, reverse_encode=data_transformation.initiate_data_transformation(train_data,test_data)

    modeltrainer=ModelTrainer(modeling_type = training_type,
                              output_base = output_base,
                              output_dir = output_dir,
                              param_finetune = param_finetune
                              )

    best_model_name, __ = modeltrainer.initiate_model_trainer(train_arr,test_arr)
    print(__)

    if training_type == "clf":
        # plot results
        logging.info("Plotting Results")
        results_plotter = VisualizeResults(os.path.join(f"{output_dir}",f"{training_type}_{output_base}_report.json"), output_dir = output_dir)
        results_plotter.run_make_plots()

        # get classification report for best model.
        test_data_path = os.path.join(f"{output_dir}",f"{training_type}_{output_base}_test.csv")
        test_df=pd.read_csv(test_data_path)
        preprocessing_obj=load_object(os.path.join(f"{output_dir}",f"{training_type}_{output_base}_preprocessor.pkl"))

        input_feature_test_df=test_df.drop(columns=[target_column],axis=1)
        target_feature_test_df=test_df[target_column].values

        input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

        # load best model
        best_model_path = os.path.join(f"{output_dir}",f"{training_type}_{output_base}_{best_model_name}_model.pkl")
        best_model = load_object(best_model_path)
        y_pred = best_model.predict(input_feature_test_arr)
        y_pred = [reverse_encode[label] for label in y_pred]
        y_true = list(target_feature_test_df)
        target_names = [item for item in np.unique(y_pred)]

        clf_report = classification_report(y_true, y_pred)

        f = open(os.path.join(f"{output_dir}",f"{output_base}_classification_report.txt"), 'w')
        f.write(clf_report)
        f.close()

    # delete data from output folder
    if not return_data:
        os.remove(os.path.join(f"{output_dir}",f"{training_type}_{output_base}_data.csv"))
        os.remove(os.path.join(f"{output_dir}",f"{training_type}_{output_base}_train.csv"))
        os.remove(os.path.join(f"{output_dir}",f"{training_type}_{output_base}_test.csv"))

if __name__ == "__main__":
    main()
