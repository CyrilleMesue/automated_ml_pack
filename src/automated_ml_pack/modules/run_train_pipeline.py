# def train pipeline

def run_train_pipeline(input_file:str = None,  
                       target_column:str = None,
                       input_type:str = "csv",
                       training_type:str = "clf",
                       engineer_new_features:bool = False, 
                       output_base:str = "", 
                       test_size:float = 0.2, 
                       no_standard_scaling:bool = False,
                       feature_selection:bool = False, 
                       feature_selection_method:str = "addition",
                       selectkbest_num_features: int = 32, 
                       output_dir:str = None,
                       return_data:bool = False,
                       no_param_finetune:bool = False,
                       finetune_fraction:float = 1.0,
                       verbose:bool = True
                      ):
    """
    Function to facilitate the training of multiple machine learning models,
    optimize the models, and save the trained models. It also conducts model
    evaluation using diverse methods. Additionally, the function is capable
    of handling both regression and classification tasks.

    Args:
        input_file (str): Path to the input data in CSV/TSV format.
        input_type (str): Type of input file format (csv or tsv).
        training_type (str): Type of training ("classification" or "regression").
        target_column (str): Name of the target column in the input dataframe.
        engineer_new_features (bool, optional): Flag to perform engineering of
            new features or not. Defaults to False.
        output_base (str, optional): Base Name for most output files. Defaults to None.
        test_size (float, optional): Fraction of the dataset to be used for testing.
            Defaults to 0.2.
        no_standard_scaling (bool, optional): Whether or not to apply scikit-learn
            standard scaler on the data. Defaults to False.
        feature_selection (bool, optional): Whether or not to perform feature
            selection on the dataset. Defaults to False.
        feature_selection_method (str, optional): Specify between recursive
            feature addition and recursive feature elimination algorithms for
            classification. Defaults to None.
        selectkbest_num_features (int, optional): Number of top features to select.
            For regression only. Defaults to None.
        output_dir (str, optional): Custom Name of Output Folder. Defaults to None.
        return_data (bool, optional): Select to include raw data, training data
            and test data in the output folders. Defaults to False.
    """
    
    import os
    
    if  feature_selection:
        feature_selection = " --feature_selection"
    else:
        feature_selection = ""

    if engineer_new_features:
        engineer_new_features = " --engineer_new_features"
    else:
        engineer_new_features = ""

    if no_standard_scaling:
        no_standard_scaling = " --no_standard_scaling"
    else:
        no_standard_scaling = ""

    if output_base == "":
        output_base = f" --output_base {training_type}"
    else:
        output_base = f" --output_base {output_base}"
    if no_param_finetune:
        no_param_finetune = " --no_param_finetune"
    else:
        no_param_finetune = ""
        
    # handle options
    script = f"run_train_pipeline --input_file {input_file} --target_column {target_column} --selectkbest_num_features {selectkbest_num_features} --training_type {training_type} --test_size {test_size} --feature_selection_method {feature_selection_method} --output_dir {output_dir} --finetune_fraction {finetune_fraction}" + feature_selection + engineer_new_features + no_standard_scaling + output_base + no_param_finetune

    if verbose:
        print(script)
    # Your code to execute the training pipeline goes here
    os.system(script)
