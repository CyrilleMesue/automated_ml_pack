
def main:
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

if __name__ == "__main__":
    main()