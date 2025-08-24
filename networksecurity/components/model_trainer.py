import os
import sys

from ..utilities import evaluate_models
from ..exception import NetworkSecurityException
from ..logging.logger import logging
from ..entity import DataTransformationArtifact,ModelTrainerArtifact,ModelTrainerConfig


from ..utilities import (
    NetworkModel,
    save_object,
    load_object,
    load_numpy_array_data,
    get_classification_score )

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier
)
import mlflow


class ModelTrainer:
    def __init__(self,data_transformation_artifact:DataTransformationArtifact,
                 model_trainer_config:ModelTrainerConfig):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def track_mlflow(self,best_model,classificationmetric):
        with mlflow.start_run():
            f1_score = classificationmetric.f1_score
            recall_score = classificationmetric.recall_score
            precision_score = classificationmetric.precision_score

            mlflow.log_metric("f1_score",f1_score)
            mlflow.log_metric("precision_score",precision_score)
            mlflow.log_metric("recall_score",recall_score)
            mlflow.sklearn.log_model(best_model,"model")

        
    def train_model_hyperparameter_tuning_and_evaluate_and_artifact_creation(self,x_train,y_train,x_test,y_test):
        models = {
            "Random Forest" : RandomForestClassifier(verbose=1),
            "Decision Tree":DecisionTreeClassifier(),
            "Logistic Regression":LogisticRegression(verbose=1),
            "Gradient Boosting": GradientBoostingClassifier(verbose=1),
            "Adaboost":AdaBoostClassifier()

        }

        params = {
            "Decision Tree":{
                'criterion':['gini','entropy','log_loss'],
                #'splitter':['best','random'],
                #'max_features':['sqrt','log2']
            },
            "Random Forest":{
                #'criterion':['gini','entropy','log_loss'],
                #'max_features':['sqrt','log2'],
                'n_estimators':[8,16,32,64,128,256]
            },
            "Gradient Boosting":{
                'learning_rate':[.1,.01,.05,.001],
                'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                'n_estimators':[8,16,32,64,128,256]
                #loss,criterion,max_features
            },
            "Logistic Regression":{},
            "Adaboost":{
                'learning_rate':[.1,.01,.05,.001],
                'n_estimators':[8,16,32,64,128,256]
            }
        }

        model_report:dict = evaluate_models(x_train,y_train,x_test,y_test,models,params)

        #To get best model score from dict
        best_model_score = max(sorted(model_report.values()))

        # To get best model name from dict
        best_model_name = list(model_report.keys())[
            list(model_report.values()).index(best_model_score)
        ]
        print(f">>> Best model selected: {best_model_name}")
        logging.info(f"Best model selected: {best_model_name}")

        best_model = models[best_model_name]
        y_train_pred = best_model.predict(x_train)

        classification_train_metric = get_classification_score(y_train,y_train_pred)

        #Track the mlflow - life cycle of a machine learning project

        self.track_mlflow(best_model,classification_train_metric)

        y_test_pred = best_model.predict(x_test)
        classification_test_metric = get_classification_score(y_test,y_test_pred)

        self.track_mlflow(best_model,classification_test_metric)


        preprocessor = load_object(file_path = self.data_transformation_artifact.transformed_object_file_path)
        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path,exist_ok=True)

        Network_model = NetworkModel(preprocessor,best_model)
        save_object(self.model_trainer_config.trained_model_file_path,obj=Network_model)

        save_object("final_model/model.pkl",best_model)

        #Model Trainer Artifact
        model_trainer_artifact = ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                                                      train_metric_artifact=classification_train_metric,
                                                      test_metric_artifact=classification_test_metric)
        
        logging.info(f"Model Trainer Artifact : {model_trainer_artifact}")
        return model_trainer_artifact

        
    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            #load those train and test files
            train_df = load_numpy_array_data(train_file_path)
            test_df = load_numpy_array_data(test_file_path)

            x_train,y_train,x_test,y_test = (
                train_df[:,:-1],
                train_df[:,-1],
                test_df[:,:-1],
                test_df[:,-1]
            )

            model_trainer_artifact_here = self.train_model_hyperparameter_tuning_and_evaluate_and_artifact_creation(x_train,y_train,x_test,y_test)
            return model_trainer_artifact_here

        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
