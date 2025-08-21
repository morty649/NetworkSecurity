import sys
from networksecurity import(
    DataIngestion,
    NetworkSecurityException,
    DataIngestionConfig,
    TrainingPipelineConfig,
    DataValidationConfig,
    DataValidation,
    DataTransformation,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelTrainer
)
from networksecurity.logging.logger import logging

""" 
Extract,Transform and Load Pipeline - 
End To End MLOPS Projects With ETL Pipelines- Building Network Security System
SMOTETomek is used for conversion of imbalanced dataset to balanced

"""


if __name__=='__main__':
    try:
        trainingpipelineconfig = TrainingPipelineConfig()
        dataingestionconfig = DataIngestionConfig(trainingpipelineconfig)
        dataingestion = DataIngestion(dataingestionconfig)
        logging.info("Initiate the Data Ingestion")
        dataingestionartifact = dataingestion.initiate_data_ingestion()
        logging.info("Data Initiation Completed")
        #print(dataingestionartifact)
        data_validation_config = DataValidationConfig(trainingpipelineconfig)
        data_validation = DataValidation(dataingestionartifact,data_validation_config)
        logging.info("Initiate the Data validation")
        data_validation_artifact = data_validation.initiate_data_validation()
        logging.info("Data Validation Completed")
        #print(data_validation_artifact)
        data_transformation_config = DataTransformationConfig(trainingpipelineconfig)
        logging.info("Data Transformation Started")
        data_transformation = DataTransformation(data_validation_artifact,data_transformation_config)
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        #print(data_transformation_artifact)
        logging.info("Data Transformation Completed")

        logging.info("Model Training Started")
        model_trainer_config = ModelTrainerConfig(trainingpipelineconfig)
        model_trainer = ModelTrainer(data_transformation_artifact=data_transformation_artifact,model_trainer_config=model_trainer_config)
        model_trainer_artifact = model_trainer.initiate_model_trainer()

        logging.info("Model Training Artifact Created")
        
        
    except Exception as e:
        raise NetworkSecurityException(e,sys)
