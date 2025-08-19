import sys
from networksecurity import(
    DataIngestion,
    NetworkSecurityException,
    DataIngestionConfig,
    TrainingPipelineConfig,
    DataValidationConfig,
    DataValidation
)
from networksecurity.logging.logger import logging

""" 
Extract,Transform and Load Pipeline - 
End To End MLOPS Projects With ETL Pipelines- Building Network Security System

"""


if __name__=='__main__':
    try:
        trainingpipelineconfig = TrainingPipelineConfig()
        dataingestionconfig = DataIngestionConfig(trainingpipelineconfig)
        dataingestion = DataIngestion(dataingestionconfig)
        logging.info("Initiate the Data Ingestion")
        dataingestionartifact = dataingestion.initiate_data_ingestion()
        logging.info("Data Initiation Completed")
        print(dataingestionartifact)
        data_validation_config = DataValidationConfig(trainingpipelineconfig)
        data_validation = DataValidation(dataingestionartifact,data_validation_config)
        logging.info("Initiate the Data validation")
        data_validation_artifact = data_validation.initiate_data_validation()
        logging.info("Data Validation Completed")
        print(data_validation_artifact)
        
    except Exception as e:
        raise NetworkSecurityException(e,sys)
