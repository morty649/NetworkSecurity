import sys
from networksecurity import(
    DataIngestion,
    NetworkSystemException,
    DataIngestionConfig,
    TrainingPipelineConfig
)
from networksecurity.logging.logger import logging



if __name__=='__main__':
    try:
        trainingpipelineconfig = TrainingPipelineConfig()
        dataingestionconfig = DataIngestionConfig(trainingpipelineconfig)
        dataingestion = DataIngestion(dataingestionconfig)
        logging.info("Initiate the Data Ingestion")
        dataingestionartifact = dataingestion.initiate_data_ingestion()
        print(dataingestionartifact)
        
    except Exception as e:
        raise NetworkSystemException(e,sys)
