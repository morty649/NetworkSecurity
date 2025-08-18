"""
What is the output you would expect from data_ingestion -> train file path and test file path
"""

from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    trained_file_path:str
    test_file_path:str