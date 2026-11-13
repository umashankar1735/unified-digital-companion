import os

class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY") or "your_default_secret_key"
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or "your_openai_api_key"
    MODEL_PATH = os.environ.get("MODEL_PATH") or "/path/to/your/model"
    DATASET_PATH = os.environ.get("DATASET_PATH") or "/path/to/your/dataset.csv"
    DEBUG = os.environ.get("DEBUG", "False") == "True"