from langchain_core.agents import AgentExecutor
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
import pandas as pd
import os

def load_csv(file_path):
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    
    return pd.read_csv(file_path)

DATASET_PATH = os.path.join("../Data","preprocessed_data.csv")

def load_dataset():
    return load_csv(DATASET_PATH)   



if __name__ == "__main__":

    ds = load_csv(DATASET_PATH)
    ds.head()