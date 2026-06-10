from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool

import pandas as pd
import os

from langchain_community.document_loaders import CSVLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate

from langchain_core.globals import set_debug
set_debug(True)

from dotenv import load_dotenv
load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

def load_csv(file_path):
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    
    return pd.read_csv(file_path)

DATASET_PATH = os.path.join("../Data","preprocessed_data.csv")
csv_file = CSVLoader(
        file_path=DATASET_PATH
        )
    
doc = csv_file.load()
    
embed = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 250,
    chunk_overlap = 50
)
    
chunks = text_splitter.split_documents(
        documents=doc
        )
    
retriever = BM25Retriever.from_documents(
        documents=chunks,
        k = 5
        )

@tool
def product_search(product_name: str) -> str:
    """
    Search the product catalog for items matching the query.
    Use this for any request involving finding or recommending products.
    """
    return f"Macbook pro m1: {product_name}"

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0)

# prompt_1 = ChatPromptTemplate.from_messages(
#     [
#         ("system", "You are a helpful shopping assistant. "),
#         ("human", "{input}")
#     ]
# )



if __name__ == "__main__":

    ds = load_csv(DATASET_PATH)
    ds.head()

    agent = create_agent(
        tools=[product_search],
        model=model)
    
    ans = agent.invoke(
    {"messages":[{"role":"user","content":"What are the best laptops available in the market?"}]}
    )

    print(ans["messages"][-1].content)

    

