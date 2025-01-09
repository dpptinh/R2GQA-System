import os
from dotenv import load_dotenv

load_dotenv()

full_data_path = os.getenv("FULL_DATA_PATH")
documents_link_path = os.getenv("DOCUMENTS_LINK_PATH")
embedding_model_path = os.getenv("EMBEDDING_MODEL_PATH")
extractive_model_path = os.getenv("EXTRACTIVE_MODEL_PATH")
abstractive_model_path = os.getenv("ABSTRACTIVE_MODEL_PATH")
top_n_retrieval_documents = 30