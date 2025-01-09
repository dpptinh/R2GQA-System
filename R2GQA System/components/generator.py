from components.machine_reader import get_final_extractive_answer
import pandas as pd
import streamlit as st
from utils.variable import *

def get_final_abstractive_answer(question):
  abstractive_model = st.session_state.abstractive_model
  # Get the extractive answer and document information
  answer_extractive, document_name, article = get_final_extractive_answer(question)
  # print("Got here already")
  # Read document links from Excel file
  document_links = pd.read_excel(documents_link_path)
  filtered_df = document_links[document_links['document'] == document_name]
  
  # Get the document link
  link = filtered_df['link'].tolist()[0]
  
  # Predict the abstractive answer
  abstractive_answer = abstractive_model.predict([f"{question} </s> {answer_extractive} </s>"])
  
  # Create the final answer
  final_answer = f"According to {article} of document {document_name}: \n {abstractive_answer[0]} \n\n Reference link: {link}"
  
  return final_answer
