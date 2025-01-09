from utils.utils import ExtractiveAnswerPostProcessor
from utils.constant import *

def get_final_extractive_answer(question):
  print("original question:   ", question)
  multiple_answers_dataframe = ExtractiveAnswerPostProcessor().get_multiple_answer(question = question)
  
  # Calculate the 'sum' value for each answer
  multiple_answers_dataframe['sum'] = (
      mrc_weight * multiple_answers_dataframe['score'] + 
      (1 - mrc_weight) * multiple_answers_dataframe['answer_prob']
  )

  # Get the answer with the highest 'sum' value for each index
  new_multiple_answers_dataframe = multiple_answers_dataframe.loc[
      multiple_answers_dataframe.groupby('index')['sum'].idxmax()
  ]
  
  # Get the final answer and document information
  final_answer = new_multiple_answers_dataframe['answer_predict'].values[0]
  article = new_multiple_answers_dataframe['context'].values[0].split("\n")[0]
  document_name = new_multiple_answers_dataframe['document'].values[0]
  
  return final_answer, document_name, article