from datasets import Dataset
import numpy as np
import streamlit as st
import pandas as pd
from utils.variable import *
from scipy.special import softmax
from components.retriever import Retriever
from transformers import AutoTokenizer
class ConvertToMultiSpanQA():
  def get_new_df(self, original_df):
    data=[]
    df=original_df.reset_index()
    for i in range(len(original_df)):
      dct=dict()
      context=" ".join(original_df.at[i,'context'].strip().split())
      dct['words']=context
      data.append(dct)
    return data

  def find_word_index(self, char_index, word_indices):
      for i, index in enumerate(word_indices):
          if char_index >= index and (i == len(word_indices) - 1 or char_index < word_indices[i+1]):
              return i
      return None

  def preprocess_aspect(self, original_df ):
    processed_data = []
    data = self.get_new_df(original_df)
    for sample in data:
        text = sample['words']

        words = []
        word_indices = [] 

        current_word = ""
        for i, char in enumerate(text):
            if char.isspace():
                if current_word:
                    words.append(str(current_word))
                    word_indices.append(i - len(current_word))
                    current_word = ""
                if not char.isspace():
                    words.append(char)
                    word_indices.append(i)
            else:
                current_word += char

        if current_word:
            words.append(str(current_word))
            word_indices.append(len(text) - len(current_word))

        word_seq = words
        processed_data.append((word_seq))

    return processed_data



  def get_df_iob(self, original_df):
    processed_data = self.preprocess_aspect(original_df)
    processed_data_filtered = []
    for word_seq in processed_data:
      processed_data_filtered.append((word_seq))

    # Create empty lists for each column
    ids = []
    words = []

    # Iterate through the list of data and separate into corresponding columns
    for i, (sentence) in enumerate(processed_data_filtered):
        # Get the length of the sentence
        sentence_length = len(sentence)

        # Add ID for the sentence
        ids.extend([i] * sentence_length)

        # Add words and IO for the sentence
        words.extend(sentence)

    # Create DataFrame from the separated columns
    df = pd.DataFrame({'sentence_id': ids, 'words': words, })
    return df
  # # Use groupby to group words in column b based on the value of column a, then use .join() to join the words back into a sentence

  def get_data_convert(self, original_df):
    df = self.get_df_iob(original_df)
    df = df.groupby('sentence_id').agg({'words': list})
    output = dict()
    output['version'] = 1.0
    list_data = []
    for index, row in df.iterrows():
      dct = dict()
      dct['id'] = f"row_{index}"
      dct['question'] = (original_df.at[index, 'question'] + ". " + original_df.at[index,'document'] + " ").split()
      dct['context'] = df.at[index,'words']
      list_data.append(dct)
    output['data'] = list_data
    return output

# Ví dụ sử dụng



class LongContextHandler():
    def __init__(self):
        self.tokenizer = st.session_state.tokenizer
    def prepare_train_features(self, examples):
          tokenized_examples = self.tokenizer(
              examples['question'],
              examples['context'],
              truncation="only_second",
              max_length=512,
              stride=64,
              return_overflowing_tokens=True,
              return_offsets_mapping=True,
              padding='max_length',
              is_split_into_words=True,
          )
          # Since one example might give us several features if it has a long context, we need a map from a feature to
          # its corresponding example. This key gives us just that.
          sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
          # The offset mappings will give us a map from token to character position in the original context. This will
          # help us compute the start_positions and end_positions.          offset_mapping = tokenized_examples.pop("offset_mapping")

          tokenized_examples["example_id"] = []
          tokenized_examples["word_ids"] = []
          tokenized_examples["sequence_ids"] = []

          for i, sample_index in enumerate(sample_mapping):
              # Grab the sequence corresponding to that example (to know what is the context and what is the question).
              sequence_ids = tokenized_examples.sequence_ids(i)

              # Start token index of the current span in the text.
              token_start_index = 0
              while sequence_ids[token_start_index] != 1:
                  token_start_index += 1

              word_ids = tokenized_examples.word_ids(i)
              tokenized_examples["example_id"].append(examples["id"][sample_index])
              tokenized_examples["word_ids"].append(word_ids)
              tokenized_examples["sequence_ids"].append(sequence_ids)
          print("ĐÃ TOKENIZER XONG")
          return tokenized_examples
      
    def get_word_ids_context(self, word_ids):
    # Find the position of the first pair of None, which is correct for CafeBERT, but for ViBERT, mBERT, velectra, None is in the middle of a word
        index_of_first_none_pair = -1
        for i in range(len(word_ids) - 1):
            if word_ids[i] is None and word_ids[i + 1] is None:
                index_of_first_none_pair = i
                break
        word_question = list(set(word_ids[1:index_of_first_none_pair+1]))
        word_question = [x for x in word_question if x is not None]
        if index_of_first_none_pair != -1:
            word_context = list(set(word_ids[index_of_first_none_pair + 1:]))
        word_context = [x for x in word_context if x is not None]
        return word_context, word_question


    def  get_context_label_id(self, raw_datasets_type, raw_iob):
        list_final_context = []
        list_final_id = []
        list_final_true_id = []
        word_min_index = []
        word_max_index = []
        type_column_names = raw_datasets_type.column_names
        type_dataset = raw_datasets_type.map(
            self.prepare_train_features,
                batched=True,
                remove_columns=type_column_names,
                desc="Running tokenizer on train dataset",
            )
        count = 0
        for item in type_dataset:
            row_id = item['example_id']
            raw_context = raw_iob[int(row_id.split("_")[1])]['context']
            raw_question = raw_iob[int(row_id.split("_")[1])]['question']
            context_ids_word, question_ids_word = self.get_word_ids_context(item['word_ids'])
            context_split = raw_question + raw_context[min(context_ids_word):max(context_ids_word)+1]

            list_final_context+=context_split
            list_final_id+=[count]*len(context_split)
            list_final_true_id += [row_id]*len(context_split)
            word_min_index += [min(context_ids_word)]*len(context_split)
            word_max_index += [max(context_ids_word)]*len(context_split)
            count += 1
        dataframe = pd.DataFrame({"true_sentence_id": list_final_true_id, "sentence_id": list_final_id , "words": list_final_context,  "word_min_index":word_min_index,'word_max_index': word_max_index})
        return dataframe

class ExtractiveAnswerPostProcessor():
    def get_prediction_sentence(self, sentence_prob, sentence_word, question_length):
        # count=0
        prob_sentence=[]
        word_sentence=[]
        list_index = []
        for index in range(question_length, len(sentence_word)):
            word = list(sentence_word[index].keys())[0]
            label = list(sentence_word[index].values())[0]
            list_prob_sentence = softmax(list(sentence_prob[index].values())[0][0])
            if label == 'B':
                prob_sentence.append(list_prob_sentence[0])
                word_sentence.append(word)
                list_index.append(index - question_length)
            elif label == 'I':
                prob_sentence.append(list_prob_sentence[1])
                word_sentence.append(word)
                list_index.append(index - question_length)
        return prob_sentence, word_sentence, list_index

    def get_extractive_answers_prob(self, raw_test, test, raw_output, prediction_label):
        list_final_answer = []
        list_final_prob = []
        list_id_true = test['true_sentence_id'].values
        list_min_word = test['word_min_index'].values
        list_max_word = test['word_max_index'].values
        merged_dict_word = {}
        merged_dict_prob = {}
        # flag = 0
        print("Length of raw_output:   ", len(raw_output))
        for index in range(len(raw_output)):
            true_id = list_id_true[index]
            question_length = len(raw_test[int(true_id.split("_")[1])]['question'])
            min_id = list_min_word[index]
            max_id = list_max_word[index]
            sentence_prob = raw_output[index]
            sentence_word = prediction_label[index]
            prob_sentence, word_sentence, list_index = ExtractiveAnswerPostProcessor().get_prediction_sentence(sentence_prob, sentence_word, question_length)
            if true_id not in merged_dict_word:
                merged_dict_word[true_id] = list_index
                merged_dict_prob[true_id] = prob_sentence
            else:
                # If the value has already appeared, merge the subarray into the existing array
                new_list_index = []
                new_prob = []
                count = 0
                for x in list_index:
                    new_index = x + min_id
                    new_list_index.append(new_index)
                    if new_index not in merged_dict_word[true_id]:
                        new_prob.append(prob_sentence[count])
                    else:
                        current_prob = prob_sentence[count]
                        duplicate_index = merged_dict_word[true_id].index(new_index)
                        duplicate_prob = merged_dict_prob[true_id][duplicate_index]
                        if current_prob > duplicate_prob:
                            max_prob = current_prob
                        else:
                            max_prob = duplicate_prob
                        merged_dict_prob[true_id][duplicate_index] = max_prob
                count += 1

                previous_prob = merged_dict_prob[true_id]
                merged_dict_prob[true_id].extend(new_prob)
                previous_word_index = merged_dict_word[true_id]
                merged_dict_word[true_id] = list(set(previous_word_index + new_list_index))
        
        print("Length of raw_test:   ", len(raw_test))
        for count in range(len(raw_test)):
            true_context = raw_test[count]['context']
            row_number = f'row_{count}'
            answer = ''
            assert len(list(set(merged_dict_word[row_number]))) == len(merged_dict_prob[row_number]), f'Number of words and number of probs are different {len(list(set(merged_dict_word[row_number])))} and {len(merged_dict_prob[row_number])}'
            for word_id in sorted(list(set(merged_dict_word[row_number]))):
              answer += " " + true_context[word_id]
            # print("draft answer:     ", answer)
            list_final_answer.append(answer)
            prob = np.mean(merged_dict_prob[row_number])
            list_final_prob.append(prob)
        return list_final_answer, list_final_prob, merged_dict_word, merged_dict_prob
    
    def get_multiple_answer(self, question):
      st.spinner(text="Searching for information..... ")
      extractive_model = st.session_state.extractive_model
      test_model = Retriever().get_context_for_MRC(question)
      print("Length of test_model:   ", test_model.shape)
      test_converted = ConvertToMultiSpanQA().get_data_convert(test_model)
      print("test_converted:    \n", test_converted)
      raw_test_retrieval_iob = test_converted['data']
      print("raw_test_retrieval_iob:   \n", raw_test_retrieval_iob)
      datase_test_retrieval_iob = Dataset.from_list(test_converted['data'])
      print("Converted retrieval document to iob ")
      datase_test_retrieval_iob_converted = LongContextHandler().get_context_label_id(datase_test_retrieval_iob, raw_test_retrieval_iob)
      datase_test_retrieval_iob_converted['words'] = datase_test_retrieval_iob_converted['words'].astype(str)
      # Using groupby to group words in column b based on the value of column a, then using .join() to concatenate the words back into a sentence
      datase_test_retrieval_iob_converted['d'] = datase_test_retrieval_iob_converted.groupby('sentence_id')['words'].transform(' '.join)
      datase_test_retrieval_iob_converted.drop_duplicates(subset='sentence_id',inplace=True)
      print("Preparing prediction ")
      print(datase_test_retrieval_iob_converted['d'].values)
      print(extractive_model)
      prediction, raw_output = extractive_model.predict(datase_test_retrieval_iob_converted['d'].values)
      print("Length of prediction:    ", len(prediction))
      list_final_answer, list_final_prob, merged_dict_word, merged_dict_prob = ExtractiveAnswerPostProcessor().get_extractive_answers_prob(raw_test_retrieval_iob, datase_test_retrieval_iob_converted, raw_output, prediction )
      print("List of extractive answers:   \n", list_final_answer)
      print("\n multiple answers dataframe: \n", test_model)
      test_model['answer_predict'] = list_final_answer
      test_model['answer_prob'] = list_final_prob
      return test_model
