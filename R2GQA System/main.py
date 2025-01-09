import os
import gradio as gr
import streamlit as st
from utils.variable import *
from components.generator import get_final_abstractive_answer
from transformers import AutoTokenizer

current_path = os.getcwd()
print("Current path:", current_path)

if "extractive_model" not in st.session_state:
    print("lưu extractive model vào state")
    from model.reader import extractive_model
    st.session_state['extractive_model'] = extractive_model


if "extractive_model" not in st.session_state:
    print("lưu extractive model vào state")
    from model.reader import extractive_model
    st.session_state['extractive_model'] = extractive_model

if "abstractive_model" not in st.session_state:
    print("lưu abstractive model vào state")
    from model.generator import abstractive_model
    st.session_state['abstractive_model'] = abstractive_model

if "tokenizer" not in st.session_state:
    print("lưu tokenizer model vào state")
    st.session_state['tokenizer'] = AutoTokenizer.from_pretrained(extractive_model_path)

def get_final_answer(question, x):
    final_answer = get_final_abstractive_answer(question)
    return final_answer
chatbot = gr.Chatbot(
              [],
              elem_id="chatbot",
              bubble_full_width=False,
              show_label = True,
              show_copy_button = True,
              every = "float",
              height = 500
          )
button = gr.Button(value = 'Gửi')#, variant ="primary", min_width = 20)
textbox = gr.Textbox(placeholder = "Nhập câu hỏi của bạn")
gr.ChatInterface(chatbot = chatbot, fn = get_final_answer, title = "Hệ thống hỏi đáp tự động R2GQA", submit_btn = button).launch(debug = True, share = True, inbrowser  = True)