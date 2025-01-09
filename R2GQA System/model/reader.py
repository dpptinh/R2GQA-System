from simpletransformers.ner import NERModel, NERArgs
from utils.variable import *
import streamlit as st
# Define labels
label_list = ["B", "I", "O"]
label2id = {l: i for i, l in enumerate(label_list)}
id2label = {i: l for i, l in enumerate(label_list)}
num_labels = len(label_list)

# Model configuration
model_args = NERArgs()
model_args.labels_list = label_list  # Use the defined label_list
model_args.label2id = label2id  # Use the defined label2id
model_args.train_batch_size = 2
model_args.evaluate_during_training = True
model_args.evaluate_during_training_steps = 10000  # Reduce the number of steps for evaluation
model_args.do_lower_case = True
model_args.early_stopping_metric = "eval_loss"
model_args.early_stopping_patience = 5
model_args.eval_batch_size = 8
model_args.learning_rate = 5e-5
model_args.manual_seed = 42
model_args.max_seq_length = 512
model_args.num_train_epochs = 5
model_args.optimizer = "AdamW"
model_args.overwrite_output_dir = True
model_args.save_best_model = False
model_args.save_eval_checkpoints = True
model_args.save_optimizer_and_scheduler = True
model_args.save_steps = 500000
model_args.use_early_stopping = True
model_args.max_query_length = 512
model_args.max_answer_length = 512
model_args.doc_stride = 64
extractive_model = NERModel(
      "auto", extractive_model_path, args=model_args
  )

