from simpletransformers.seq2seq import (
    Seq2SeqModel,
    Seq2SeqArgs,
)
from utils.variable import *
model_args = Seq2SeqArgs()
model_args.src_lang = 'vi_VN'
model_args.tgt_lang = 'vi_VN'
model_args.eval_batch_size = 8
model_args.max_length = 1024
model_args.max_seq_length = 1024
abstractive_model = Seq2SeqModel(
      encoder_decoder_type="mbart",
      encoder_decoder_name=abstractive_model_path,
      args=model_args,
      use_cuda=True,
      max_length=1024
  )
