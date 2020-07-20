import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
import numpy as np
# https://github.com/google-research/bert/issues/349


class BertEncoder(object):
    tokenizer = None
    model = None

    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = TFBertModel.from_pretrained('bert-base-uncased')

    def bert_encoder(self, text):
        input_ids = tf.constant(self.tokenizer.encode(text))[None, :]  # Batch size 1
        outputs = self.model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
        return np.average(last_hidden_states, axis=1)
