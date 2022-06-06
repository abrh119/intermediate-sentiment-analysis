import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import tensorflow as tf
from transformers import BertTokenizer
from tensorflow.python.keras.models import Model, load_model

model_path = "./sentiment_model/"

new_model = load_model(model_path)

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')


def prepare_data(input_text, tokenizer):
    token = tokenizer.encode_plus(
        input_text,
        max_length=256,
        truncation=True,
        padding='max_length',
        add_special_tokens=True,
        return_tensors='tf'
    )
    return {
        'input_ids': tf.cast(token.input_ids, tf.float64),
        'attention_mask': tf.cast(token.attention_mask, tf.float64)
    }


def make_prediction(model, processed_data, classes=['Lowest Rated', 'Not Reccommended', 'Neutral', 'Reccommended', 'Top Rated']):
    probs = model.predict(processed_data)[0]
    print(probs)
    return classes[np.argmax(probs)]


input_text = "the movie is garbage, dont ever watch it "
processed_data = prepare_data(input_text, tokenizer)
result = make_prediction(new_model, processed_data=processed_data)
print(f"Predicted Sentiment: {result}")
