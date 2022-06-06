from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import run
from uvicorn.config import LOGGING_CONFIG
import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import tensorflow as tf
from transformers import BertTokenizer
from tensorflow.python.keras.models import load_model
from pydantic import BaseModel
# from fastapi.encoders import jsonable_encoder
# from typing import List, Dict

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


app = FastAPI()
model_path = "./sentiment_model/"
new_model = load_model(model_path)




origins = ["*"]
methods = ["*"]
headers = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=methods,
    allow_headers=headers
)


def make_prediction(model, processed_data, classes=[1, 2, 3, 4, 5]): # inistal classes are strings
    probs = model.predict(processed_data)[0]
    # print(probs)
    # print(classes[np.argmax(probs)])
    return classes[np.argmax(probs)]

# def make_prediction(model, processed_data, classes=['Lowest Rated', 'Not Reccommended', 'Neutral', 'Reccommended', 'Top Rated']):
#     probs = model.predict(processed_data)[0]
#     print(probs)
#     return classes[np.argmax(probs)]

class UserInput(BaseModel):
    comment: str


class Response(BaseModel):
    result: int
# initial value str

@app.post("/predict/", response_model=Response)
async def root(comment: UserInput):
    text = comment.comment
    processed_data = prepare_data(text, tokenizer)
    result = make_prediction(new_model, processed_data=processed_data)
    res = Response(result=result)
    return res


@app.get("/")
async def root():
    return {"message": "Server working!"}


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    run(app, host="0.0.0.0", port=port)


# to run
# python -m uvicorn main:app --reload
