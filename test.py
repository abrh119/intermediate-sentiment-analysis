from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import run
from uvicorn.config import LOGGING_CONFIG
import os
import transformers
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import tensorflow as tf
from transformers import BertTokenizer
from tensorflow.python.keras.models import Model, load_model
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder
from typing import List, Dict


tokenizer = BertTokenizer.from_pretrained('bert-base-cased')


def prepare_data(input_text, tokenizer):
   # input_text = list(input_text)
    token = tokenizer.encode_plus(
        list(input_text),
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


def makePrediction(text):
     tokenizedValues = prepare_data(text, tokenizer)
     probs = new_model.predict(tokenizedValues)[0]
     print(probs)
     labels = ['Lowest Rated', 'Not Reccommended',
               'Neutral', 'Reccommended', 'Top Rated']
     newLabels = labels[np.argmax(probs)]
     print(newLabels)
     return newLabels
# def make_prediction(model, processed_data, classes=['Lowest Rated', 'Not Reccommended', 'Neutral', 'Reccommended', 'Top Rated']):
#     probs = model.predict(processed_data)[0]
#     print(probs)
#     return classes[np.argmax(probs)]
  


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


class UserInput(BaseModel):
    comment: str


class Response(BaseModel):
    result: List[str] = None


@app.post("/predict/", response_model=Response)
async def root(comment: UserInput):
    text = [comment.comment]
    processed_data = prepare_data(text, tokenizer)
    results = makePrediction(text)
    res = Response(result=results)
    return res


@app.get("/")
async def root():
    return {"message": "Server working!"}


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    run(app, host="0.0.0.0", port=port)


# to run
# python -m uvicorn main:app --reload
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import run
from uvicorn.config import LOGGING_CONFIG
import os
import transformers
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import tensorflow as tf
from transformers import BertTokenizer
from tensorflow.python.keras.models import Model, load_model
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder
from typing import List, Dict


tokenizer = BertTokenizer.from_pretrained('bert-base-cased')


def prepare_data(input_text, tokenizer):
   # input_text = list(input_text)
    token = tokenizer.encode_plus(
        list(input_text),
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


def makePrediction(text):
     tokenizedValues = prepare_data(text, tokenizer)
     probs = new_model.predict(tokenizedValues)[0]
     print(probs)
     labels = ['Lowest Rated', 'Not Reccommended',
               'Neutral', 'Reccommended', 'Top Rated']
     newLabels = labels[np.argmax(probs)]
     print(newLabels)
     return newLabels
# def make_prediction(model, processed_data, classes=['Lowest Rated', 'Not Reccommended', 'Neutral', 'Reccommended', 'Top Rated']):
#     probs = model.predict(processed_data)[0]
#     print(probs)
#     return classes[np.argmax(probs)]
  


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


class UserInput(BaseModel):
    comment: str


class Response(BaseModel):
    result: List[str] = None


@app.post("/predict/", response_model=Response)
async def root(comment: UserInput):
    text = [comment.comment]
    processed_data = prepare_data(text, tokenizer)
    results = makePrediction(text)
    res = Response(result=results)
    return res


@app.get("/")
async def root():
    return {"message": "Server working!"}


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    run(app, host="0.0.0.0", port=port)


# to run
# python -m uvicorn main:app --reload
