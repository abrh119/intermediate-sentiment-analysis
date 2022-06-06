# intermediate-sentiment-analysis
BERT based sentiment analysis model trained on a movies review dataset, that is capable of detecting most of the reviews based on books to movies to classes

Dataset Link: https://www.kaggle.com/competitions/sentiment-analysis-on-movie-reviews/data
This model can be trained and saved using the notebook, and once saved it can be used a RESTFUL api using fastapi

# To Run
Clone the files and open up a folder named "sentiment_model" and save the model files and run "pip install -r requirements" and then finally run "# python -m uvicorn main:app --reload" in the terminal
