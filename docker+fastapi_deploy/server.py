import joblib
import uvicorn
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer

# App creation and model loading
app = FastAPI()
model = joblib.load("./model.joblib")

# Reading file for tfidf transformation
vectorizer = TfidfVectorizer()
vectorizer.fit_transform(pd.read_pickle('tfidf.pkl'))


class Textclassifier(BaseModel):
    """
    Input features validation for the ML model
    """
    text: str

@model.post('/predict')
def predict(text: Textclassifier):
    """
    :param text: input data from the post request
    :return: predicted text label
    """
    texts = [[
        text
    ]]

    label = model.predict(vectorizer.transform(texts)).tolist()[0]
    return {
        "label": label
    } 

if __name__ == '__main__':
    # Run server using given host and port
    uvicorn.run(model, host='127.0.0.1', port=19)
