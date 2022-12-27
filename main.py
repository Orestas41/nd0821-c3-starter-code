from fastapi import FastAPI
#from typing import Union
from pydantic import BaseModel

import os
import joblib
import pandas as pd
import numpy as np

app = FastAPI()


class TaggedItem(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: float
    capital_loss: float
    hours_per_week: float
    native_country: str

    class Config:
        schema_extra = {
            "example": {
                "age": 39,
                "workclass": "State-gov",
                "fnlgt": 77516,
                "education": "Bachelors",
                "education_num": 13,
                "marital_status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital_gain": 2174,
                "capital_loss": 0,
                "hours_per_week": 40,
                "native_country": "United-States",
            }
        }


@app.get('/')
async def say_hello():
    return {'greeting': 'Hello World!'}


@app.post("/predict")
async def model_inference(data: TaggedItem):

    dirname = os.path.dirname(__file__)
    encoder = joblib.load(os.path.join(dirname, "model/encoder.joblib"))
    model = joblib.load(os.path.join(dirname, "model/model.joblib"))

    sample = {}
    for d in data:
        sample[d[0].replace("_", "-")] = [d[1]]
    sample = pd.DataFrame.from_dict(sample)

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    X_categorical = sample[cat_features].values
    X_continuous = sample.drop(*[cat_features], axis=1)
    X_categorical = encoder.transform(X_categorical)
    X = np.concatenate([X_continuous, X_categorical], axis=1)

    # inference
    pred = model.predict(X)
    res = "<=50K" if pred[0] == 0 else ">50K"

    # turn prediction into JSON
    return {"prediction": res}
    