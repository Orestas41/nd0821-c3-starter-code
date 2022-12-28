from fastapi import FastAPI
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
                "age": 25,
                "workclass": "State-gov",
                "fnlgt": 68472,
                "education": "Masters",
                "education_num": 13,
                "marital_status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Female",
                "capital_gain": 1786,
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

    pred = model.predict(X)
    res = "<=50K" if pred[0] == 0 else ">50K"

    return {"prediction": res}

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")
