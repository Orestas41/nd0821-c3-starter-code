import os
import pytest
import pandas as pd
from sklearn.model_selection import train_test_split
from starter.ml.data import process_data
from starter.ml.model import train_model
from fastapi.testclient import TestClient
from main import app

@pytest.fixture()
def data():
    """10 sample data for test"""

    dirname = os.path.dirname(__file__)
    df = pd.read_csv(os.path.join(dirname, "data/clean_census.csv"))[:10]
    return df


@pytest.fixture()
def processedData(data):
    """processed data for train/test"""

    train, test = train_test_split(data, test_size=0.20)
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
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    X_test, y_test, encoder, lb = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )

    res = {
        "train": train,
        "test": test,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "encoder": encoder,
        "lb": lb,
    }
    return res


@pytest.fixture()
def trainedModel(processedData):
    """simple model for testing"""

    X_train = processedData["X_train"]
    y_train = processedData["y_train"]
    trained_model = train_model(X_train, y_train)
    return trained_model

@pytest.fixture()
def client():
    client = TestClient(app)
    return 