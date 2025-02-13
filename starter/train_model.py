# Script to train machine learning model.

from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics
from data_slice import data_slicing_cat

# Add the necessary imports for the starter code.
import pandas as pd
import joblib
import os
# Add code to load in the data.
dirname = os.path.dirname(__file__)
df = pd.read_csv('data/clean_census.csv')
# Optional enhancement,
# use K-fold cross validation instead of a train-test split.
train, test = train_test_split(df, test_size=0.20)

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

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label='salary', training=False,
    encoder=encoder, lb=lb)


# Train and save a model.
trained_model = train_model(X_train, y_train)

# Determine the classification metrics
preds = inference(trained_model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)
print(f'Precision:{precision}, Recall:{recall}, Fbeta:{fbeta}')

joblib.dump(trained_model, os.path.join(dirname, "../model/model.joblib"))
joblib.dump(encoder, os.path.join(dirname, "../model/encoder.joblib"))
data_slicing_cat(
    test, cat_features, trained_model, encoder, lb, "education")
