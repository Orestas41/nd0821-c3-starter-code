import os
from ml.data import process_data
from ml.model import inference, compute_model_metrics


def categorical_data_slice(test, cat_features, trained_model, encoder, lb, col):
    
    unique = test[col].unique()

    for val in unique:
        id = test[col] == val
        temp = test[id]

        X_test, y_test, encoder, lb = process_data(
            temp,
            categorical_features=cat_features,
            label="salary",
            training=False,
            encoder=encoder,
            lb=lb,
        )

        preds = inference(trained_model, X_test)
        precision, recall, fbeta = compute_model_metrics(y_test, preds)

        dirname = os.path.dirname(__file__)
        with open(os.path.join(dirname, "../screenshots/slice_output.txt"), "w") as f:
            f.write(f"{col}\n")
            for value in unique_val:
                f.write(f"\t {value.strip()}\n")
                f.write(f"\t\t precision:{precision} recall:{recall} fbeta:{fbeta}\n")