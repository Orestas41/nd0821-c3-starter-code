from starter.ml.model import train_model, compute_model_metrics, inference
from sklearn.ensemble import RandomForestClassifier


def test_train_model(processedData):

    X_train = processedData['X_train']
    y_train = processedData['y_train']
    trained_model = train_model(X_train, y_train)

    assert isinstance(trained_model, RandomForestClassifier)


def test_compute_metrics(processedData, trainedModel):

    X_test = processedData['X_test']
    y_test = processedData['y_test']
    predicion = inference(trainedModel, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, predicion)

    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= fbeta <= 1


def test_inference(processedData, trainedModel):

    X_test = processedData['X_test']
    predicion = inference(trainedModel, X_test)

    assert len(X_test) == len(predicion)
