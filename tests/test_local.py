def test_local_api_greeting(client):
    response = client.get('/')
    assert response.status_code == 200
    assert response.json() == {'greeting': 'Hello World!'}


def test_local_api_predict_less(client):

    sample = {
        'age': 45,
        'workclass': 'State-gov',
        'fnlgt': 65241,
        'education': 'Bachelors',
        'education_num': 13,
        'marital_status': 'Married-civ-spouse',
        'occupation': 'Sales',
        'relationship': 'Wife',
        'race': 'White',
        'sex': 'Male',
        'capital_gain': 1949,
        'capital_loss': 0,
        'hours_per_week': 40,
        'native_country': 'United-States',
    }

    response = client.post('/predict', json=sample)

    assert response.status_code == 200
    assert response.json()['prediction'] == '<=50K'


def test_local_api_predict_more(client):

    sample = {
        'age': 28,
        'workclass': 'Self-emp-not-inc',
        'fnlgt': 44467,
        'education': 'Masters',
        'education_num': 13,
        'marital_status': 'Never-married',
        'occupation': 'Adm-clerical',
        'relationship': 'Not-in-family',
        'race': 'White',
        'sex': 'Male',
        'capital_gain': 0,
        'capital_loss': 0,
        'hours_per_week': 80,
        'native_country': 'United-States',
    }

    response = client.post('/predict', json=sample)

    assert response.status_code == 200
    assert response.json()['prediction'] == '>50K'
