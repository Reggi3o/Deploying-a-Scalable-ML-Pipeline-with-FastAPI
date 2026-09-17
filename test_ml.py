import pytest
import pandas as pd
from ml.model import compute_model_metrics,train_model,performance_on_categorical_slice
from ml.data import process_data

@pytest.fixture
def sample_data():
    # create sample data for testing
        X_train = [[0,1], [1,0], [1,1], [0,0]]
        y_train = [0,1,1,0]

        return X_train, y_train

@pytest.fixture
def sample_data_with_categorical_slice():
    # create sample data for testing
        data = pd.DataFrame({
              'age': [25, 30, 35, 40],
              'workclass': ['Private', 'Self-emp', 'Private', 'Gov'],
              'fnlgt': [123456, 234567, 345678, 456789],
              'education': ['Bachelors', 'Masters', 'PhD', 'HS-grad'],
              'education-num': [13, 14, 16, 9],
              'marital-status': ['Never-married', 'Married', 'Divorced', 'Widowed'],
              'occupation': ['Tech-support', 'Exec-managerial', 'Sales', 'Other-service'],
              'relationship': ['Not-in-family', 'Husband', 'Wife', 'Own-child'],
              'race': ['White', 'Black', 'Asian-Pac-Islander', 'white'],
              'sex': ['Male', 'Female', 'Male', 'Female'],
              'capital-gain': [0, 5000, 10000, 0],
              'capital-loss': [0, 0, 0, 0],
              'hours-per-week': [40, 50, 60, 20],
              'native-country': ['United-States', 'Canada', 'Mexico', 'United-States'],
              'salary': ['<=50K', '>50K', '>50K', '<=50K']
              })

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

        return data, cat_features

# TODO: implement the first test. Change the function name and input as needed
def test_train_model(sample_data):
    """
    Test the train_model function
    """
    X_train, y_train = sample_data
    model = train_model(X_train, y_train)

    assert model is not None

# TODO: implement the second test. Change the function name and input as needed
def test_model_metrics(sample_data):
    """
    Test the metrics of the trained model
    """
    X_train, y_train = sample_data
    model = train_model(X_train, y_train)
    preds = model.predict(X_train)

    precision, recall, fbeta = compute_model_metrics(y_train, preds)

    assert precision >= 0 and precision <= 1
    assert recall >= 0 and recall <= 1
    assert fbeta >= 0 and fbeta <= 1


# TODO: implement the third test. Change the function name and input as needed
def test_performance_on_categorical_slice(sample_data_with_categorical_slice):
    """
    Test the performance of the model on a categorical slice of the data
    """
    data, cat_features = sample_data_with_categorical_slice
    X_train, y_train, encoder, lb = process_data(
        data, categorical_features=cat_features, label='salary', training=True
    )

    column_name = 'workclass'
    slice_value = 'Private'
    categorical_features = cat_features
    label = 'salary'
    encoder = encoder
    label_binarizer = lb
    model = train_model(X_train, y_train)

    precision, recall, fbeta = performance_on_categorical_slice(
        data, column_name, slice_value, categorical_features, label, encoder, label_binarizer, model
    )

    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)