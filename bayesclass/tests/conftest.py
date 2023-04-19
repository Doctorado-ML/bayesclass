import pytest
from sklearn.datasets import load_iris
from fimdlp.mdlp import FImdlp


@pytest.fixture
def iris():
    dataset = load_iris()
    X = dataset["data"]
    y = dataset["target"]
    features = dataset["feature_names"]
    # To make iris dataset has the same values as our iris.arff dataset
    patch = {(34, 3): (0.2, 0.1), (37, 1): (3.6, 3.1), (37, 2): (1.4, 1.5)}
    for key, value in patch.items():
        X[key] = value[1]
    return X, y, features


@pytest.fixture
def data(iris):
    return iris[0], iris[1]


@pytest.fixture
def features(iris):
    return iris[2]


@pytest.fixture
def class_name():
    return "class"


@pytest.fixture
def data_disc(data):
    clf = FImdlp()
    X, y = data
    return clf.fit_transform(X, y), y
