import pytest
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import KBinsDiscretizer
from matplotlib.testing.decorators import image_comparison
from matplotlib.testing.conftest import mpl_test_settings


from bayesclass import TAN


@pytest.fixture
def data():
    X, y = load_iris(return_X_y=True)
    enc = KBinsDiscretizer(encode="ordinal")
    return enc.fit_transform(X), y


def test_TAN_constructor():
    clf = TAN()
    # Test default values of hyperparameters
    assert not clf.simple_init
    assert not clf.show_progress
    assert clf.random_state is None
    clf = TAN(simple_init=True, show_progress=True, random_state=17)
    assert clf.simple_init
    assert clf.show_progress
    assert clf.random_state == 17


def test_TAN_random_head(data):
    clf = TAN(random_state=17)
    clf.fit(*data, head="random")
    assert clf.head_ == 3


def test_TAN_classifier(data):
    clf = TAN()

    clf.fit(*data)
    attribs = ["classes_", "X_", "y_", "head_", "features_", "class_name_"]
    for attr in attribs:
        assert hasattr(clf, attr)

    X = data[0]
    y = data[1]
    y_pred = clf.predict(X)
    y = y.reshape(-1, 1)
    assert y_pred.shape == (X.shape[0], 1)
    assert sum(y == y_pred) == 147


@image_comparison(
    baseline_images=["line_dashes"], remove_text=True, extensions=["png"]
)
def test_TAN_plot(data):
    # mpl_test_settings will automatically clean these internal side effects
    mpl_test_settings
    clf = TAN()
    dataset = load_iris(as_frame=True)
    clf.fit(*data, features=dataset["feature_names"], head=0)
    clf.plot("TAN Iris head=0")


def test_TAN_classifier_simple_init(data):
    dataset = load_iris(as_frame=True)
    features = dataset["feature_names"]
    clf = TAN(simple_init=True)
    clf.fit(*data, features=features, head=0)

    # Test default values of hyperparameters
    assert clf.simple_init

    clf.fit(*data)
    attribs = ["classes_", "X_", "y_", "head_", "features_", "class_name_"]
    for attr in attribs:
        assert hasattr(clf, attr)

    X = data[0]
    y = data[1]
    y_pred = clf.predict(X)
    y = y.reshape(-1, 1)
    assert y_pred.shape == (X.shape[0], 1)
    assert sum(y == y_pred) == 147


def test_TAN_wrong_num_features(data):
    clf = TAN()
    with pytest.raises(
        ValueError,
        match="Number of features does not match the number of columns in X",
    ):
        clf.fit(*data, features=["feature_1", "feature_2"])


def test_TAN_wrong_hyperparam(data):
    clf = TAN()
    with pytest.raises(ValueError, match="Unexpected argument: wrong_param"):
        clf.fit(*data, wrong_param="wrong_param")


def test_TAN_head_out_of_range(data):
    clf = TAN()
    with pytest.raises(ValueError, match="Head index out of range"):
        clf.fit(*data, head=4)


def test_TAN_error_size_predict(data):
    X, y = data
    clf = TAN()
    clf.fit(X, y)
    with pytest.raises(ValueError):
        X_diff_size = np.ones((10, X.shape[1] + 1))
        clf.predict(X_diff_size)
