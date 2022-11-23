import pytest
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import KBinsDiscretizer
from matplotlib.testing.decorators import image_comparison
from matplotlib.testing.conftest import mpl_test_settings


from bayesclass.clfs import TAN
from .._version import __version__


@pytest.fixture
def data():
    X, y = load_iris(return_X_y=True)
    enc = KBinsDiscretizer(encode="ordinal")
    return enc.fit_transform(X), y


@pytest.fixture
def clf():
    return TAN()


def test_TAN_default_hyperparameters(data, clf):
    # Test default values of hyperparameters
    assert not clf.show_progress
    assert clf.random_state is None
    clf = TAN(show_progress=True, random_state=17)
    assert clf.show_progress
    assert clf.random_state == 17
    clf.fit(*data)
    assert clf.head_ == 0
    assert clf.class_name_ == "class"
    assert clf.features_ == [
        "feature_0",
        "feature_1",
        "feature_2",
        "feature_3",
    ]


def test_TAN_version(clf):
    """Check TAN version."""
    assert __version__ == clf.version()


def test_TAN_nodes_leaves(clf):
    assert clf.nodes_leaves() == (0, 0)


def test_TAN_random_head(data):
    clf = TAN(random_state=17)
    clf.fit(*data, head="random")
    assert clf.head_ == 3


def test_TAN_classifier(data, clf):
    clf.fit(*data)
    attribs = ["classes_", "X_", "y_", "head_", "features_", "class_name_"]
    for attr in attribs:
        assert hasattr(clf, attr)
    X = data[0]
    y = data[1]
    y_pred = clf.predict(X)
    assert y_pred.shape == (X.shape[0],)
    assert sum(y == y_pred) == 147


@image_comparison(
    baseline_images=["line_dashes_TAN"], remove_text=True, extensions=["png"]
)
def test_TAN_plot(data, clf):
    # mpl_test_settings will automatically clean these internal side effects
    mpl_test_settings
    dataset = load_iris(as_frame=True)
    clf.fit(*data, features=dataset["feature_names"], head=0)
    clf.plot("TAN Iris head=0")


def test_KDB_wrong_num_features(data, clf):
    with pytest.raises(
        ValueError,
        match="Number of features does not match the number of columns in X",
    ):
        clf.fit(*data, features=["feature_1", "feature_2"])


def test_TAN_wrong_hyperparam(data, clf):
    with pytest.raises(ValueError, match="Unexpected argument: wrong_param"):
        clf.fit(*data, wrong_param="wrong_param")


def test_TAN_head_out_of_range(data, clf):
    with pytest.raises(ValueError, match="Head index out of range"):
        clf.fit(*data, head=4)


def test_TAN_error_size_predict(data, clf):
    X, y = data
    clf.fit(X, y)
    with pytest.raises(ValueError):
        X_diff_size = np.ones((10, X.shape[1] + 1))
        clf.predict(X_diff_size)
