import pytest
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import KBinsDiscretizer
from matplotlib.testing.decorators import image_comparison
from matplotlib.testing.conftest import mpl_test_settings


from bayesclass.clfs import AODENew
from .._version import __version__


@pytest.fixture
def data():
    X, y = load_iris(return_X_y=True)
    enc = KBinsDiscretizer(encode="ordinal")
    return enc.fit_transform(X), y


@pytest.fixture
def clf():
    return AODENew(random_state=17)


def test_AODENew_default_hyperparameters(data, clf):
    # Test default values of hyperparameters
    assert not clf.show_progress
    assert clf.random_state == 17
    clf = AODENew(show_progress=True)
    assert clf.show_progress
    assert clf.random_state is None
    clf.fit(*data)
    assert clf.class_name_ == "class"
    assert clf.feature_names_in_ == [
        "feature_0",
        "feature_1",
        "feature_2",
        "feature_3",
    ]


@image_comparison(
    baseline_images=["line_dashes_AODE"], remove_text=True, extensions=["png"]
)
def test_AODENew_plot(data, clf):
    # mpl_test_settings will automatically clean these internal side effects
    mpl_test_settings
    dataset = load_iris(as_frame=True)
    clf.fit(*data, features=dataset["feature_names"])
    clf.plot("AODE Iris")


def test_AODENew_version(clf):
    """Check AODE version."""
    assert __version__ == clf.version()


def test_AODENew_nodes_edges(clf, data):
    assert clf.nodes_edges() == (0, 0)
    clf.fit(*data)
    assert clf.nodes_leaves() == (20, 28)


def test_AODENew_states(clf, data):
    assert clf.states_ == 0
    clf.fit(*data)
    assert clf.states_ == 23
    assert clf.depth_ == clf.states_


def test_AODENew_classifier(data, clf):
    clf.fit(*data)
    attribs = [
        "classes_",
        "X_",
        "y_",
        "feature_names_in_",
        "class_name_",
        "n_features_in_",
    ]
    for attr in attribs:
        assert hasattr(clf, attr)
    X = data[0]
    y = data[1]
    y_pred = clf.predict(X)
    assert y_pred.shape == (X.shape[0],)
    assert sum(y == y_pred) == 147


def test_AODENew_wrong_num_features(data, clf):
    with pytest.raises(
        ValueError,
        match="Number of features does not match the number of columns in X",
    ):
        clf.fit(*data, features=["feature_1", "feature_2"])


def test_AODENew_wrong_hyperparam(data, clf):
    with pytest.raises(ValueError, match="Unexpected argument: wrong_param"):
        clf.fit(*data, wrong_param="wrong_param")


def test_AODENew_error_size_predict(data, clf):
    X, y = data
    clf.fit(X, y)
    with pytest.raises(ValueError):
        X_diff_size = np.ones((10, X.shape[1] + 1))
        clf.predict(X_diff_size)