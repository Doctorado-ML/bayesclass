import pytest
import numpy as np
from matplotlib.testing.decorators import image_comparison
from matplotlib.testing.conftest import mpl_test_settings


from bayesclass.clfs import AODENew
from .._version import __version__


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
    baseline_images=["line_dashes_AODENew"],
    remove_text=True,
    extensions=["png"],
)
def test_AODENew_plot(data, features, clf):
    # mpl_test_settings will automatically clean these internal side effects
    mpl_test_settings
    clf.fit(*data, features=features)
    clf.plot("AODE Iris")


def test_AODENew_version(clf, data):
    """Check AODENew version."""
    assert __version__ == clf.version()
    clf.fit(*data)
    assert __version__ == clf.version()


def test_AODENew_nodes_edges(clf, data):
    assert clf.nodes_edges() == (0, 0)
    clf.fit(*data)
    assert clf.nodes_leaves() == (20, 28)


def test_AODENew_states(clf, data):
    assert clf.states_ == 0
    clf.fit(*data)
    assert clf.states_ == 17.75
    assert clf.depth_ == clf.states_


def test_AODENew_classifier(data, clf):
    clf.fit(*data)
    attribs = [
        "feature_names_in_",
        "class_name_",
        "n_features_in_",
        "X_",
        "y_",
    ]
    for attr in attribs:
        assert hasattr(clf, attr)
    X = data[0]
    y = data[1]
    y_pred = clf.predict(X)
    assert y_pred.shape == (X.shape[0],)
    assert sum(y == y_pred) == 146


def test_AODENew_local_discretization(clf, data_disc):
    expected_data = [
        [-1, [0, -1], [0, -1], [0, -1]],
        [[1, -1], -1, [1, -1], [1, -1]],
        [[2, -1], [2, -1], -1, [2, -1]],
        [[3, -1], [3, -1], [3, -1], -1],
    ]
    clf.fit(*data_disc)
    for idx, estimator in enumerate(clf.estimators_):
        expected = expected_data[idx]
        for feature in range(4):
            computed = estimator.discretizer_.target_[feature]
            if type(computed) == list:
                for j, k in zip(expected[feature], computed):
                    assert j == k
            else:
                assert (
                    expected[feature]
                    == estimator.discretizer_.target_[feature]
                )


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
