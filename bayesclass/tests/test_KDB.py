import pytest
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from matplotlib.testing.decorators import image_comparison
from matplotlib.testing.conftest import mpl_test_settings
from pgmpy.models import BayesianNetwork


from bayesclass.clfs import KDB
from .._version import __version__


@pytest.fixture
def clf():
    return KDB(k=3)


def test_KDB_default_hyperparameters(data_disc, clf):
    # Test default values of hyperparameters
    assert not clf.show_progress
    assert clf.random_state is None
    assert clf.theta == 0.03
    clf = KDB(show_progress=True, random_state=17, k=3)
    assert clf.show_progress
    assert clf.random_state == 17
    assert clf.k == 3
    clf.fit(*data_disc)
    assert clf.class_name_ == "class"
    assert clf.feature_names_in_ == [
        "feature_0",
        "feature_1",
        "feature_2",
        "feature_3",
    ]


def test_KDB_version(clf):
    """Check KDB version."""
    assert __version__ == clf.version()


def test_KDB_nodes_edges(clf, data_disc):
    assert clf.nodes_edges() == (0, 0)
    clf.fit(*data_disc)
    assert clf.nodes_leaves() == (5, 9)


def test_KDB_states(clf, data_disc):
    assert clf.states_ == 0
    clf.fit(*data_disc)
    assert clf.states_ == 19
    assert clf.depth_ == clf.states_


def test_KDB_classifier(data_disc, clf):
    clf.fit(*data_disc)
    attribs = ["classes_", "X_", "y_", "feature_names_in_", "class_name_"]
    for attr in attribs:
        assert hasattr(clf, attr)
    X = data_disc[0]
    y = data_disc[1]
    y_pred = clf.predict(X)
    assert y_pred.shape == (X.shape[0],)
    assert sum(y == y_pred) == 146


def test_KDB_classifier_weighted(data_disc, clf):
    sample_weight = [1] * data_disc[0].shape[0]
    sample_weight[:50] = [0] * 50
    clf.fit(*data_disc, sample_weight=sample_weight, weighted=True)
    assert clf.score(*data_disc) == 0.64


@image_comparison(
    baseline_images=["line_dashes_KDB"], remove_text=True, extensions=["png"]
)
def test_KDB_plot(data_disc, features, clf):
    # mpl_test_settings will automatically clean these internal side effects
    mpl_test_settings
    clf.fit(*data_disc, features=features)
    clf.plot("KDB Iris")


def test_KDB_wrong_num_features(data_disc, clf):
    with pytest.raises(
        ValueError,
        match="Number of features does not match the number of columns in X",
    ):
        clf.fit(*data_disc, features=["feature_1", "feature_2"])


def test_KDB_wrong_hyperparam(data_disc, clf):
    with pytest.raises(ValueError, match="Unexpected argument: wrong_param"):
        clf.fit(*data_disc, wrong_param="wrong_param")


def test_KDB_error_size_predict(data_disc, clf):
    X, y = data_disc
    clf.fit(X, y)
    with pytest.raises(ValueError):
        X_diff_size = np.ones((10, X.shape[1] + 1))
        clf.predict(X_diff_size)


def test_KDB_dont_do_cycles():
    clf = KDB(k=4)
    dag = BayesianNetwork()
    clf.feature_names_in_ = [
        "feature_0",
        "feature_1",
        "feature_2",
        "feature_3",
    ]
    nodes = list(range(4))
    weights = np.ones((4, 4))
    for idx in range(1, 4):
        dag.add_edge(clf.feature_names_in_[0], clf.feature_names_in_[idx])
    dag.add_edge(clf.feature_names_in_[1], clf.feature_names_in_[2])
    dag.add_edge(clf.feature_names_in_[1], clf.feature_names_in_[3])
    dag.add_edge(clf.feature_names_in_[2], clf.feature_names_in_[3])
    for idx in range(4):
        clf._add_m_edges(dag, idx, nodes, weights)
        assert len(dag.edges()) == 6
