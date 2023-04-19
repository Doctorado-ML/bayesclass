import pytest
import numpy as np
from matplotlib.testing.decorators import image_comparison
from matplotlib.testing.conftest import mpl_test_settings
from pgmpy.models import BayesianNetwork


from bayesclass.clfs import KDBNew
from .._version import __version__


@pytest.fixture
def clf():
    return KDBNew(k=3)


def test_KDBNew_default_hyperparameters(data, clf):
    # Test default values of hyperparameters
    assert not clf.show_progress
    assert clf.random_state is None
    assert clf.theta == 0.03
    clf = KDBNew(show_progress=True, random_state=17, k=3)
    assert clf.show_progress
    assert clf.random_state == 17
    assert clf.k == 3
    clf.fit(*data)
    assert clf.class_name_ == "class"
    assert clf.feature_names_in_ == [
        "feature_0",
        "feature_1",
        "feature_2",
        "feature_3",
    ]


def test_KDBNew_version(clf):
    """Check KDBNew version."""
    assert __version__ == clf.version()


def test_KDBNew_nodes_edges(clf, data):
    assert clf.nodes_edges() == (0, 0)
    clf.fit(*data)
    assert clf.nodes_leaves() == (5, 9)


def test_KDBNew_states(clf, data):
    assert clf.states_ == 0
    clf.fit(*data)
    assert clf.states_ == 22
    assert clf.depth_ == clf.states_


def test_KDBNew_classifier(data, clf):
    clf.fit(*data)
    attribs = ["classes_", "X_", "y_", "feature_names_in_", "class_name_"]
    for attr in attribs:
        assert hasattr(clf, attr)
    X = data[0]
    y = data[1]
    y_pred = clf.predict(X)
    assert y_pred.shape == (X.shape[0],)
    assert sum(y == y_pred) == 145


def test_KDBNew_local_discretization(clf, data):
    expected = [[1, -1], -1, [0, 1, 3, -1], [1, -1]]
    clf.fit(*data)
    for feature in range(4):
        computed = clf.estimator_.discretizer_.target_[feature]
        print("computed:", computed)
        if type(computed) == list:
            for j, k in zip(expected[feature], computed):
                assert j == k
        else:
            assert (
                expected[feature]
                == clf.estimator_.discretizer_.target_[feature]
            )


@image_comparison(
    baseline_images=["line_dashes_KDBNew"],
    remove_text=True,
    extensions=["png"],
)
def test_KDBNew_plot(data, features, class_name, clf):
    # mpl_test_settings will automatically clean these internal side effects
    mpl_test_settings
    clf.fit(*data, features=features, class_name=class_name)
    clf.plot("KDBNew Iris")


def test_KDBNew_wrong_num_features(data, clf):
    with pytest.raises(
        ValueError,
        match="Number of features does not match the number of columns in X",
    ):
        clf.fit(*data, features=["feature_1", "feature_2"])


def test_KDBNew_wrong_hyperparam(data, clf):
    with pytest.raises(ValueError, match="Unexpected argument: wrong_param"):
        clf.fit(*data, wrong_param="wrong_param")


def test_KDBNew_error_size_predict(data, clf):
    X, y = data
    clf.fit(X, y)
    with pytest.raises(ValueError):
        X_diff_size = np.ones((10, X.shape[1] + 1))
        clf.predict(X_diff_size)


def test_KDBNew_dont_do_cycles():
    clf = KDBNew(k=4)
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
