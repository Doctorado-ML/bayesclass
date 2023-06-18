import pytest
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from matplotlib.testing.decorators import image_comparison
from matplotlib.testing.conftest import mpl_test_settings


from bayesclass.clfs import BoostAODE
from .._version import __version__


@pytest.fixture
def clf():
    return BoostAODE(random_state=17)


def test_BoostAODE_default_hyperparameters(data_disc, clf):
    # Test default values of hyperparameters
    assert not clf.show_progress
    assert clf.random_state == 17
    clf = BoostAODE(show_progress=True)
    assert clf.show_progress
    assert clf.random_state is None
    clf.fit(*data_disc)
    assert clf.class_name_ == "class"
    assert clf.feature_names_in_ == [
        "feature_0",
        "feature_1",
        "feature_2",
        "feature_3",
    ]


# @image_comparison(
#     baseline_images=["line_dashes_AODE"], remove_text=True, extensions=["png"]
# )
# def test_BoostAODE_plot(data_disc, features, clf):
#     # mpl_test_settings will automatically clean these internal side effects
#     mpl_test_settings
#     clf.fit(*data_disc, features=features)
#     clf.plot("AODE Iris")


# def test_BoostAODE_version(clf, features, data_disc):
#     """Check AODE version."""
#     assert __version__ == clf.version()
#     clf.fit(*data_disc, features=features)
#     assert __version__ == clf.version()


# def test_BoostAODE_nodes_edges(clf, data_disc):
#     assert clf.nodes_edges() == (0, 0)
#     clf.fit(*data_disc)
#     assert clf.nodes_leaves() == (20, 28)


# def test_BoostAODE_states(clf, data_disc):
#     assert clf.states_ == 0
#     clf.fit(*data_disc)
#     assert clf.states_ == 19
#     assert clf.depth_ == clf.states_


# def test_BoostAODE_classifier(data_disc, clf):
#     clf.fit(*data_disc)
#     attribs = [
#         "feature_names_in_",
#         "class_name_",
#         "n_features_in_",
#         "X_",
#         "y_",
#     ]
#     for attr in attribs:
#         assert hasattr(clf, attr)
#     X = data_disc[0]
#     y = data_disc[1]
#     y_pred = clf.predict(X)
#     assert y_pred.shape == (X.shape[0],)
#     assert sum(y == y_pred) == 146


# def test_BoostAODE_wrong_num_features(data_disc, clf):
#     with pytest.raises(
#         ValueError,
#         match="Number of features does not match the number of columns in X",
#     ):
#         clf.fit(*data_disc, features=["feature_1", "feature_2"])


# def test_BoostAODE_wrong_hyperparam(data_disc, clf):
#     with pytest.raises(ValueError, match="Unexpected argument: wrong_param"):
#         clf.fit(*data_disc, wrong_param="wrong_param")


# def test_BoostAODE_error_size_predict(data_disc, clf):
#     X, y = data_disc
#     clf.fit(X, y)
#     with pytest.raises(ValueError):
#         X_diff_size = np.ones((10, X.shape[1] + 1))
#         clf.predict(X_diff_size)
