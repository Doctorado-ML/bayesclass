import pytest
import numpy as np

from sklearn.utils.estimator_checks import check_estimator

from bayesclass.clfs import BayesBase, TAN, KDB, AODE


def test_more_tags():
    expected = {
        "requires_positive_X": True,
        "requires_positive_y": True,
        "preserve_dtype": [np.int32, np.int64],
        "requires_y": True,
    }
    clf = BayesBase(None, True)
    computed = clf._more_tags()
    for key, value in expected.items():
        assert key in computed
        assert computed[key] == value


# @pytest.mark.parametrize("estimators", [TAN(), KDB(k=2), AODE()])
@pytest.mark.parametrize("estimators", [AODE()])
def test_all_estimators(estimators):
    i = 0
    for estimator, test in check_estimator(estimators, generate_only=True):
        print(i := i + 1, test)
        # test(estimator)
