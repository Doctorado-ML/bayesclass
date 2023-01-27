import pytest

from sklearn.utils.estimator_checks import check_estimator

from bayesclass.clfs import TAN, KDB, AODE


# @pytest.mark.parametrize("estimators", [TAN(), KDB(k=2), AODE()])
@pytest.mark.parametrize("estimators", [AODE()])
def test_all_estimators(estimators):
    i = 0
    for estimator, test in check_estimator(estimators, generate_only=True):
        print(i := i + 1, test)
        # test(estimator)
