import pytest

from sklearn.utils.estimator_checks import check_estimator

from bayesclass.clfs import TAN, KDB, AODE


@pytest.mark.parametrize("estimator", [TAN(), KDB(k=2), AODE()])
# @pytest.mark.parametrize("estimator", [AODE()])
def test_all_estimators(estimator):
    i = 0
    for estimator, test in check_estimator(estimator, generate_only=True):
        print(i := i + 1, test)
        # test(estimator)
