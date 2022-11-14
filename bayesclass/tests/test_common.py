import pytest

from sklearn.utils.estimator_checks import check_estimator

from bayesclass import TAN


@pytest.mark.parametrize("estimator", [TAN()])
def test_all_estimators(estimator):
    i = 0
    for estimator, test in check_estimator(estimator, generate_only=True):
        print(i := i + 1, test, "classes_")
        # test(estimator)
