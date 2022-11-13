import pytest

from sklearn.utils.estimator_checks import check_estimator

from bayesclass import TAN


@pytest.mark.parametrize("estimator", [TAN()])
def test_all_estimators(estimator):
    return check_estimator(estimator)
