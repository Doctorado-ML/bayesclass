import pytest

from sklearn.utils.estimator_checks import check_estimator

from bayesclass import TemplateEstimator
from bayesclass import TemplateClassifier
from bayesclass import TemplateTransformer


@pytest.mark.parametrize(
    "estimator", [TemplateEstimator(), TemplateTransformer(), TemplateClassifier()]
)
def test_all_estimators(estimator):
    return check_estimator(estimator)
