import doctest
import bayesclass


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(bayesclass))
    return tests
