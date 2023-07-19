"""
    Calling
    $python setup.py build_ext --inplace
    will build the extension library in the current file.
"""

from setuptools import Extension, setup
from torch.utils.cpp_extension import (
    BuildExtension,
    CppExtension,
    include_paths,
)


setup(
    ext_modules=[
        Extension(
            name="bayesclass.cppSelectFeatures",
            sources=[
                "bayesclass/cSelectFeatures.pyx",
                "bayesclass/cpp/FeatureSelect.cpp",
            ],
            language="c++",
            include_dirs=["bayesclass"],
            extra_compile_args=[
                "-std=c++17",
            ],
        ),
        CppExtension(
            name="bayesclass.BayesNet",
            sources=[
                "bayesclass/BayesNetwork.pyx",
                "bayesclass/cpp/Network.cc",
                "bayesclass/cpp/Node.cc",
                "bayesclass/cpp/Metrics.cc",
                "bayesclass/cpp/utils.cc",
                "bayesclass/cpp/Mst.cc",
                "bayesclass/cpp/BaseClassifier.cc",
                "bayesclass/cpp/Ensemble.cc",
                "bayesclass/cpp/TAN.cc",
                "bayesclass/cpp/KDB.cc",
                "bayesclass/cpp/SPODE.cc",
                "bayesclass/cpp/AODE.cc",
            ],
            include_dirs=include_paths(),
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
