"""
    Calling
    $python setup.py build_ext --inplace
    will build the extension library in the current file.
"""

from setuptools import Extension, setup
from torch.utils import cpp_extension

setup(
    ext_modules=[
        Extension(
            name="bayesclass.cppSelectFeatures",
            sources=[
                "bayesclass/cSelectFeatures.pyx",
                "bayesclass/FeatureSelect.cpp",
            ],
            language="c++",
            include_dirs=["bayesclass"],
            extra_compile_args=[
                "-std=c++17",
            ],
        ),
        Extension(
            name="bayesclass.cppBayesNetwork",
            sources=[
                "bayesclass/BayesNetwork.pyx",
                "bayesclass/Network.cc",
                "bayesclass/Node.cc",
            ],
            include_dirs=cpp_extension.include_paths(),
            language="c++",
            extra_compile_args=[
                "-std=c++17",
            ],
        ),
    ]
)
