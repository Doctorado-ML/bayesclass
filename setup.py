"""
    Calling
    $python setup.py build_ext --inplace
    will build the extension library in the current file.
"""

from setuptools import Extension, setup

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
    ]
)
