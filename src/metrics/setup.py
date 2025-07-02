import os

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "src.metrics._confusion_matrix",
        ["_confusion_matrix.pyx"],
        include_dirs=[numpy.get_include()],
        libraries=["m"] if os.name == "posix" else [],
        extra_compile_args=["-O3"],
    ),
    Extension(
        "src.metrics._classification_performance_evaluator",
        ["_classification_performance_evaluator.pyx"],
        include_dirs=[numpy.get_include()],
        libraries=["m"] if os.name == "posix" else [],
        extra_compile_args=["-O3"],
    ),
]

import numpy
from setuptools import setup
from Cython.Build import cythonize


# Ensure the output directory exists
os.makedirs("src/metrics", exist_ok=True)

setup(
    name="metrics",
    ext_modules=cythonize(["_confusion_matrix.pyx", "_classification_performance_evaluator.pyx"]),
    include_dirs=[numpy.get_include()]
)
