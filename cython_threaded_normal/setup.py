from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
	Extension(
		"prec_landing_cython_threading",
		["prec_landing_cython_threading.pyx"],
		include_dirs=[np.get_include()],
		libraries=[],
	)
]

setup(
	name="prec_landing_cython_threading",
	ext_modules=cythonize(extensions, language_level="3"),
)