from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
	Extension(
		"twin_tag_cython",
		["twin_tag_cython.pyx"],
		include_dirs=[np.get_include()],
		libraries=[],
	)
]

setup(
	name="twin_tag_cython",
	ext_modules=cythonize(extensions, language_level="3"),
)
