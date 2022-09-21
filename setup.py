import os

import pkg_resources
from setuptools import setup, find_packages

with open('README.rst') as f:
    long_description = f.read()

setup(
    name='meerqat',
    packages=find_packages(),
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],

    long_description=long_description,
    long_description_content_type='text/markdown',

    author='Paul Lerner',
    author_email='paul.lerner@lisn.upsaclay.fr',
    url='https://github.com/PaulLerner/ViQuAE',

    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering"
    ],
)

