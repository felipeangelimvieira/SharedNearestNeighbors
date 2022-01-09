#!/usr/bin/env python

"""The setup script."""

from pathlib import Path
from setuptools import setup, find_packages
from shared_nearest_neighbors import __version__, __author__, __email__

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()


with open(Path(__file__).parent / Path("requirements.txt"), "r") as requirements_file:
    requirements = requirements_file.readlines()

test_requirements = ['pytest>=3', ]

setup(
    author=__author__,
    author_email=__email__,
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="SNN Algorithm",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='shared_nearest_neighbors',
    name='shared_nearest_neighbors',
    packages=find_packages(include=['shared_nearest_neighbors', 'shared_nearest_neighbors.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/felipeangelimvieira/shared_nearest_neighbors',
    version=__version__,
    zip_safe=False,
)
