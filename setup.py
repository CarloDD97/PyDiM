from setuptools import find_packages, setup
from pathlib import Path

setup(
    name='PyDiM',
    packages=['PyDiM'],
    version='0.1.0',
    description='Python Innovation Diffusion Models',
    long_description=Path("README.md").read_text(encoding="utf-8"),
    author='Carlo De Dominicis',
    author_email="carlo.dedominicis.1@studenti.unipd.it",
    project_urls={
        'Source Code' : 'https://github.com/CarloDD97/PyDiM',
    },
    license='MIT',
    install_requires=[
        "numpy>=1.21",
        "pandas>=1.5.0",
        "scipy>=1.9.0",
        "statsmodels>=0.13.5",
        "matplotlib>=3.6.0"
    ],
    include_package_data=True,
    package_data={'DIMORA' : ['sample_data/*.xlsx']},
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    test_suite='test',
)