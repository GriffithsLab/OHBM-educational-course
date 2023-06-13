#!/usr/bin/env python

from setuptools import setup, find_packages

with open("README.rst", "r") as fh:
    long_description = fh.read()

with open('requirements.txt', 'r') as fp:
    install_requires = fp.read().splitlines()
install_requires = [i for i in install_requires if 'http' not in i]


setup(
    name="OHBM whole-brain modelling course",
    version="0.0",
    author="Davide Momi, Joana Cabral, John Griffiths",
    author_email="momi.davide89@gmail.com",
    description="Whole-brain, Connectome-based Models of Brain Dynamics: From Principles to Applications",
    keywords="OHBM2k23, Whole-brain Dynamics, Connectomics, Neuroimaging, fMRI, EEG, fNIRS, Machine Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires = install_requires,
    url='https://github.com/griffithslab/OHBM_whole_brain_modelling_course',
    license="BSD (3-clause)",
    entry_points={},#{"console_scripts": ["eegnb=eegnb.cli.__main__:main"]},
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)
