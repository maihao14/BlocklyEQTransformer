#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 09:31:20 2022

@author: Hao Mai & Pascal Audet
"""

from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="BlocklyEQTransformer",
    author="Hao Mai & Pascal Audet",
    version="0.0.1",
    author_email="hmai090@uottawa.ca",
    description="A Python Toolbox for Customizing Seismic Phase Pickers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/maihao14/BlocklyEQTransformer",
    license="MIT",
    packages=find_packages(),
    keywords='Seismology, Earthquakes Detection, Phase Picking, Deep Learning',
    install_requires=[
	'pytest',
	'numpy~=1.19.2',     # appox version: numpy 1.19.x but at least 1.19.2
	'keyring>=15.1',
	'pkginfo>=1.4.2',
	'scipy>=1.4.1',
	'tensorflow~=2.5.0', # tensorflow <2.7.0 needs numpy <1.20.0
	'keras==2.3.1',
	'matplotlib',
	'pandas',
	'tqdm>=4.48.0',
	'h5py~=3.1.0',
	'obspy',
	'jupyter',
    'voila',
    'voila-gridstack',
    'widgetsnbextension',
    'ipywidgets'
    ],

    python_requires='>=3.6',
)
