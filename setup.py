# -*- coding: utf-8 -*-
# MIT License
#
# Copyright (c) 2022 Hao Mai & Pascal Audet
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in 
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Note that Blockly Earthquake Transformer (BET) is driven by Earthquake
Transformer V1.59 created by @author: mostafamousavi
Ref Repo: https://github.com/smousavi05/EQTransformer
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
#  	'pytest',
#  	'numpy',     # appox version: numpy 1.19.x but at least 1.19.2
#  	'keyring>=15.1',
#  	'pkginfo>=1.4.2',
#  	#'scipy>=1.4.1',
#     'scipy',
#  	#'tensorflow~=2.5.0', # tensorflow <2.7.0 needs numpy <1.20.0
#     'keras',
#     'tensorflow',
#     # 'tensorboard==2.0.2',
#     # 'tensorflow==2.0.0',
#     # 'tensorflow-estimator==2.0.1',
#  	'matplotlib',
#  	'pandas',
#  	'tqdm',
#     'h5py',
#     'obspy',
#     'jupyter',
# # 	'tqdm>=4.48.0',
# # 	'h5py~=3.1.0',
# # 	'obspy',
# # 	'jupyter',
#     'voila',
#     'voila-gridstack',
#     'widgetsnbextension',
#     'ipywidgets'
    ],

    python_requires='>=3.6',
)
