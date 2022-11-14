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
                'matplotlib',
                'h5py==2.10.0',
                'tqdm',
                'obspy',
                'pandas',
                'keras==2.3.1',
                'tensorflow==2.0.0',
                'pytest',
                'jupyter==1.0.0',
                'voila==0.3.5',
                'ipywidgets~=7.6.5',
                'protobuf<=3.20.1'

    ],

    python_requires='>=3.7',
)
