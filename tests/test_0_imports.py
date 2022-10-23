# -*- coding: utf-8 -*-
# MIT License
#
# Copyright (c) 2022 Hao Mai & Pascal Audet
#
# Note that Blockly Earthquake Transformer (BET) is driven by Earthquake Transformer
# V1.59 created by @author: mostafamousavi
# Ref Repo: https://github.com/smousavi05/EQTransformer
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

def test_import_bocklyeqtransformer():
    """Test imports."""
    import BlocklyEQTransformer
    from BlocklyEQTransformer.core.trainer import trainer
    from BlocklyEQTransformer.core.predictor import predictor
    from BlocklyEQTransformer.core.tester import tester
    print("Imports are working!")
