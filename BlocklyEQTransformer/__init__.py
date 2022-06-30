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
"""
Note that Blockly Earthquake Transformer (BET) is driven by Earthquake Transformer
V1.59 created by @author: mostafamousavi
Ref Repo: https://github.com/smousavi05/EQTransformer
"""
name='BlocklyEQTransformer'

import importlib
import sys
import warnings

from BlocklyEQTransformer.core.trainer import trainer
from BlocklyEQTransformer.core.tester import tester
from BlocklyEQTransformer.core.predictor import predictor
from BlocklyEQTransformer.core.mseed_predictor import mseed_predictor
from BlocklyEQTransformer.core.EqT_utils import *
from BlocklyEQTransformer.utils.associator import run_associator
from BlocklyEQTransformer.utils.downloader import downloadMseeds, makeStationList, downloadSacs
from BlocklyEQTransformer.utils.hdf5_maker import preprocessor
from BlocklyEQTransformer.utils.plot import plot_detections, plot_data_chart

__all__ = ['core', 'utils'] #'test'
__version__ = '0.0.1'
_import_map = {}

class EqtDeprecationWarning(UserWarning):
    """
    Force pop-up of warnings.
    """
    pass

if sys.version_info.major < 3:
    raise NotImplementedError(
        "EqT no longer supports Python 2.x.")


class EqtRestructureAndLoad(object):
    """
    Path finder and module loader for transitioning
    """

    def find_module(self, fullname, path=None):
        # Compatibility with namespace paths.
        if hasattr(path, "_path"):
            path = path._path

        if not path or not path[0].startswith(__path__[0]):
            return None

        for key in _import_map.keys():
            if fullname.startswith(key):
                break
        else:
            return None
        return self

    def load_module(self, name):
        # Use cached modules.
        if name in sys.modules:
            return sys.modules[name]
        # Otherwise check if the name is part of the import map.
        elif name in _import_map:
            new_name = _import_map[name]
        else:
            new_name = name
            for old, new in _import_map.items():
                if not new_name.startswith(old):
                    continue
                new_name = new_name.replace(old, new)
                break
            else:
                return None

        # Don't load again if already loaded.
        if new_name in sys.modules:
            module = sys.modules[new_name]
        else:
            module = importlib.import_module(new_name)

        # Warn here as at this point the module has already been imported.
        warnings.warn("Module '%s' is deprecated and will stop working "
                      "with the next delphi version. Please import module "
                      "'%s' instead." % (name, new_name),
                      EqtDeprecationWarning)
        sys.modules[new_name] = module
        sys.modules[name] = module
        return module


sys.meta_path.append(EqtRestructureAndLoad())
