"""

Compatibility module.

This module contains duplicated code from Python itself or 3rd party
extensions, which may be included for the following reasons:

  * compatibility
  * we may only need a small subset of the copied library/module

"""
from __future__ import annotations
from builtins import bytes
from builtins import int as long
from builtins import str as unicode
from builtins import str as basestring
from nt import fspath as os_fspath
from numpy._utils import _inspect
from numpy._utils._inspect import formatargspec
from numpy._utils._inspect import getargspec
from numpy.compat.py3k import asbytes
from numpy.compat.py3k import asbytes_nested
from numpy.compat.py3k import asstr
from numpy.compat.py3k import asunicode
from numpy.compat.py3k import asunicode_nested
from numpy.compat.py3k import contextlib_nullcontext
from numpy.compat.py3k import getexception
from numpy.compat.py3k import is_pathlib_path
from numpy.compat.py3k import isfileobj
from numpy.compat.py3k import npy_load_module
from numpy.compat.py3k import open_latin1
from numpy.compat.py3k import sixu
from os import PathLike as os_PathLike
from pathlib import Path
import pickle as pickle
from . import py3k
__all__: list = ['getargspec', 'formatargspec', 'bytes', 'asbytes', 'isfileobj', 'getexception', 'strchar', 'unicode', 'asunicode', 'asbytes_nested', 'asunicode_nested', 'asstr', 'open_latin1', 'long', 'basestring', 'sixu', 'integer_types', 'is_pathlib_path', 'npy_load_module', 'Path', 'pickle', 'contextlib_nullcontext', 'os_fspath', 'os_PathLike']
integer_types: tuple = (int)
strchar: str = 'U'
