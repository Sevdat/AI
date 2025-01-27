"""

Python 3.X compatibility tools.

While this file was originally intended for Python 2 -> 3 transition,
it is now used to create a compatibility layer between different
minor versions of Python 3.

While the active version of numpy may not support a given version of python, we
allow downstream libraries to continue to use these shims for forward
compatibility with numpy while they transition their code to newer versions of
Python.
"""
from __future__ import annotations
from builtins import bytes
from builtins import int as long
from builtins import str as basestring
from builtins import str as unicode
import io as io
from nt import fspath as os_fspath
import os as os
from os import PathLike as os_PathLike
from pathlib import Path
import pickle as pickle
import sys as sys
__all__: list = ['bytes', 'asbytes', 'isfileobj', 'getexception', 'strchar', 'unicode', 'asunicode', 'asbytes_nested', 'asunicode_nested', 'asstr', 'open_latin1', 'long', 'basestring', 'sixu', 'integer_types', 'is_pathlib_path', 'npy_load_module', 'Path', 'pickle', 'contextlib_nullcontext', 'os_fspath', 'os_PathLike']
class contextlib_nullcontext:
    """
    Context manager that does no additional processing.
    
        Used as a stand-in for a normal context manager, when a particular
        block of code is only sometimes used with a normal context manager:
    
        cm = optional_cm if condition else nullcontext()
        with cm:
            # Perform operation, using optional_cm if condition is True
    
        .. note::
            Prefer using `contextlib.nullcontext` instead of this context manager.
        
    """
    def __enter__(self):
        ...
    def __exit__(self, *excinfo):
        ...
    def __init__(self, enter_result = None):
        ...
def asbytes(s):
    ...
def asbytes_nested(x):
    ...
def asstr(s):
    ...
def asunicode(s):
    ...
def asunicode_nested(x):
    ...
def getexception():
    ...
def is_pathlib_path(obj):
    """
    
        Check whether obj is a `pathlib.Path` object.
    
        Prefer using ``isinstance(obj, os.PathLike)`` instead of this function.
        
    """
def isfileobj(f):
    ...
def npy_load_module(name, fn, info = None):
    """
    
        Load a module. Uses ``load_module`` which will be deprecated in python
        3.12. An alternative that uses ``exec_module`` is in
        numpy.distutils.misc_util.exec_mod_from_location
    
        .. versionadded:: 1.11.2
    
        Parameters
        ----------
        name : str
            Full module name.
        fn : str
            Path to module file.
        info : tuple, optional
            Only here for backward compatibility with Python 2.*.
    
        Returns
        -------
        mod : module
    
        
    """
def open_latin1(filename, mode = 'r'):
    ...
def sixu(s):
    ...
integer_types: tuple = (int)
strchar: str = 'U'
