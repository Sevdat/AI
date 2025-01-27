"""

This is a module for defining private helpers which do not depend on the
rest of NumPy.

Everything in here must be self-contained so that it can be
imported anywhere else without creating circular imports.
If a utility requires the import of NumPy, it probably belongs
in ``numpy.core``.
"""
from __future__ import annotations
from numpy._utils._convertions import asbytes
from numpy._utils._convertions import asunicode
from . import _convertions
from . import _inspect
__all__ = ['asbytes', 'asunicode', 'set_module']
def set_module(module):
    """
    Private decorator for overriding __module__ on a function or class.
    
        Example usage::
    
            @set_module('numpy')
            def example():
                pass
    
            assert example.__module__ == 'numpy'
        
    """
