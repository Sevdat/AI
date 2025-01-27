"""

Module defining global singleton classes.

This module raises a RuntimeError if an attempt to reload it is made. In that
way the identities of the classes defined here are fixed and will remain so
even if numpy itself is reloaded. In particular, a function like the following
will still work correctly after numpy is reloaded::

    def foo(arg=np._NoValue):
        if arg is np._NoValue:
            ...

That was not the case when the singleton classes were defined in the numpy
``__init__.py`` file. See gh-7844 for a discussion of the reload problem that
motivated this module.

"""
from __future__ import annotations
import enum as enum
from numpy import _CopyMode
from numpy._utils import set_module as _set_module
import typing
__all__: list = ['_NoValue', '_CopyMode']
class _NoValueType:
    """
    Special keyword value.
    
        The instance of this class may be used as the default value assigned to a
        keyword if no other obvious default (e.g., `None`) is suitable,
    
        Common reasons for using this keyword are:
    
        - A new keyword is added to a function, and that function forwards its
          inputs to another function or method which can be defined outside of
          NumPy. For example, ``np.std(x)`` calls ``x.std``, so when a ``keepdims``
          keyword was added that could only be forwarded if the user explicitly
          specified ``keepdims``; downstream array libraries may not have added
          the same keyword, so adding ``x.std(..., keepdims=keepdims)``
          unconditionally could have broken previously working code.
        - A keyword is being deprecated, and a deprecation warning must only be
          emitted when the keyword is used.
    
        
    """
    _NoValueType__instance: typing.ClassVar[_NoValueType]  # value = <no value>
    @classmethod
    def __new__(cls):
        ...
    def __repr__(self):
        ...
_NoValue: _NoValueType  # value = <no value>
_is_loaded: bool = True
