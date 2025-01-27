"""

DType classes and utility (:mod:`numpy.dtypes`)
===============================================

This module is home to specific dtypes related functionality and their classes.
For more general information about dtypes, also see `numpy.dtype` and
:ref:`arrays.dtypes`.

Similar to the builtin ``types`` module, this submodule defines types (classes)
that are not widely used directly.

.. versionadded:: NumPy 1.25

    The dtypes module is new in NumPy 1.25.  Previously DType classes were
    only accessible indirectly.


DType classes
-------------

The following are the classes of the corresponding NumPy dtype instances and
NumPy scalar types.  The classes can be used in ``isinstance`` checks and can
also be instantiated or used directly.  Direct use of these classes is not
typical, since their scalar counterparts (e.g. ``np.float64``) or strings
like ``"float64"`` can be used.

.. list-table::
    :header-rows: 1

    * - Group
      - DType class

    * - Boolean
      - ``BoolDType``

    * - Bit-sized integers
      - ``Int8DType``, ``UInt8DType``, ``Int16DType``, ``UInt16DType``,
        ``Int32DType``, ``UInt32DType``, ``Int64DType``, ``UInt64DType``

    * - C-named integers (may be aliases)
      - ``ByteDType``, ``UByteDType``, ``ShortDType``, ``UShortDType``,
        ``IntDType``, ``UIntDType``, ``LongDType``, ``ULongDType``,
        ``LongLongDType``, ``ULongLongDType``

    * - Floating point
      - ``Float16DType``, ``Float32DType``, ``Float64DType``,
        ``LongDoubleDType``

    * - Complex
      - ``Complex64DType``, ``Complex128DType``, ``CLongDoubleDType``

    * - Strings
      - ``BytesDType``, ``BytesDType``

    * - Times
      - ``DateTime64DType``, ``TimeDelta64DType``

    * - Others
      - ``ObjectDType``, ``VoidDType``

"""
from __future__ import annotations
import numpy
__all__: list = ['BoolDType', 'Int8DType', 'ByteDType', 'UInt8DType', 'UByteDType', 'Int16DType', 'ShortDType', 'UInt16DType', 'UShortDType', 'IntDType', 'UIntDType', 'Int32DType', 'LongDType', 'UInt32DType', 'ULongDType', 'Int64DType', 'LongLongDType', 'UInt64DType', 'ULongLongDType', 'Float16DType', 'Float32DType', 'Float64DType', 'LongDoubleDType', 'Complex64DType', 'Complex128DType', 'CLongDoubleDType', 'ObjectDType', 'BytesDType', 'StrDType', 'VoidDType', 'DateTime64DType', 'TimeDelta64DType']
class BoolDType(numpy.dtype):
    """
    DType class corresponding to the scalar type and dtype of the same name.
    
    Please see `numpy.dtype` for the typical way to create
    dtype instances and :ref:`arrays.dtypes` for additional
    information.
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
class BytesDType(numpy.dtype):
    """
    DType class corresponding to the scalar type and dtype of the same name.
    
    Please see `numpy.dtype` for the typical way to create
    dtype instances and :ref:`arrays.dtypes` for additional
    information.
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
class CLongDoubleDType(numpy.dtype):
    """
    DType class corresponding to the scalar type and dtype of the same name.
    
    Please see `numpy.dtype` for the typical way to create
    dtype instances and :ref:`arrays.dtypes` for additional
    information.
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
class Complex128DType(numpy.dtype):
    """
    DType class corresponding to the scalar type and dtype of the same name.
    
    Please see `numpy.dtype` for the typical way to create
    dtype instances and :ref:`arrays.dtypes` for additional
    information.
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
class Complex64DType(numpy.dtype):
    """
    DType class corresponding to the scalar type and dtype of the same name.
    
    Please see `numpy.dtype` for the typical way to create
    dtype instances and :ref:`arrays.dtypes` for additional
    information.
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
class DateTime64DType(numpy.dtype):
    """
    DType class corresponding to the scalar type and dtype of the same name.
    
    Please see `numpy.dtype` for the typical way to create
    dtype instances and :ref:`arrays.dtypes` for additional
    information.
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
class Float16DType(numpy.dtype):
    """
    DType class corresponding to the scalar type and dtype of the same name.
    
    Please see `numpy.dtype` for the typical way to create
    dtype instances and :ref:`arrays.dtypes` for additional
    information.
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
class Float32DType(numpy.dtype):
    """
    DType class corresponding to the scalar type and dtype of the same name.
    
    Please see `numpy.dtype` for the typical way to create
    dtype instances and :ref:`arrays.dtypes` for additional
    information.
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
class Float64DType(numpy.dtype):
    """
    DType class corresponding to the scalar type and dtype of the same name.
    
    Please see `numpy.dtype` for the typical way to create
    dtype instances and :ref:`arrays.dtypes` for additional
    information.
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
class Int16DType(numpy.dtype):
    """
    DType class corresponding to the scalar type and dtype of the same name.
    
    Please see `numpy.dtype` for the typical way to create
    dtype instances and :ref:`arrays.dtypes` for additional
    information.
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
class Int32DType(numpy.dtype):
    """
    DType class corresponding to the scalar type and dtype of the same name.
    
    Please see `numpy.dtype` for the typical way to create
    dtype instances and :ref:`arrays.dtypes` for additional
    information.
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
class Int64DType(numpy.dtype):
    """
    DType class corresponding to the scalar type and dtype of the same name.
    
    Please see `numpy.dtype` for the typical way to create
    dtype instances and :ref:`arrays.dtypes` for additional
    information.
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
class Int8DType(numpy.dtype):
    """
    DType class corresponding to the scalar type and dtype of the same name.
    
    Please see `numpy.dtype` for the typical way to create
    dtype instances and :ref:`arrays.dtypes` for additional
    information.
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
class IntDType(numpy.dtype):
    """
    DType class corresponding to the scalar type and dtype of the same name.
    
    Please see `numpy.dtype` for the typical way to create
    dtype instances and :ref:`arrays.dtypes` for additional
    information.
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
class LongDoubleDType(numpy.dtype):
    """
    DType class corresponding to the scalar type and dtype of the same name.
    
    Please see `numpy.dtype` for the typical way to create
    dtype instances and :ref:`arrays.dtypes` for additional
    information.
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
class ObjectDType(numpy.dtype):
    """
    DType class corresponding to the scalar type and dtype of the same name.
    
    Please see `numpy.dtype` for the typical way to create
    dtype instances and :ref:`arrays.dtypes` for additional
    information.
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
class StrDType(numpy.dtype):
    """
    DType class corresponding to the scalar type and dtype of the same name.
    
    Please see `numpy.dtype` for the typical way to create
    dtype instances and :ref:`arrays.dtypes` for additional
    information.
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
class TimeDelta64DType(numpy.dtype):
    """
    DType class corresponding to the scalar type and dtype of the same name.
    
    Please see `numpy.dtype` for the typical way to create
    dtype instances and :ref:`arrays.dtypes` for additional
    information.
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
class UInt16DType(numpy.dtype):
    """
    DType class corresponding to the scalar type and dtype of the same name.
    
    Please see `numpy.dtype` for the typical way to create
    dtype instances and :ref:`arrays.dtypes` for additional
    information.
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
class UInt32DType(numpy.dtype):
    """
    DType class corresponding to the scalar type and dtype of the same name.
    
    Please see `numpy.dtype` for the typical way to create
    dtype instances and :ref:`arrays.dtypes` for additional
    information.
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
class UInt64DType(numpy.dtype):
    """
    DType class corresponding to the scalar type and dtype of the same name.
    
    Please see `numpy.dtype` for the typical way to create
    dtype instances and :ref:`arrays.dtypes` for additional
    information.
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
class UInt8DType(numpy.dtype):
    """
    DType class corresponding to the scalar type and dtype of the same name.
    
    Please see `numpy.dtype` for the typical way to create
    dtype instances and :ref:`arrays.dtypes` for additional
    information.
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
class UIntDType(numpy.dtype):
    """
    DType class corresponding to the scalar type and dtype of the same name.
    
    Please see `numpy.dtype` for the typical way to create
    dtype instances and :ref:`arrays.dtypes` for additional
    information.
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
class VoidDType(numpy.dtype):
    """
    DType class corresponding to the scalar type and dtype of the same name.
    
    Please see `numpy.dtype` for the typical way to create
    dtype instances and :ref:`arrays.dtypes` for additional
    information.
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
def _add_dtype_helper(DType, alias):
    ...
ByteDType = Int8DType
LongDType = Int32DType
LongLongDType = Int64DType
ShortDType = Int16DType
UByteDType = UInt8DType
ULongDType = UInt32DType
ULongLongDType = UInt64DType
UShortDType = UInt16DType
