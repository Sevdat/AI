"""

============================
``ctypes`` Utility Functions
============================

See Also
--------
load_library : Load a C library.
ndpointer : Array restype/argtype with verification.
as_ctypes : Create a ctypes array from an ndarray.
as_array : Create an ndarray from a ctypes array.

References
----------
.. [1] "SciPy Cookbook: ctypes", https://scipy-cookbook.readthedocs.io/items/Ctypes.html

Examples
--------
Load the C library:

>>> _lib = np.ctypeslib.load_library('libmystuff', '.')     #doctest: +SKIP

Our result type, an ndarray that must be of type double, be 1-dimensional
and is C-contiguous in memory:

>>> array_1d_double = np.ctypeslib.ndpointer(
...                          dtype=np.double,
...                          ndim=1, flags='CONTIGUOUS')    #doctest: +SKIP

Our C-function typically takes an array and updates its values
in-place.  For example::

    void foo_func(double* x, int length)
    {
        int i;
        for (i = 0; i < length; i++) {
            x[i] = i*i;
        }
    }

We wrap it using:

>>> _lib.foo_func.restype = None                      #doctest: +SKIP
>>> _lib.foo_func.argtypes = [array_1d_double, c_int] #doctest: +SKIP

Then, we're ready to call ``foo_func``:

>>> out = np.empty(15, dtype=np.double)
>>> _lib.foo_func(out, len(out))                #doctest: +SKIP

"""
from __future__ import annotations
import ctypes as ctypes
from ctypes import c_longlong as c_intp
from ctypes import c_void_p as _ndptr_base
from numpy import asarray
from numpy.core.multiarray import flagsobj
from numpy import dtype as _dtype
from numpy import frombuffer
from numpy import integer
from numpy import ndarray
import os as os
__all__: list = ['load_library', 'ndpointer', 'c_intp', 'as_ctypes', 'as_array', 'as_ctypes_type']
class _concrete_ndptr(_ndptr):
    """
    
        Like _ndptr, but with `_shape_` and `_dtype_` specified.
    
        Notably, this means the pointer has enough information to reconstruct
        the array, which is not generally true.
        
    """
    def _check_retval_(self):
        """
        
                This method is called when this class is used as the .restype
                attribute for a shared-library function, to automatically wrap the
                pointer into an array.
                
        """
    @property
    def contents(self):
        """
        
                Get an ndarray viewing the data pointed to by this pointer.
        
                This mirrors the `contents` attribute of a normal ctypes pointer
                
        """
class _ndptr(ctypes.c_void_p):
    @classmethod
    def from_param(cls, obj):
        ...
def _ctype_from_dtype(dtype):
    ...
def _ctype_from_dtype_scalar(dtype):
    ...
def _ctype_from_dtype_structured(dtype):
    ...
def _ctype_from_dtype_subarray(dtype):
    ...
def _ctype_ndarray(element_type, shape):
    """
     Create an ndarray of the given element type and shape 
    """
def _flags_fromnum(num):
    ...
def _get_scalar_type_map():
    """
    
            Return a dictionary mapping native endian scalar dtype to ctypes types
            
    """
def _num_fromflags(flaglist):
    ...
def as_array(obj, shape = None):
    """
    
            Create a numpy array from a ctypes array or POINTER.
    
            The numpy array shares the memory with the ctypes object.
    
            The shape parameter must be given if converting from a ctypes POINTER.
            The shape parameter is ignored if converting from a ctypes array
            
    """
def as_ctypes(obj):
    """
    Create and return a ctypes object from a numpy array.  Actually
            anything that exposes the __array_interface__ is accepted.
    """
def as_ctypes_type(dtype):
    """
    
            Convert a dtype into a ctypes type.
    
            Parameters
            ----------
            dtype : dtype
                The dtype to convert
    
            Returns
            -------
            ctype
                A ctype scalar, union, array, or struct
    
            Raises
            ------
            NotImplementedError
                If the conversion is not possible
    
            Notes
            -----
            This function does not losslessly round-trip in either direction.
    
            ``np.dtype(as_ctypes_type(dt))`` will:
    
             - insert padding fields
             - reorder fields to be sorted by offset
             - discard field titles
    
            ``as_ctypes_type(np.dtype(ctype))`` will:
    
             - discard the class names of `ctypes.Structure`\\ s and
               `ctypes.Union`\\ s
             - convert single-element `ctypes.Union`\\ s into single-element
               `ctypes.Structure`\\ s
             - insert padding fields
    
            
    """
def load_library(libname, loader_path):
    """
    
            It is possible to load a library using
    
            >>> lib = ctypes.cdll[<full_path_name>] # doctest: +SKIP
    
            But there are cross-platform considerations, such as library file extensions,
            plus the fact Windows will just load the first library it finds with that name.
            NumPy supplies the load_library function as a convenience.
    
            .. versionchanged:: 1.20.0
                Allow libname and loader_path to take any
                :term:`python:path-like object`.
    
            Parameters
            ----------
            libname : path-like
                Name of the library, which can have 'lib' as a prefix,
                but without an extension.
            loader_path : path-like
                Where the library can be found.
    
            Returns
            -------
            ctypes.cdll[libpath] : library object
               A ctypes library object
    
            Raises
            ------
            OSError
                If there is no library with the expected extension, or the
                library is defective and cannot be loaded.
            
    """
def ndpointer(dtype = None, ndim = None, shape = None, flags = None):
    """
    
        Array-checking restype/argtypes.
    
        An ndpointer instance is used to describe an ndarray in restypes
        and argtypes specifications.  This approach is more flexible than
        using, for example, ``POINTER(c_double)``, since several restrictions
        can be specified, which are verified upon calling the ctypes function.
        These include data type, number of dimensions, shape and flags.  If a
        given array does not satisfy the specified restrictions,
        a ``TypeError`` is raised.
    
        Parameters
        ----------
        dtype : data-type, optional
            Array data-type.
        ndim : int, optional
            Number of array dimensions.
        shape : tuple of ints, optional
            Array shape.
        flags : str or tuple of str
            Array flags; may be one or more of:
    
              - C_CONTIGUOUS / C / CONTIGUOUS
              - F_CONTIGUOUS / F / FORTRAN
              - OWNDATA / O
              - WRITEABLE / W
              - ALIGNED / A
              - WRITEBACKIFCOPY / X
    
        Returns
        -------
        klass : ndpointer type object
            A type object, which is an ``_ndtpr`` instance containing
            dtype, ndim, shape and flags information.
    
        Raises
        ------
        TypeError
            If a given array does not satisfy the specified restrictions.
    
        Examples
        --------
        >>> clib.somefunc.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64,
        ...                                                  ndim=1,
        ...                                                  flags='C_CONTIGUOUS')]
        ... #doctest: +SKIP
        >>> clib.somefunc(np.array([1, 2, 3], dtype=np.float64))
        ... #doctest: +SKIP
    
        
    """
_flagdict: dict = {'OWNDATA': 4, 'O': 4, 'FORTRAN': 2, 'F': 2, 'CONTIGUOUS': 1, 'C': 1, 'ALIGNED': 256, 'A': 256, 'WRITEBACKIFCOPY': 8192, 'X': 8192, 'WRITEABLE': 1024, 'W': 1024, 'C_CONTIGUOUS': 1, 'F_CONTIGUOUS': 2}
_flagnames: list = ['C_CONTIGUOUS', 'F_CONTIGUOUS', 'ALIGNED', 'WRITEABLE', 'OWNDATA', 'WRITEBACKIFCOPY']
_pointer_type_cache: dict = {}
_scalar_type_map: dict  # value = {dtype('int8'): ctypes.c_byte, dtype('int16'): ctypes.c_short, dtype('int32'): ctypes.c_long, dtype('int64'): ctypes.c_longlong, dtype('uint8'): ctypes.c_ubyte, dtype('uint16'): ctypes.c_ushort, dtype('uint32'): ctypes.c_ulong, dtype('uint64'): ctypes.c_ulonglong, dtype('float32'): ctypes.c_float, dtype('float64'): ctypes.c_double, dtype('bool'): ctypes.c_bool}
