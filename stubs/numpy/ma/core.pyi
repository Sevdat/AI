"""

numpy.ma : a package to handle missing or invalid values.

This package was initially written for numarray by Paul F. Dubois
at Lawrence Livermore National Laboratory.
In 2006, the package was completely rewritten by Pierre Gerard-Marchant
(University of Georgia) to make the MaskedArray class a subclass of ndarray,
and to improve support of structured arrays.


Copyright 1999, 2000, 2001 Regents of the University of California.
Released for unlimited redistribution.

* Adapted for numpy_core 2005 by Travis Oliphant and (mainly) Paul Dubois.
* Subclassing of the base `ndarray` 2006 by Pierre Gerard-Marchant
  (pgmdevlist_AT_gmail_DOT_com)
* Improvements suggested by Reggie Dugard (reggie_AT_merfinllc_DOT_com)

.. moduleauthor:: Pierre Gerard-Marchant

"""
from __future__ import annotations
from _functools import reduce
import builtins as builtins
from builtins import bytes
from builtins import int as long
from builtins import str as unicode
import inspect as inspect
import numpy
import numpy as np
import numpy._globals
from numpy._utils._inspect import formatargspec
from numpy._utils._inspect import getargspec
from numpy import amax
from numpy import amin
from numpy import array as narray
from numpy import bool_ as MaskType
from numpy import bool_
from numpy.core import multiarray as mu
from numpy.core.numeric import normalize_axis_tuple
from numpy.core import numerictypes as ntypes
import numpy.core.numerictypes
from numpy.core import umath
from numpy import expand_dims
from numpy import iscomplexobj
from numpy import ndarray
import operator as operator
import re as re
import textwrap as textwrap
import typing
import warnings as warnings
__all__: list = ['MAError', 'MaskError', 'MaskType', 'MaskedArray', 'abs', 'absolute', 'add', 'all', 'allclose', 'allequal', 'alltrue', 'amax', 'amin', 'angle', 'anom', 'anomalies', 'any', 'append', 'arange', 'arccos', 'arccosh', 'arcsin', 'arcsinh', 'arctan', 'arctan2', 'arctanh', 'argmax', 'argmin', 'argsort', 'around', 'array', 'asanyarray', 'asarray', 'bitwise_and', 'bitwise_or', 'bitwise_xor', 'bool_', 'ceil', 'choose', 'clip', 'common_fill_value', 'compress', 'compressed', 'concatenate', 'conjugate', 'convolve', 'copy', 'correlate', 'cos', 'cosh', 'count', 'cumprod', 'cumsum', 'default_fill_value', 'diag', 'diagonal', 'diff', 'divide', 'empty', 'empty_like', 'equal', 'exp', 'expand_dims', 'fabs', 'filled', 'fix_invalid', 'flatten_mask', 'flatten_structured_array', 'floor', 'floor_divide', 'fmod', 'frombuffer', 'fromflex', 'fromfunction', 'getdata', 'getmask', 'getmaskarray', 'greater', 'greater_equal', 'harden_mask', 'hypot', 'identity', 'ids', 'indices', 'inner', 'innerproduct', 'isMA', 'isMaskedArray', 'is_mask', 'is_masked', 'isarray', 'left_shift', 'less', 'less_equal', 'log', 'log10', 'log2', 'logical_and', 'logical_not', 'logical_or', 'logical_xor', 'make_mask', 'make_mask_descr', 'make_mask_none', 'mask_or', 'masked', 'masked_array', 'masked_equal', 'masked_greater', 'masked_greater_equal', 'masked_inside', 'masked_invalid', 'masked_less', 'masked_less_equal', 'masked_not_equal', 'masked_object', 'masked_outside', 'masked_print_option', 'masked_singleton', 'masked_values', 'masked_where', 'max', 'maximum', 'maximum_fill_value', 'mean', 'min', 'minimum', 'minimum_fill_value', 'mod', 'multiply', 'mvoid', 'ndim', 'negative', 'nomask', 'nonzero', 'not_equal', 'ones', 'ones_like', 'outer', 'outerproduct', 'power', 'prod', 'product', 'ptp', 'put', 'putmask', 'ravel', 'remainder', 'repeat', 'reshape', 'resize', 'right_shift', 'round', 'round_', 'set_fill_value', 'shape', 'sin', 'sinh', 'size', 'soften_mask', 'sometrue', 'sort', 'sqrt', 'squeeze', 'std', 'subtract', 'sum', 'swapaxes', 'take', 'tan', 'tanh', 'trace', 'transpose', 'true_divide', 'var', 'where', 'zeros', 'zeros_like']
class MAError(Exception):
    """
    
        Class for masked array related errors.
    
        
    """
class MaskError(MAError):
    """
    
        Class for mask related errors.
    
        
    """
class MaskedArray(numpy.ndarray):
    """
    
        An array class with possibly masked values.
    
        Masked values of True exclude the corresponding element from any
        computation.
    
        Construction::
    
          x = MaskedArray(data, mask=nomask, dtype=None, copy=False, subok=True,
                          ndmin=0, fill_value=None, keep_mask=True, hard_mask=None,
                          shrink=True, order=None)
    
        Parameters
        ----------
        data : array_like
            Input data.
        mask : sequence, optional
            Mask. Must be convertible to an array of booleans with the same
            shape as `data`. True indicates a masked (i.e. invalid) data.
        dtype : dtype, optional
            Data type of the output.
            If `dtype` is None, the type of the data argument (``data.dtype``)
            is used. If `dtype` is not None and different from ``data.dtype``,
            a copy is performed.
        copy : bool, optional
            Whether to copy the input data (True), or to use a reference instead.
            Default is False.
        subok : bool, optional
            Whether to return a subclass of `MaskedArray` if possible (True) or a
            plain `MaskedArray`. Default is True.
        ndmin : int, optional
            Minimum number of dimensions. Default is 0.
        fill_value : scalar, optional
            Value used to fill in the masked values when necessary.
            If None, a default based on the data-type is used.
        keep_mask : bool, optional
            Whether to combine `mask` with the mask of the input data, if any
            (True), or to use only `mask` for the output (False). Default is True.
        hard_mask : bool, optional
            Whether to use a hard mask or not. With a hard mask, masked values
            cannot be unmasked. Default is False.
        shrink : bool, optional
            Whether to force compression of an empty mask. Default is True.
        order : {'C', 'F', 'A'}, optional
            Specify the order of the array.  If order is 'C', then the array
            will be in C-contiguous order (last-index varies the fastest).
            If order is 'F', then the returned array will be in
            Fortran-contiguous order (first-index varies the fastest).
            If order is 'A' (default), then the returned array may be
            in any order (either C-, Fortran-contiguous, or even discontiguous),
            unless a copy is required, in which case it will be C-contiguous.
    
        Examples
        --------
    
        The ``mask`` can be initialized with an array of boolean values
        with the same shape as ``data``.
    
        >>> data = np.arange(6).reshape((2, 3))
        >>> np.ma.MaskedArray(data, mask=[[False, True, False],
        ...                               [False, False, True]])
        masked_array(
          data=[[0, --, 2],
                [3, 4, --]],
          mask=[[False,  True, False],
                [False, False,  True]],
          fill_value=999999)
    
        Alternatively, the ``mask`` can be initialized to homogeneous boolean
        array with the same shape as ``data`` by passing in a scalar
        boolean value:
    
        >>> np.ma.MaskedArray(data, mask=False)
        masked_array(
          data=[[0, 1, 2],
                [3, 4, 5]],
          mask=[[False, False, False],
                [False, False, False]],
          fill_value=999999)
    
        >>> np.ma.MaskedArray(data, mask=True)
        masked_array(
          data=[[--, --, --],
                [--, --, --]],
          mask=[[ True,  True,  True],
                [ True,  True,  True]],
          fill_value=999999,
          dtype=int64)
    
        .. note::
            The recommended practice for initializing ``mask`` with a scalar
            boolean value is to use ``True``/``False`` rather than
            ``np.True_``/``np.False_``. The reason is :attr:`nomask`
            is represented internally as ``np.False_``.
    
            >>> np.False_ is np.ma.nomask
            True
    
        
    """
    __array_priority__: typing.ClassVar[int] = 15
    __hash__: typing.ClassVar[None] = None
    _defaulthardmask: typing.ClassVar[bool] = False
    _defaultmask: typing.ClassVar[numpy.bool_]  # value = False
    _print_width: typing.ClassVar[int] = 100
    _print_width_1d: typing.ClassVar[int] = 1500
    dtype = ...
    shape = ...
    _baseclass = numpy.ndarray
    @staticmethod
    def __setitem__(*args, **kwds):
        """
        
                x.__setitem__(i, y) <==> x[i]=y
        
                Set item described by index. If value is masked, masks those
                locations.
        
                
        """
    @classmethod
    def __new__(cls, data = None, mask = ..., dtype = None, copy = False, subok = True, ndmin = 0, fill_value = None, keep_mask = True, hard_mask = None, shrink = True, order = None):
        """
        
                Create a new masked array from scratch.
        
                Notes
                -----
                A masked array can also be created by taking a .view(MaskedArray).
        
                
        """
    def __add__(self, other):
        """
        
                Add self to other, and return a new masked array.
        
                
        """
    def __array_finalize__(self, obj):
        """
        
                Finalizes the masked array.
        
                
        """
    def __array_wrap__(self, obj, context = None):
        """
        
                Special hook for ufuncs.
        
                Wraps the numpy array and sets the mask according to context.
        
                
        """
    def __deepcopy__(self, memo = None):
        ...
    def __div__(self, other):
        """
        
                Divide other into self, and return a new masked array.
        
                
        """
    def __eq__(self, other):
        """
        Check whether other equals self elementwise.
        
                When either of the elements is masked, the result is masked as well,
                but the underlying boolean data are still set, with self and other
                considered equal if both are masked, and unequal otherwise.
        
                For structured arrays, all fields are combined, with masked values
                ignored. The result is masked if all fields were masked, with self
                and other considered equal only if both were fully masked.
                
        """
    def __float__(self):
        """
        
                Convert to float.
        
                
        """
    def __floordiv__(self, other):
        """
        
                Divide other into self, and return a new masked array.
        
                
        """
    def __ge__(self, other):
        ...
    def __getitem__(self, indx):
        """
        
                x.__getitem__(y) <==> x[y]
        
                Return the item described by i, as a masked array.
        
                
        """
    def __getstate__(self):
        """
        Return the internal state of the masked array, for pickling
                purposes.
        
                
        """
    def __gt__(self, other):
        ...
    def __iadd__(self, other):
        """
        
                Add other to self in-place.
        
                
        """
    def __idiv__(self, other):
        """
        
                Divide self by other in-place.
        
                
        """
    def __ifloordiv__(self, other):
        """
        
                Floor divide self by other in-place.
        
                
        """
    def __imul__(self, other):
        """
        
                Multiply self by other in-place.
        
                
        """
    def __int__(self):
        """
        
                Convert to int.
        
                
        """
    def __ipow__(self, other):
        """
        
                Raise self to the power other, in place.
        
                
        """
    def __isub__(self, other):
        """
        
                Subtract other from self in-place.
        
                
        """
    def __itruediv__(self, other):
        """
        
                True divide self by other in-place.
        
                
        """
    def __le__(self, other):
        ...
    def __lt__(self, other):
        ...
    def __mul__(self, other):
        """
        Multiply self by other, and return a new masked array.
        """
    def __ne__(self, other):
        """
        Check whether other does not equal self elementwise.
        
                When either of the elements is masked, the result is masked as well,
                but the underlying boolean data are still set, with self and other
                considered equal if both are masked, and unequal otherwise.
        
                For structured arrays, all fields are combined, with masked values
                ignored. The result is masked if all fields were masked, with self
                and other considered equal only if both were fully masked.
                
        """
    def __pow__(self, other):
        """
        
                Raise self to the power other, masking the potential NaNs/Infs
        
                
        """
    def __radd__(self, other):
        """
        
                Add other to self, and return a new masked array.
        
                
        """
    def __reduce__(self):
        """
        Return a 3-tuple for pickling a MaskedArray.
        
                
        """
    def __repr__(self):
        """
        
                Literal string representation.
        
                
        """
    def __rfloordiv__(self, other):
        """
        
                Divide self into other, and return a new masked array.
        
                
        """
    def __rmul__(self, other):
        """
        
                Multiply other by self, and return a new masked array.
        
                
        """
    def __rpow__(self, other):
        """
        
                Raise other to the power self, masking the potential NaNs/Infs
        
                
        """
    def __rsub__(self, other):
        """
        
                Subtract self from other, and return a new masked array.
        
                
        """
    def __rtruediv__(self, other):
        """
        
                Divide self into other, and return a new masked array.
        
                
        """
    def __setmask__(self, mask, copy = False):
        """
        
                Set the mask.
        
                
        """
    def __setstate__(self, state):
        """
        Restore the internal state of the masked array, for
                pickling purposes.  ``state`` is typically the output of the
                ``__getstate__`` output, and is a 5-tuple:
        
                - class name
                - a tuple giving the shape of the data
                - a typecode for the data
                - a binary string for the data
                - a binary string for the mask.
        
                
        """
    def __str__(self):
        ...
    def __sub__(self, other):
        """
        
                Subtract other from self, and return a new masked array.
        
                
        """
    def __truediv__(self, other):
        """
        
                Divide other into self, and return a new masked array.
        
                
        """
    def _comparison(self, other, compare):
        """
        Compare self with other using operator.eq or operator.ne.
        
                When either of the elements is masked, the result is masked as well,
                but the underlying boolean data are still set, with self and other
                considered equal if both are masked, and unequal otherwise.
        
                For structured arrays, all fields are combined, with masked values
                ignored. The result is masked if all fields were masked, with self
                and other considered equal only if both were fully masked.
                
        """
    def _delegate_binop(self, other):
        ...
    def _get_data(self):
        """
        
                Returns the underlying data, as a view of the masked array.
        
                If the underlying data is a subclass of :class:`numpy.ndarray`, it is
                returned as such.
        
                >>> x = np.ma.array(np.matrix([[1, 2], [3, 4]]), mask=[[0, 1], [1, 0]])
                >>> x.data
                matrix([[1, 2],
                        [3, 4]])
        
                The type of the data can be accessed through the :attr:`baseclass`
                attribute.
                
        """
    def _insert_masked_print(self):
        """
        
                Replace masked values with masked_print_option, casting all innermost
                dtypes to object.
                
        """
    def _set_mask(self, mask, copy = False):
        """
        
                Set the mask.
        
                
        """
    def _update_from(self, obj):
        """
        
                Copies some attributes of obj to self.
        
                
        """
    def all(self, axis = None, out = None, keepdims = ...):
        """
        
                Returns True if all elements evaluate to True.
        
                The output array is masked where all the values along the given axis
                are masked: if the output would have been a scalar and that all the
                values are masked, then the output is `masked`.
        
                Refer to `numpy.all` for full documentation.
        
                See Also
                --------
                numpy.ndarray.all : corresponding function for ndarrays
                numpy.all : equivalent function
        
                Examples
                --------
                >>> np.ma.array([1,2,3]).all()
                True
                >>> a = np.ma.array([1,2,3], mask=True)
                >>> (a.all() is np.ma.masked)
                True
        
                
        """
    def anom(self, axis = None, dtype = None):
        """
        
                Compute the anomalies (deviations from the arithmetic mean)
                along the given axis.
        
                Returns an array of anomalies, with the same shape as the input and
                where the arithmetic mean is computed along the given axis.
        
                Parameters
                ----------
                axis : int, optional
                    Axis over which the anomalies are taken.
                    The default is to use the mean of the flattened array as reference.
                dtype : dtype, optional
                    Type to use in computing the variance. For arrays of integer type
                     the default is float32; for arrays of float types it is the same as
                     the array type.
        
                See Also
                --------
                mean : Compute the mean of the array.
        
                Examples
                --------
                >>> a = np.ma.array([1,2,3])
                >>> a.anom()
                masked_array(data=[-1.,  0.,  1.],
                             mask=False,
                       fill_value=1e+20)
        
                
        """
    def any(self, axis = None, out = None, keepdims = ...):
        """
        
                Returns True if any of the elements of `a` evaluate to True.
        
                Masked values are considered as False during computation.
        
                Refer to `numpy.any` for full documentation.
        
                See Also
                --------
                numpy.ndarray.any : corresponding function for ndarrays
                numpy.any : equivalent function
        
                
        """
    def argmax(self, axis = None, fill_value = None, out = None, *, keepdims = ...):
        """
        
                Returns array of indices of the maximum values along the given axis.
                Masked values are treated as if they had the value fill_value.
        
                Parameters
                ----------
                axis : {None, integer}
                    If None, the index is into the flattened array, otherwise along
                    the specified axis
                fill_value : scalar or None, optional
                    Value used to fill in the masked values.  If None, the output of
                    maximum_fill_value(self._data) is used instead.
                out : {None, array}, optional
                    Array into which the result can be placed. Its type is preserved
                    and it must be of the right shape to hold the output.
        
                Returns
                -------
                index_array : {integer_array}
        
                Examples
                --------
                >>> a = np.arange(6).reshape(2,3)
                >>> a.argmax()
                5
                >>> a.argmax(0)
                array([1, 1, 1])
                >>> a.argmax(1)
                array([2, 2])
        
                
        """
    def argmin(self, axis = None, fill_value = None, out = None, *, keepdims = ...):
        """
        
                Return array of indices to the minimum values along the given axis.
        
                Parameters
                ----------
                axis : {None, integer}
                    If None, the index is into the flattened array, otherwise along
                    the specified axis
                fill_value : scalar or None, optional
                    Value used to fill in the masked values.  If None, the output of
                    minimum_fill_value(self._data) is used instead.
                out : {None, array}, optional
                    Array into which the result can be placed. Its type is preserved
                    and it must be of the right shape to hold the output.
        
                Returns
                -------
                ndarray or scalar
                    If multi-dimension input, returns a new ndarray of indices to the
                    minimum values along the given axis.  Otherwise, returns a scalar
                    of index to the minimum values along the given axis.
        
                Examples
                --------
                >>> x = np.ma.array(np.arange(4), mask=[1,1,0,0])
                >>> x.shape = (2,2)
                >>> x
                masked_array(
                  data=[[--, --],
                        [2, 3]],
                  mask=[[ True,  True],
                        [False, False]],
                  fill_value=999999)
                >>> x.argmin(axis=0, fill_value=-1)
                array([0, 0])
                >>> x.argmin(axis=0, fill_value=9)
                array([1, 1])
        
                
        """
    def argpartition(self, *args, **kwargs):
        ...
    def argsort(self, axis = ..., kind = None, order = None, endwith = True, fill_value = None):
        """
        
                Return an ndarray of indices that sort the array along the
                specified axis.  Masked values are filled beforehand to
                `fill_value`.
        
                Parameters
                ----------
                axis : int, optional
                    Axis along which to sort. If None, the default, the flattened array
                    is used.
        
                    ..  versionchanged:: 1.13.0
                        Previously, the default was documented to be -1, but that was
                        in error. At some future date, the default will change to -1, as
                        originally intended.
                        Until then, the axis should be given explicitly when
                        ``arr.ndim > 1``, to avoid a FutureWarning.
                kind : {'quicksort', 'mergesort', 'heapsort', 'stable'}, optional
                    The sorting algorithm used.
                order : list, optional
                    When `a` is an array with fields defined, this argument specifies
                    which fields to compare first, second, etc.  Not all fields need be
                    specified.
                endwith : {True, False}, optional
                    Whether missing values (if any) should be treated as the largest values
                    (True) or the smallest values (False)
                    When the array contains unmasked values at the same extremes of the
                    datatype, the ordering of these values and the masked values is
                    undefined.
                fill_value : scalar or None, optional
                    Value used internally for the masked values.
                    If ``fill_value`` is not None, it supersedes ``endwith``.
        
                Returns
                -------
                index_array : ndarray, int
                    Array of indices that sort `a` along the specified axis.
                    In other words, ``a[index_array]`` yields a sorted `a`.
        
                See Also
                --------
                ma.MaskedArray.sort : Describes sorting algorithms used.
                lexsort : Indirect stable sort with multiple keys.
                numpy.ndarray.sort : Inplace sort.
        
                Notes
                -----
                See `sort` for notes on the different sorting algorithms.
        
                Examples
                --------
                >>> a = np.ma.array([3,2,1], mask=[False, False, True])
                >>> a
                masked_array(data=[3, 2, --],
                             mask=[False, False,  True],
                       fill_value=999999)
                >>> a.argsort()
                array([1, 0, 2])
        
                
        """
    def compress(self, condition, axis = None, out = None):
        """
        
                Return `a` where condition is ``True``.
        
                If condition is a `~ma.MaskedArray`, missing values are considered
                as ``False``.
        
                Parameters
                ----------
                condition : var
                    Boolean 1-d array selecting which entries to return. If len(condition)
                    is less than the size of a along the axis, then output is truncated
                    to length of condition array.
                axis : {None, int}, optional
                    Axis along which the operation must be performed.
                out : {None, ndarray}, optional
                    Alternative output array in which to place the result. It must have
                    the same shape as the expected output but the type will be cast if
                    necessary.
        
                Returns
                -------
                result : MaskedArray
                    A :class:`~ma.MaskedArray` object.
        
                Notes
                -----
                Please note the difference with :meth:`compressed` !
                The output of :meth:`compress` has a mask, the output of
                :meth:`compressed` does not.
        
                Examples
                --------
                >>> x = np.ma.array([[1,2,3],[4,5,6],[7,8,9]], mask=[0] + [1,0]*4)
                >>> x
                masked_array(
                  data=[[1, --, 3],
                        [--, 5, --],
                        [7, --, 9]],
                  mask=[[False,  True, False],
                        [ True, False,  True],
                        [False,  True, False]],
                  fill_value=999999)
                >>> x.compress([1, 0, 1])
                masked_array(data=[1, 3],
                             mask=[False, False],
                       fill_value=999999)
        
                >>> x.compress([1, 0, 1], axis=1)
                masked_array(
                  data=[[1, 3],
                        [--, --],
                        [7, 9]],
                  mask=[[False, False],
                        [ True,  True],
                        [False, False]],
                  fill_value=999999)
        
                
        """
    def compressed(self):
        """
        
                Return all the non-masked data as a 1-D array.
        
                Returns
                -------
                data : ndarray
                    A new `ndarray` holding the non-masked data is returned.
        
                Notes
                -----
                The result is **not** a MaskedArray!
        
                Examples
                --------
                >>> x = np.ma.array(np.arange(5), mask=[0]*2 + [1]*3)
                >>> x.compressed()
                array([0, 1])
                >>> type(x.compressed())
                <class 'numpy.ndarray'>
        
                
        """
    def copy(self, *args, **params):
        """
        a.copy(order='C')
        
            Return a copy of the array.
        
            Parameters
            ----------
            order : {'C', 'F', 'A', 'K'}, optional
                Controls the memory layout of the copy. 'C' means C-order,
                'F' means F-order, 'A' means 'F' if `a` is Fortran contiguous,
                'C' otherwise. 'K' means match the layout of `a` as closely
                as possible. (Note that this function and :func:`numpy.copy` are very
                similar but have different default values for their order=
                arguments, and this function always passes sub-classes through.)
        
            See also
            --------
            numpy.copy : Similar function with different default behavior
            numpy.copyto
        
            Notes
            -----
            This function is the preferred method for creating an array copy.  The
            function :func:`numpy.copy` is similar, but it defaults to using order 'K',
            and will not pass sub-classes through by default.
        
            Examples
            --------
            >>> x = np.array([[1,2,3],[4,5,6]], order='F')
        
            >>> y = x.copy()
        
            >>> x.fill(0)
        
            >>> x
            array([[0, 0, 0],
                   [0, 0, 0]])
        
            >>> y
            array([[1, 2, 3],
                   [4, 5, 6]])
        
            >>> y.flags['C_CONTIGUOUS']
            True
        """
    def count(self, axis = None, keepdims = ...):
        """
        
                Count the non-masked elements of the array along the given axis.
        
                Parameters
                ----------
                axis : None or int or tuple of ints, optional
                    Axis or axes along which the count is performed.
                    The default, None, performs the count over all
                    the dimensions of the input array. `axis` may be negative, in
                    which case it counts from the last to the first axis.
        
                    .. versionadded:: 1.10.0
        
                    If this is a tuple of ints, the count is performed on multiple
                    axes, instead of a single axis or all the axes as before.
                keepdims : bool, optional
                    If this is set to True, the axes which are reduced are left
                    in the result as dimensions with size one. With this option,
                    the result will broadcast correctly against the array.
        
                Returns
                -------
                result : ndarray or scalar
                    An array with the same shape as the input array, with the specified
                    axis removed. If the array is a 0-d array, or if `axis` is None, a
                    scalar is returned.
        
                See Also
                --------
                ma.count_masked : Count masked elements in array or along a given axis.
        
                Examples
                --------
                >>> import numpy.ma as ma
                >>> a = ma.arange(6).reshape((2, 3))
                >>> a[1, :] = ma.masked
                >>> a
                masked_array(
                  data=[[0, 1, 2],
                        [--, --, --]],
                  mask=[[False, False, False],
                        [ True,  True,  True]],
                  fill_value=999999)
                >>> a.count()
                3
        
                When the `axis` keyword is specified an array of appropriate size is
                returned.
        
                >>> a.count(axis=0)
                array([1, 1, 1])
                >>> a.count(axis=1)
                array([3, 0])
        
                
        """
    def cumprod(self, axis = None, dtype = None, out = None):
        """
        
                Return the cumulative product of the array elements over the given axis.
        
                Masked values are set to 1 internally during the computation.
                However, their position is saved, and the result will be masked at
                the same locations.
        
                Refer to `numpy.cumprod` for full documentation.
        
                Notes
                -----
                The mask is lost if `out` is not a valid MaskedArray !
        
                Arithmetic is modular when using integer types, and no error is
                raised on overflow.
        
                See Also
                --------
                numpy.ndarray.cumprod : corresponding function for ndarrays
                numpy.cumprod : equivalent function
                
        """
    def cumsum(self, axis = None, dtype = None, out = None):
        """
        
                Return the cumulative sum of the array elements over the given axis.
        
                Masked values are set to 0 internally during the computation.
                However, their position is saved, and the result will be masked at
                the same locations.
        
                Refer to `numpy.cumsum` for full documentation.
        
                Notes
                -----
                The mask is lost if `out` is not a valid :class:`ma.MaskedArray` !
        
                Arithmetic is modular when using integer types, and no error is
                raised on overflow.
        
                See Also
                --------
                numpy.ndarray.cumsum : corresponding function for ndarrays
                numpy.cumsum : equivalent function
        
                Examples
                --------
                >>> marr = np.ma.array(np.arange(10), mask=[0,0,0,1,1,1,0,0,0,0])
                >>> marr.cumsum()
                masked_array(data=[0, 1, 3, --, --, --, 9, 16, 24, 33],
                             mask=[False, False, False,  True,  True,  True, False, False,
                                   False, False],
                       fill_value=999999)
        
                
        """
    def diagonal(self, *args, **params):
        """
        a.diagonal(offset=0, axis1=0, axis2=1)
        
            Return specified diagonals. In NumPy 1.9 the returned array is a
            read-only view instead of a copy as in previous NumPy versions.  In
            a future version the read-only restriction will be removed.
        
            Refer to :func:`numpy.diagonal` for full documentation.
        
            See Also
            --------
            numpy.diagonal : equivalent function
        """
    def dot(self, b, out = None, strict = False):
        """
        
                a.dot(b, out=None)
        
                Masked dot product of two arrays. Note that `out` and `strict` are
                located in different positions than in `ma.dot`. In order to
                maintain compatibility with the functional version, it is
                recommended that the optional arguments be treated as keyword only.
                At some point that may be mandatory.
        
                .. versionadded:: 1.10.0
        
                Parameters
                ----------
                b : masked_array_like
                    Inputs array.
                out : masked_array, optional
                    Output argument. This must have the exact kind that would be
                    returned if it was not used. In particular, it must have the
                    right type, must be C-contiguous, and its dtype must be the
                    dtype that would be returned for `ma.dot(a,b)`. This is a
                    performance feature. Therefore, if these conditions are not
                    met, an exception is raised, instead of attempting to be
                    flexible.
                strict : bool, optional
                    Whether masked data are propagated (True) or set to 0 (False)
                    for the computation. Default is False.  Propagating the mask
                    means that if a masked value appears in a row or column, the
                    whole row or column is considered masked.
        
                    .. versionadded:: 1.10.2
        
                See Also
                --------
                numpy.ma.dot : equivalent function
        
                
        """
    def filled(self, fill_value = None):
        """
        
                Return a copy of self, with masked values filled with a given value.
                **However**, if there are no masked values to fill, self will be
                returned instead as an ndarray.
        
                Parameters
                ----------
                fill_value : array_like, optional
                    The value to use for invalid entries. Can be scalar or non-scalar.
                    If non-scalar, the resulting ndarray must be broadcastable over
                    input array. Default is None, in which case, the `fill_value`
                    attribute of the array is used instead.
        
                Returns
                -------
                filled_array : ndarray
                    A copy of ``self`` with invalid entries replaced by *fill_value*
                    (be it the function argument or the attribute of ``self``), or
                    ``self`` itself as an ndarray if there are no invalid entries to
                    be replaced.
        
                Notes
                -----
                The result is **not** a MaskedArray!
        
                Examples
                --------
                >>> x = np.ma.array([1,2,3,4,5], mask=[0,0,1,0,1], fill_value=-999)
                >>> x.filled()
                array([   1,    2, -999,    4, -999])
                >>> x.filled(fill_value=1000)
                array([   1,    2, 1000,    4, 1000])
                >>> type(x.filled())
                <class 'numpy.ndarray'>
        
                Subclassing is preserved. This means that if, e.g., the data part of
                the masked array is a recarray, `filled` returns a recarray:
        
                >>> x = np.array([(-1, 2), (-3, 4)], dtype='i8,i8').view(np.recarray)
                >>> m = np.ma.array(x, mask=[(True, False), (False, True)])
                >>> m.filled()
                rec.array([(999999,      2), (    -3, 999999)],
                          dtype=[('f0', '<i8'), ('f1', '<i8')])
                
        """
    def flatten(self, *args, **params):
        """
        a.flatten(order='C')
        
            Return a copy of the array collapsed into one dimension.
        
            Parameters
            ----------
            order : {'C', 'F', 'A', 'K'}, optional
                'C' means to flatten in row-major (C-style) order.
                'F' means to flatten in column-major (Fortran-
                style) order. 'A' means to flatten in column-major
                order if `a` is Fortran *contiguous* in memory,
                row-major order otherwise. 'K' means to flatten
                `a` in the order the elements occur in memory.
                The default is 'C'.
        
            Returns
            -------
            y : ndarray
                A copy of the input array, flattened to one dimension.
        
            See Also
            --------
            ravel : Return a flattened array.
            flat : A 1-D flat iterator over the array.
        
            Examples
            --------
            >>> a = np.array([[1,2], [3,4]])
            >>> a.flatten()
            array([1, 2, 3, 4])
            >>> a.flatten('F')
            array([1, 3, 2, 4])
        """
    def get_fill_value(self):
        """
        
                The filling value of the masked array is a scalar. When setting, None
                will set to a default based on the data type.
        
                Examples
                --------
                >>> for dt in [np.int32, np.int64, np.float64, np.complex128]:
                ...     np.ma.array([0, 1], dtype=dt).get_fill_value()
                ...
                999999
                999999
                1e+20
                (1e+20+0j)
        
                >>> x = np.ma.array([0, 1.], fill_value=-np.inf)
                >>> x.fill_value
                -inf
                >>> x.fill_value = np.pi
                >>> x.fill_value
                3.1415926535897931 # may vary
        
                Reset to default:
        
                >>> x.fill_value = None
                >>> x.fill_value
                1e+20
        
                
        """
    def get_imag(self):
        """
        
                The imaginary part of the masked array.
        
                This property is a view on the imaginary part of this `MaskedArray`.
        
                See Also
                --------
                real
        
                Examples
                --------
                >>> x = np.ma.array([1+1.j, -2j, 3.45+1.6j], mask=[False, True, False])
                >>> x.imag
                masked_array(data=[1.0, --, 1.6],
                             mask=[False,  True, False],
                       fill_value=1e+20)
        
                
        """
    def get_real(self):
        """
        
                The real part of the masked array.
        
                This property is a view on the real part of this `MaskedArray`.
        
                See Also
                --------
                imag
        
                Examples
                --------
                >>> x = np.ma.array([1+1.j, -2j, 3.45+1.6j], mask=[False, True, False])
                >>> x.real
                masked_array(data=[1.0, --, 3.45],
                             mask=[False,  True, False],
                       fill_value=1e+20)
        
                
        """
    def harden_mask(self):
        """
        
                Force the mask to hard, preventing unmasking by assignment.
        
                Whether the mask of a masked array is hard or soft is determined by
                its `~ma.MaskedArray.hardmask` property. `harden_mask` sets
                `~ma.MaskedArray.hardmask` to ``True`` (and returns the modified
                self).
        
                See Also
                --------
                ma.MaskedArray.hardmask
                ma.MaskedArray.soften_mask
        
                
        """
    def ids(self):
        """
        
                Return the addresses of the data and mask areas.
        
                Parameters
                ----------
                None
        
                Examples
                --------
                >>> x = np.ma.array([1, 2, 3], mask=[0, 1, 1])
                >>> x.ids()
                (166670640, 166659832) # may vary
        
                If the array has no mask, the address of `nomask` is returned. This address
                is typically not close to the data in memory:
        
                >>> x = np.ma.array([1, 2, 3])
                >>> x.ids()
                (166691080, 3083169284) # may vary
        
                
        """
    def iscontiguous(self):
        """
        
                Return a boolean indicating whether the data is contiguous.
        
                Parameters
                ----------
                None
        
                Examples
                --------
                >>> x = np.ma.array([1, 2, 3])
                >>> x.iscontiguous()
                True
        
                `iscontiguous` returns one of the flags of the masked array:
        
                >>> x.flags
                  C_CONTIGUOUS : True
                  F_CONTIGUOUS : True
                  OWNDATA : False
                  WRITEABLE : True
                  ALIGNED : True
                  WRITEBACKIFCOPY : False
        
                
        """
    def max(self, axis = None, out = None, fill_value = None, keepdims = ...):
        """
        
                Return the maximum along a given axis.
        
                Parameters
                ----------
                axis : None or int or tuple of ints, optional
                    Axis along which to operate.  By default, ``axis`` is None and the
                    flattened input is used.
                    .. versionadded:: 1.7.0
                    If this is a tuple of ints, the maximum is selected over multiple
                    axes, instead of a single axis or all the axes as before.
                out : array_like, optional
                    Alternative output array in which to place the result.  Must
                    be of the same shape and buffer length as the expected output.
                fill_value : scalar or None, optional
                    Value used to fill in the masked values.
                    If None, use the output of maximum_fill_value().
                keepdims : bool, optional
                    If this is set to True, the axes which are reduced are left
                    in the result as dimensions with size one. With this option,
                    the result will broadcast correctly against the array.
        
                Returns
                -------
                amax : array_like
                    New array holding the result.
                    If ``out`` was specified, ``out`` is returned.
        
                See Also
                --------
                ma.maximum_fill_value
                    Returns the maximum filling value for a given datatype.
        
                Examples
                --------
                >>> import numpy.ma as ma
                >>> x = [[-1., 2.5], [4., -2.], [3., 0.]]
                >>> mask = [[0, 0], [1, 0], [1, 0]]
                >>> masked_x = ma.masked_array(x, mask)
                >>> masked_x
                masked_array(
                  data=[[-1.0, 2.5],
                        [--, -2.0],
                        [--, 0.0]],
                  mask=[[False, False],
                        [ True, False],
                        [ True, False]],
                  fill_value=1e+20)
                >>> ma.max(masked_x)
                2.5
                >>> ma.max(masked_x, axis=0)
                masked_array(data=[-1.0, 2.5],
                             mask=[False, False],
                       fill_value=1e+20)
                >>> ma.max(masked_x, axis=1, keepdims=True)
                masked_array(
                  data=[[2.5],
                        [-2.0],
                        [0.0]],
                  mask=[[False],
                        [False],
                        [False]],
                  fill_value=1e+20)
                >>> mask = [[1, 1], [1, 1], [1, 1]]
                >>> masked_x = ma.masked_array(x, mask)
                >>> ma.max(masked_x, axis=1)
                masked_array(data=[--, --, --],
                             mask=[ True,  True,  True],
                       fill_value=1e+20,
                            dtype=float64)
                
        """
    def mean(self, axis = None, dtype = None, out = None, keepdims = ...):
        """
        
                Returns the average of the array elements along given axis.
        
                Masked entries are ignored, and result elements which are not
                finite will be masked.
        
                Refer to `numpy.mean` for full documentation.
        
                See Also
                --------
                numpy.ndarray.mean : corresponding function for ndarrays
                numpy.mean : Equivalent function
                numpy.ma.average : Weighted average.
        
                Examples
                --------
                >>> a = np.ma.array([1,2,3], mask=[False, False, True])
                >>> a
                masked_array(data=[1, 2, --],
                             mask=[False, False,  True],
                       fill_value=999999)
                >>> a.mean()
                1.5
        
                
        """
    def min(self, axis = None, out = None, fill_value = None, keepdims = ...):
        """
        
                Return the minimum along a given axis.
        
                Parameters
                ----------
                axis : None or int or tuple of ints, optional
                    Axis along which to operate.  By default, ``axis`` is None and the
                    flattened input is used.
                    .. versionadded:: 1.7.0
                    If this is a tuple of ints, the minimum is selected over multiple
                    axes, instead of a single axis or all the axes as before.
                out : array_like, optional
                    Alternative output array in which to place the result.  Must be of
                    the same shape and buffer length as the expected output.
                fill_value : scalar or None, optional
                    Value used to fill in the masked values.
                    If None, use the output of `minimum_fill_value`.
                keepdims : bool, optional
                    If this is set to True, the axes which are reduced are left
                    in the result as dimensions with size one. With this option,
                    the result will broadcast correctly against the array.
        
                Returns
                -------
                amin : array_like
                    New array holding the result.
                    If ``out`` was specified, ``out`` is returned.
        
                See Also
                --------
                ma.minimum_fill_value
                    Returns the minimum filling value for a given datatype.
        
                Examples
                --------
                >>> import numpy.ma as ma
                >>> x = [[1., -2., 3.], [0.2, -0.7, 0.1]]
                >>> mask = [[1, 1, 0], [0, 0, 1]]
                >>> masked_x = ma.masked_array(x, mask)
                >>> masked_x
                masked_array(
                  data=[[--, --, 3.0],
                        [0.2, -0.7, --]],
                  mask=[[ True,  True, False],
                        [False, False,  True]],
                  fill_value=1e+20)
                >>> ma.min(masked_x)
                -0.7
                >>> ma.min(masked_x, axis=-1)
                masked_array(data=[3.0, -0.7],
                             mask=[False, False],
                        fill_value=1e+20)
                >>> ma.min(masked_x, axis=0, keepdims=True)
                masked_array(data=[[0.2, -0.7, 3.0]],
                             mask=[[False, False, False]],
                        fill_value=1e+20)
                >>> mask = [[1, 1, 1,], [1, 1, 1]]
                >>> masked_x = ma.masked_array(x, mask)
                >>> ma.min(masked_x, axis=0)
                masked_array(data=[--, --, --],
                             mask=[ True,  True,  True],
                        fill_value=1e+20,
                            dtype=float64)
                
        """
    def nonzero(self):
        """
        
                Return the indices of unmasked elements that are not zero.
        
                Returns a tuple of arrays, one for each dimension, containing the
                indices of the non-zero elements in that dimension. The corresponding
                non-zero values can be obtained with::
        
                    a[a.nonzero()]
        
                To group the indices by element, rather than dimension, use
                instead::
        
                    np.transpose(a.nonzero())
        
                The result of this is always a 2d array, with a row for each non-zero
                element.
        
                Parameters
                ----------
                None
        
                Returns
                -------
                tuple_of_arrays : tuple
                    Indices of elements that are non-zero.
        
                See Also
                --------
                numpy.nonzero :
                    Function operating on ndarrays.
                flatnonzero :
                    Return indices that are non-zero in the flattened version of the input
                    array.
                numpy.ndarray.nonzero :
                    Equivalent ndarray method.
                count_nonzero :
                    Counts the number of non-zero elements in the input array.
        
                Examples
                --------
                >>> import numpy.ma as ma
                >>> x = ma.array(np.eye(3))
                >>> x
                masked_array(
                  data=[[1., 0., 0.],
                        [0., 1., 0.],
                        [0., 0., 1.]],
                  mask=False,
                  fill_value=1e+20)
                >>> x.nonzero()
                (array([0, 1, 2]), array([0, 1, 2]))
        
                Masked elements are ignored.
        
                >>> x[1, 1] = ma.masked
                >>> x
                masked_array(
                  data=[[1.0, 0.0, 0.0],
                        [0.0, --, 0.0],
                        [0.0, 0.0, 1.0]],
                  mask=[[False, False, False],
                        [False,  True, False],
                        [False, False, False]],
                  fill_value=1e+20)
                >>> x.nonzero()
                (array([0, 2]), array([0, 2]))
        
                Indices can also be grouped by element.
        
                >>> np.transpose(x.nonzero())
                array([[0, 0],
                       [2, 2]])
        
                A common use for ``nonzero`` is to find the indices of an array, where
                a condition is True.  Given an array `a`, the condition `a` > 3 is a
                boolean array and since False is interpreted as 0, ma.nonzero(a > 3)
                yields the indices of the `a` where the condition is true.
        
                >>> a = ma.array([[1,2,3],[4,5,6],[7,8,9]])
                >>> a > 3
                masked_array(
                  data=[[False, False, False],
                        [ True,  True,  True],
                        [ True,  True,  True]],
                  mask=False,
                  fill_value=True)
                >>> ma.nonzero(a > 3)
                (array([1, 1, 1, 2, 2, 2]), array([0, 1, 2, 0, 1, 2]))
        
                The ``nonzero`` method of the condition array can also be called.
        
                >>> (a > 3).nonzero()
                (array([1, 1, 1, 2, 2, 2]), array([0, 1, 2, 0, 1, 2]))
        
                
        """
    def partition(self, *args, **kwargs):
        ...
    def prod(self, axis = None, dtype = None, out = None, keepdims = ...):
        """
        
                Return the product of the array elements over the given axis.
        
                Masked elements are set to 1 internally for computation.
        
                Refer to `numpy.prod` for full documentation.
        
                Notes
                -----
                Arithmetic is modular when using integer types, and no error is raised
                on overflow.
        
                See Also
                --------
                numpy.ndarray.prod : corresponding function for ndarrays
                numpy.prod : equivalent function
                
        """
    def product(self, axis = None, dtype = None, out = None, keepdims = ...):
        """
        
                Return the product of the array elements over the given axis.
        
                Masked elements are set to 1 internally for computation.
        
                Refer to `numpy.prod` for full documentation.
        
                Notes
                -----
                Arithmetic is modular when using integer types, and no error is raised
                on overflow.
        
                See Also
                --------
                numpy.ndarray.prod : corresponding function for ndarrays
                numpy.prod : equivalent function
                
        """
    def ptp(self, axis = None, out = None, fill_value = None, keepdims = False):
        """
        
                Return (maximum - minimum) along the given dimension
                (i.e. peak-to-peak value).
        
                .. warning::
                    `ptp` preserves the data type of the array. This means the
                    return value for an input of signed integers with n bits
                    (e.g. `np.int8`, `np.int16`, etc) is also a signed integer
                    with n bits.  In that case, peak-to-peak values greater than
                    ``2**(n-1)-1`` will be returned as negative values. An example
                    with a work-around is shown below.
        
                Parameters
                ----------
                axis : {None, int}, optional
                    Axis along which to find the peaks.  If None (default) the
                    flattened array is used.
                out : {None, array_like}, optional
                    Alternative output array in which to place the result. It must
                    have the same shape and buffer length as the expected output
                    but the type will be cast if necessary.
                fill_value : scalar or None, optional
                    Value used to fill in the masked values.
                keepdims : bool, optional
                    If this is set to True, the axes which are reduced are left
                    in the result as dimensions with size one. With this option,
                    the result will broadcast correctly against the array.
        
                Returns
                -------
                ptp : ndarray.
                    A new array holding the result, unless ``out`` was
                    specified, in which case a reference to ``out`` is returned.
        
                Examples
                --------
                >>> x = np.ma.MaskedArray([[4, 9, 2, 10],
                ...                        [6, 9, 7, 12]])
        
                >>> x.ptp(axis=1)
                masked_array(data=[8, 6],
                             mask=False,
                       fill_value=999999)
        
                >>> x.ptp(axis=0)
                masked_array(data=[2, 0, 5, 2],
                             mask=False,
                       fill_value=999999)
        
                >>> x.ptp()
                10
        
                This example shows that a negative value can be returned when
                the input is an array of signed integers.
        
                >>> y = np.ma.MaskedArray([[1, 127],
                ...                        [0, 127],
                ...                        [-1, 127],
                ...                        [-2, 127]], dtype=np.int8)
                >>> y.ptp(axis=1)
                masked_array(data=[ 126,  127, -128, -127],
                             mask=False,
                       fill_value=999999,
                            dtype=int8)
        
                A work-around is to use the `view()` method to view the result as
                unsigned integers with the same bit width:
        
                >>> y.ptp(axis=1).view(np.uint8)
                masked_array(data=[126, 127, 128, 129],
                             mask=False,
                       fill_value=999999,
                            dtype=uint8)
                
        """
    def put(self, indices, values, mode = 'raise'):
        """
        
                Set storage-indexed locations to corresponding values.
        
                Sets self._data.flat[n] = values[n] for each n in indices.
                If `values` is shorter than `indices` then it will repeat.
                If `values` has some masked values, the initial mask is updated
                in consequence, else the corresponding values are unmasked.
        
                Parameters
                ----------
                indices : 1-D array_like
                    Target indices, interpreted as integers.
                values : array_like
                    Values to place in self._data copy at target indices.
                mode : {'raise', 'wrap', 'clip'}, optional
                    Specifies how out-of-bounds indices will behave.
                    'raise' : raise an error.
                    'wrap' : wrap around.
                    'clip' : clip to the range.
        
                Notes
                -----
                `values` can be a scalar or length 1 array.
        
                Examples
                --------
                >>> x = np.ma.array([[1,2,3],[4,5,6],[7,8,9]], mask=[0] + [1,0]*4)
                >>> x
                masked_array(
                  data=[[1, --, 3],
                        [--, 5, --],
                        [7, --, 9]],
                  mask=[[False,  True, False],
                        [ True, False,  True],
                        [False,  True, False]],
                  fill_value=999999)
                >>> x.put([0,4,8],[10,20,30])
                >>> x
                masked_array(
                  data=[[10, --, 3],
                        [--, 20, --],
                        [7, --, 30]],
                  mask=[[False,  True, False],
                        [ True, False,  True],
                        [False,  True, False]],
                  fill_value=999999)
        
                >>> x.put(4,999)
                >>> x
                masked_array(
                  data=[[10, --, 3],
                        [--, 999, --],
                        [7, --, 30]],
                  mask=[[False,  True, False],
                        [ True, False,  True],
                        [False,  True, False]],
                  fill_value=999999)
        
                
        """
    def ravel(self, order = 'C'):
        """
        
                Returns a 1D version of self, as a view.
        
                Parameters
                ----------
                order : {'C', 'F', 'A', 'K'}, optional
                    The elements of `a` are read using this index order. 'C' means to
                    index the elements in C-like order, with the last axis index
                    changing fastest, back to the first axis index changing slowest.
                    'F' means to index the elements in Fortran-like index order, with
                    the first index changing fastest, and the last index changing
                    slowest. Note that the 'C' and 'F' options take no account of the
                    memory layout of the underlying array, and only refer to the order
                    of axis indexing.  'A' means to read the elements in Fortran-like
                    index order if `m` is Fortran *contiguous* in memory, C-like order
                    otherwise.  'K' means to read the elements in the order they occur
                    in memory, except for reversing the data when strides are negative.
                    By default, 'C' index order is used.
                    (Masked arrays currently use 'A' on the data when 'K' is passed.)
        
                Returns
                -------
                MaskedArray
                    Output view is of shape ``(self.size,)`` (or
                    ``(np.ma.product(self.shape),)``).
        
                Examples
                --------
                >>> x = np.ma.array([[1,2,3],[4,5,6],[7,8,9]], mask=[0] + [1,0]*4)
                >>> x
                masked_array(
                  data=[[1, --, 3],
                        [--, 5, --],
                        [7, --, 9]],
                  mask=[[False,  True, False],
                        [ True, False,  True],
                        [False,  True, False]],
                  fill_value=999999)
                >>> x.ravel()
                masked_array(data=[1, --, 3, --, 5, --, 7, --, 9],
                             mask=[False,  True, False,  True, False,  True, False,  True,
                                   False],
                       fill_value=999999)
        
                
        """
    def repeat(self, *args, **params):
        """
        a.repeat(repeats, axis=None)
        
            Repeat elements of an array.
        
            Refer to `numpy.repeat` for full documentation.
        
            See Also
            --------
            numpy.repeat : equivalent function
        """
    def reshape(self, *s, **kwargs):
        """
        
                Give a new shape to the array without changing its data.
        
                Returns a masked array containing the same data, but with a new shape.
                The result is a view on the original array; if this is not possible, a
                ValueError is raised.
        
                Parameters
                ----------
                shape : int or tuple of ints
                    The new shape should be compatible with the original shape. If an
                    integer is supplied, then the result will be a 1-D array of that
                    length.
                order : {'C', 'F'}, optional
                    Determines whether the array data should be viewed as in C
                    (row-major) or FORTRAN (column-major) order.
        
                Returns
                -------
                reshaped_array : array
                    A new view on the array.
        
                See Also
                --------
                reshape : Equivalent function in the masked array module.
                numpy.ndarray.reshape : Equivalent method on ndarray object.
                numpy.reshape : Equivalent function in the NumPy module.
        
                Notes
                -----
                The reshaping operation cannot guarantee that a copy will not be made,
                to modify the shape in place, use ``a.shape = s``
        
                Examples
                --------
                >>> x = np.ma.array([[1,2],[3,4]], mask=[1,0,0,1])
                >>> x
                masked_array(
                  data=[[--, 2],
                        [3, --]],
                  mask=[[ True, False],
                        [False,  True]],
                  fill_value=999999)
                >>> x = x.reshape((4,1))
                >>> x
                masked_array(
                  data=[[--],
                        [2],
                        [3],
                        [--]],
                  mask=[[ True],
                        [False],
                        [False],
                        [ True]],
                  fill_value=999999)
        
                
        """
    def resize(self, newshape, refcheck = True, order = False):
        """
        
                .. warning::
        
                    This method does nothing, except raise a ValueError exception. A
                    masked array does not own its data and therefore cannot safely be
                    resized in place. Use the `numpy.ma.resize` function instead.
        
                This method is difficult to implement safely and may be deprecated in
                future releases of NumPy.
        
                
        """
    def round(self, decimals = 0, out = None):
        """
        
                Return each element rounded to the given number of decimals.
        
                Refer to `numpy.around` for full documentation.
        
                See Also
                --------
                numpy.ndarray.round : corresponding function for ndarrays
                numpy.around : equivalent function
                
        """
    def set_fill_value(self, value = None):
        ...
    def shrink_mask(self):
        """
        
                Reduce a mask to nomask when possible.
        
                Parameters
                ----------
                None
        
                Returns
                -------
                None
        
                Examples
                --------
                >>> x = np.ma.array([[1,2 ], [3, 4]], mask=[0]*4)
                >>> x.mask
                array([[False, False],
                       [False, False]])
                >>> x.shrink_mask()
                masked_array(
                  data=[[1, 2],
                        [3, 4]],
                  mask=False,
                  fill_value=999999)
                >>> x.mask
                False
        
                
        """
    def soften_mask(self):
        """
        
                Force the mask to soft (default), allowing unmasking by assignment.
        
                Whether the mask of a masked array is hard or soft is determined by
                its `~ma.MaskedArray.hardmask` property. `soften_mask` sets
                `~ma.MaskedArray.hardmask` to ``False`` (and returns the modified
                self).
        
                See Also
                --------
                ma.MaskedArray.hardmask
                ma.MaskedArray.harden_mask
        
                
        """
    def sort(self, axis = -1, kind = None, order = None, endwith = True, fill_value = None):
        """
        
                Sort the array, in-place
        
                Parameters
                ----------
                a : array_like
                    Array to be sorted.
                axis : int, optional
                    Axis along which to sort. If None, the array is flattened before
                    sorting. The default is -1, which sorts along the last axis.
                kind : {'quicksort', 'mergesort', 'heapsort', 'stable'}, optional
                    The sorting algorithm used.
                order : list, optional
                    When `a` is a structured array, this argument specifies which fields
                    to compare first, second, and so on.  This list does not need to
                    include all of the fields.
                endwith : {True, False}, optional
                    Whether missing values (if any) should be treated as the largest values
                    (True) or the smallest values (False)
                    When the array contains unmasked values sorting at the same extremes of the
                    datatype, the ordering of these values and the masked values is
                    undefined.
                fill_value : scalar or None, optional
                    Value used internally for the masked values.
                    If ``fill_value`` is not None, it supersedes ``endwith``.
        
                Returns
                -------
                sorted_array : ndarray
                    Array of the same type and shape as `a`.
        
                See Also
                --------
                numpy.ndarray.sort : Method to sort an array in-place.
                argsort : Indirect sort.
                lexsort : Indirect stable sort on multiple keys.
                searchsorted : Find elements in a sorted array.
        
                Notes
                -----
                See ``sort`` for notes on the different sorting algorithms.
        
                Examples
                --------
                >>> a = np.ma.array([1, 2, 5, 4, 3],mask=[0, 1, 0, 1, 0])
                >>> # Default
                >>> a.sort()
                >>> a
                masked_array(data=[1, 3, 5, --, --],
                             mask=[False, False, False,  True,  True],
                       fill_value=999999)
        
                >>> a = np.ma.array([1, 2, 5, 4, 3],mask=[0, 1, 0, 1, 0])
                >>> # Put missing values in the front
                >>> a.sort(endwith=False)
                >>> a
                masked_array(data=[--, --, 1, 3, 5],
                             mask=[ True,  True, False, False, False],
                       fill_value=999999)
        
                >>> a = np.ma.array([1, 2, 5, 4, 3],mask=[0, 1, 0, 1, 0])
                >>> # fill_value takes over endwith
                >>> a.sort(endwith=False, fill_value=3)
                >>> a
                masked_array(data=[1, --, --, 3, 5],
                             mask=[False,  True,  True, False, False],
                       fill_value=999999)
        
                
        """
    def squeeze(self, *args, **params):
        """
        a.squeeze(axis=None)
        
            Remove axes of length one from `a`.
        
            Refer to `numpy.squeeze` for full documentation.
        
            See Also
            --------
            numpy.squeeze : equivalent function
        """
    def std(self, axis = None, dtype = None, out = None, ddof = 0, keepdims = ...):
        """
        
                Returns the standard deviation of the array elements along given axis.
        
                Masked entries are ignored.
        
                Refer to `numpy.std` for full documentation.
        
                See Also
                --------
                numpy.ndarray.std : corresponding function for ndarrays
                numpy.std : Equivalent function
                
        """
    def sum(self, axis = None, dtype = None, out = None, keepdims = ...):
        """
        
                Return the sum of the array elements over the given axis.
        
                Masked elements are set to 0 internally.
        
                Refer to `numpy.sum` for full documentation.
        
                See Also
                --------
                numpy.ndarray.sum : corresponding function for ndarrays
                numpy.sum : equivalent function
        
                Examples
                --------
                >>> x = np.ma.array([[1,2,3],[4,5,6],[7,8,9]], mask=[0] + [1,0]*4)
                >>> x
                masked_array(
                  data=[[1, --, 3],
                        [--, 5, --],
                        [7, --, 9]],
                  mask=[[False,  True, False],
                        [ True, False,  True],
                        [False,  True, False]],
                  fill_value=999999)
                >>> x.sum()
                25
                >>> x.sum(axis=1)
                masked_array(data=[4, 5, 16],
                             mask=[False, False, False],
                       fill_value=999999)
                >>> x.sum(axis=0)
                masked_array(data=[8, 5, 12],
                             mask=[False, False, False],
                       fill_value=999999)
                >>> print(type(x.sum(axis=0, dtype=np.int64)[0]))
                <class 'numpy.int64'>
        
                
        """
    def swapaxes(self, *args, **params):
        """
        a.swapaxes(axis1, axis2)
        
            Return a view of the array with `axis1` and `axis2` interchanged.
        
            Refer to `numpy.swapaxes` for full documentation.
        
            See Also
            --------
            numpy.swapaxes : equivalent function
        """
    def take(self, indices, axis = None, out = None, mode = 'raise'):
        """
        
                
        """
    def tobytes(self, fill_value = None, order = 'C'):
        """
        
                Return the array data as a string containing the raw bytes in the array.
        
                The array is filled with a fill value before the string conversion.
        
                .. versionadded:: 1.9.0
        
                Parameters
                ----------
                fill_value : scalar, optional
                    Value used to fill in the masked values. Default is None, in which
                    case `MaskedArray.fill_value` is used.
                order : {'C','F','A'}, optional
                    Order of the data item in the copy. Default is 'C'.
        
                    - 'C'   -- C order (row major).
                    - 'F'   -- Fortran order (column major).
                    - 'A'   -- Any, current order of array.
                    - None  -- Same as 'A'.
        
                See Also
                --------
                numpy.ndarray.tobytes
                tolist, tofile
        
                Notes
                -----
                As for `ndarray.tobytes`, information about the shape, dtype, etc.,
                but also about `fill_value`, will be lost.
        
                Examples
                --------
                >>> x = np.ma.array(np.array([[1, 2], [3, 4]]), mask=[[0, 1], [1, 0]])
                >>> x.tobytes()
                b'\\x01\\x00\\x00\\x00\\x00\\x00\\x00\\x00?B\\x0f\\x00\\x00\\x00\\x00\\x00?B\\x0f\\x00\\x00\\x00\\x00\\x00\\x04\\x00\\x00\\x00\\x00\\x00\\x00\\x00'
        
                
        """
    def tofile(self, fid, sep = '', format = '%s'):
        """
        
                Save a masked array to a file in binary format.
        
                .. warning::
                  This function is not implemented yet.
        
                Raises
                ------
                NotImplementedError
                    When `tofile` is called.
        
                
        """
    def toflex(self):
        """
        
                Transforms a masked array into a flexible-type array.
        
                The flexible type array that is returned will have two fields:
        
                * the ``_data`` field stores the ``_data`` part of the array.
                * the ``_mask`` field stores the ``_mask`` part of the array.
        
                Parameters
                ----------
                None
        
                Returns
                -------
                record : ndarray
                    A new flexible-type `ndarray` with two fields: the first element
                    containing a value, the second element containing the corresponding
                    mask boolean. The returned record shape matches self.shape.
        
                Notes
                -----
                A side-effect of transforming a masked array into a flexible `ndarray` is
                that meta information (``fill_value``, ...) will be lost.
        
                Examples
                --------
                >>> x = np.ma.array([[1,2,3],[4,5,6],[7,8,9]], mask=[0] + [1,0]*4)
                >>> x
                masked_array(
                  data=[[1, --, 3],
                        [--, 5, --],
                        [7, --, 9]],
                  mask=[[False,  True, False],
                        [ True, False,  True],
                        [False,  True, False]],
                  fill_value=999999)
                >>> x.toflex()
                array([[(1, False), (2,  True), (3, False)],
                       [(4,  True), (5, False), (6,  True)],
                       [(7, False), (8,  True), (9, False)]],
                      dtype=[('_data', '<i8'), ('_mask', '?')])
        
                
        """
    def tolist(self, fill_value = None):
        """
        
                Return the data portion of the masked array as a hierarchical Python list.
        
                Data items are converted to the nearest compatible Python type.
                Masked values are converted to `fill_value`. If `fill_value` is None,
                the corresponding entries in the output list will be ``None``.
        
                Parameters
                ----------
                fill_value : scalar, optional
                    The value to use for invalid entries. Default is None.
        
                Returns
                -------
                result : list
                    The Python list representation of the masked array.
        
                Examples
                --------
                >>> x = np.ma.array([[1,2,3], [4,5,6], [7,8,9]], mask=[0] + [1,0]*4)
                >>> x.tolist()
                [[1, None, 3], [None, 5, None], [7, None, 9]]
                >>> x.tolist(-999)
                [[1, -999, 3], [-999, 5, -999], [7, -999, 9]]
        
                
        """
    def torecords(self):
        """
        
                Transforms a masked array into a flexible-type array.
        
                The flexible type array that is returned will have two fields:
        
                * the ``_data`` field stores the ``_data`` part of the array.
                * the ``_mask`` field stores the ``_mask`` part of the array.
        
                Parameters
                ----------
                None
        
                Returns
                -------
                record : ndarray
                    A new flexible-type `ndarray` with two fields: the first element
                    containing a value, the second element containing the corresponding
                    mask boolean. The returned record shape matches self.shape.
        
                Notes
                -----
                A side-effect of transforming a masked array into a flexible `ndarray` is
                that meta information (``fill_value``, ...) will be lost.
        
                Examples
                --------
                >>> x = np.ma.array([[1,2,3],[4,5,6],[7,8,9]], mask=[0] + [1,0]*4)
                >>> x
                masked_array(
                  data=[[1, --, 3],
                        [--, 5, --],
                        [7, --, 9]],
                  mask=[[False,  True, False],
                        [ True, False,  True],
                        [False,  True, False]],
                  fill_value=999999)
                >>> x.toflex()
                array([[(1, False), (2,  True), (3, False)],
                       [(4,  True), (5, False), (6,  True)],
                       [(7, False), (8,  True), (9, False)]],
                      dtype=[('_data', '<i8'), ('_mask', '?')])
        
                
        """
    def tostring(self, fill_value = None, order = 'C'):
        """
        
                A compatibility alias for `tobytes`, with exactly the same behavior.
        
                Despite its name, it returns `bytes` not `str`\\ s.
        
                .. deprecated:: 1.19.0
                
        """
    def trace(self, offset = 0, axis1 = 0, axis2 = 1, dtype = None, out = None):
        """
        a.trace(offset=0, axis1=0, axis2=1, dtype=None, out=None)
        
            Return the sum along diagonals of the array.
        
            Refer to `numpy.trace` for full documentation.
        
            See Also
            --------
            numpy.trace : equivalent function
        """
    def transpose(self, *args, **params):
        """
        a.transpose(*axes)
        
            Returns a view of the array with axes transposed.
        
            Refer to `numpy.transpose` for full documentation.
        
            Parameters
            ----------
            axes : None, tuple of ints, or `n` ints
        
             * None or no argument: reverses the order of the axes.
        
             * tuple of ints: `i` in the `j`-th place in the tuple means that the
               array's `i`-th axis becomes the transposed array's `j`-th axis.
        
             * `n` ints: same as an n-tuple of the same ints (this form is
               intended simply as a "convenience" alternative to the tuple form).
        
            Returns
            -------
            p : ndarray
                View of the array with its axes suitably permuted.
        
            See Also
            --------
            transpose : Equivalent function.
            ndarray.T : Array property returning the array transposed.
            ndarray.reshape : Give a new shape to an array without changing its data.
        
            Examples
            --------
            >>> a = np.array([[1, 2], [3, 4]])
            >>> a
            array([[1, 2],
                   [3, 4]])
            >>> a.transpose()
            array([[1, 3],
                   [2, 4]])
            >>> a.transpose((1, 0))
            array([[1, 3],
                   [2, 4]])
            >>> a.transpose(1, 0)
            array([[1, 3],
                   [2, 4]])
        
            >>> a = np.array([1, 2, 3, 4])
            >>> a
            array([1, 2, 3, 4])
            >>> a.transpose()
            array([1, 2, 3, 4])
        """
    def unshare_mask(self):
        """
        
                Copy the mask and set the `sharedmask` flag to ``False``.
        
                Whether the mask is shared between masked arrays can be seen from
                the `sharedmask` property. `unshare_mask` ensures the mask is not
                shared. A copy of the mask is only made if it was shared.
        
                See Also
                --------
                sharedmask
        
                
        """
    def var(self, axis = None, dtype = None, out = None, ddof = 0, keepdims = ...):
        """
        
            Compute the variance along the specified axis.
        
            Returns the variance of the array elements, a measure of the spread of a
            distribution.  The variance is computed for the flattened array by
            default, otherwise over the specified axis.
        
            Parameters
            ----------
            a : array_like
                Array containing numbers whose variance is desired.  If `a` is not an
                array, a conversion is attempted.
            axis : None or int or tuple of ints, optional
                Axis or axes along which the variance is computed.  The default is to
                compute the variance of the flattened array.
        
                .. versionadded:: 1.7.0
        
                If this is a tuple of ints, a variance is performed over multiple axes,
                instead of a single axis or all the axes as before.
            dtype : data-type, optional
                Type to use in computing the variance.  For arrays of integer type
                the default is `float64`; for arrays of float types it is the same as
                the array type.
            out : ndarray, optional
                Alternate output array in which to place the result.  It must have
                the same shape as the expected output, but the type is cast if
                necessary.
            ddof : int, optional
                "Delta Degrees of Freedom": the divisor used in the calculation is
                ``N - ddof``, where ``N`` represents the number of elements. By
                default `ddof` is zero.
            keepdims : bool, optional
                If this is set to True, the axes which are reduced are left
                in the result as dimensions with size one. With this option,
                the result will broadcast correctly against the input array.
        
                If the default value is passed, then `keepdims` will not be
                passed through to the `var` method of sub-classes of
                `ndarray`, however any non-default value will be.  If the
                sub-class' method does not implement `keepdims` any
                exceptions will be raised.
        
            where : array_like of bool, optional
                Elements to include in the variance. See `~numpy.ufunc.reduce` for
                details.
        
                .. versionadded:: 1.20.0
        
            Returns
            -------
            variance : ndarray, see dtype parameter above
                If ``out=None``, returns a new array containing the variance;
                otherwise, a reference to the output array is returned.
        
            See Also
            --------
            std, mean, nanmean, nanstd, nanvar
            :ref:`ufuncs-output-type`
        
            Notes
            -----
            The variance is the average of the squared deviations from the mean,
            i.e.,  ``var = mean(x)``, where ``x = abs(a - a.mean())**2``.
        
            The mean is typically calculated as ``x.sum() / N``, where ``N = len(x)``.
            If, however, `ddof` is specified, the divisor ``N - ddof`` is used
            instead.  In standard statistical practice, ``ddof=1`` provides an
            unbiased estimator of the variance of a hypothetical infinite population.
            ``ddof=0`` provides a maximum likelihood estimate of the variance for
            normally distributed variables.
        
            Note that for complex numbers, the absolute value is taken before
            squaring, so that the result is always real and nonnegative.
        
            For floating-point input, the variance is computed using the same
            precision the input has.  Depending on the input data, this can cause
            the results to be inaccurate, especially for `float32` (see example
            below).  Specifying a higher-accuracy accumulator using the ``dtype``
            keyword can alleviate this issue.
        
            Examples
            --------
            >>> a = np.array([[1, 2], [3, 4]])
            >>> np.var(a)
            1.25
            >>> np.var(a, axis=0)
            array([1.,  1.])
            >>> np.var(a, axis=1)
            array([0.25,  0.25])
        
            In single precision, var() can be inaccurate:
        
            >>> a = np.zeros((2, 512*512), dtype=np.float32)
            >>> a[0, :] = 1.0
            >>> a[1, :] = 0.1
            >>> np.var(a)
            0.20250003
        
            Computing the variance in float64 is more accurate:
        
            >>> np.var(a, dtype=np.float64)
            0.20249999932944759 # may vary
            >>> ((1-0.55)**2 + (0.1-0.55)**2)/2
            0.2025
        
            Specifying a where argument:
        
            >>> a = np.array([[14, 8, 11, 10], [7, 9, 10, 11], [10, 15, 5, 10]])
            >>> np.var(a)
            6.833333333333333 # may vary
            >>> np.var(a, where=[[True], [True], [False]])
            4.0
        
            
        """
    def view(self, dtype = None, type = None, fill_value = None):
        """
        
                Return a view of the MaskedArray data.
        
                Parameters
                ----------
                dtype : data-type or ndarray sub-class, optional
                    Data-type descriptor of the returned view, e.g., float32 or int16.
                    The default, None, results in the view having the same data-type
                    as `a`. As with ``ndarray.view``, dtype can also be specified as
                    an ndarray sub-class, which then specifies the type of the
                    returned object (this is equivalent to setting the ``type``
                    parameter).
                type : Python type, optional
                    Type of the returned view, either ndarray or a subclass.  The
                    default None results in type preservation.
                fill_value : scalar, optional
                    The value to use for invalid entries (None by default).
                    If None, then this argument is inferred from the passed `dtype`, or
                    in its absence the original array, as discussed in the notes below.
        
                See Also
                --------
                numpy.ndarray.view : Equivalent method on ndarray object.
        
                Notes
                -----
        
                ``a.view()`` is used two different ways:
        
                ``a.view(some_dtype)`` or ``a.view(dtype=some_dtype)`` constructs a view
                of the array's memory with a different data-type.  This can cause a
                reinterpretation of the bytes of memory.
        
                ``a.view(ndarray_subclass)`` or ``a.view(type=ndarray_subclass)`` just
                returns an instance of `ndarray_subclass` that looks at the same array
                (same shape, dtype, etc.)  This does not cause a reinterpretation of the
                memory.
        
                If `fill_value` is not specified, but `dtype` is specified (and is not
                an ndarray sub-class), the `fill_value` of the MaskedArray will be
                reset. If neither `fill_value` nor `dtype` are specified (or if
                `dtype` is an ndarray sub-class), then the fill value is preserved.
                Finally, if `fill_value` is specified, but `dtype` is not, the fill
                value is set to the specified value.
        
                For ``a.view(some_dtype)``, if ``some_dtype`` has a different number of
                bytes per entry than the previous dtype (for example, converting a
                regular array to a structured array), then the behavior of the view
                cannot be predicted just from the superficial appearance of ``a`` (shown
                by ``print(a)``). It also depends on exactly how ``a`` is stored in
                memory. Therefore if ``a`` is C-ordered versus fortran-ordered, versus
                defined as a slice or transpose, etc., the view may give different
                results.
                
        """
    @property
    def T(self):
        ...
    @property
    def _data(self):
        """
        
                Returns the underlying data, as a view of the masked array.
        
                If the underlying data is a subclass of :class:`numpy.ndarray`, it is
                returned as such.
        
                >>> x = np.ma.array(np.matrix([[1, 2], [3, 4]]), mask=[[0, 1], [1, 0]])
                >>> x.data
                matrix([[1, 2],
                        [3, 4]])
        
                The type of the data can be accessed through the :attr:`baseclass`
                attribute.
                
        """
    @property
    def baseclass(self):
        """
         Class of the underlying data (read-only). 
        """
    @property
    def data(self):
        """
        
                Returns the underlying data, as a view of the masked array.
        
                If the underlying data is a subclass of :class:`numpy.ndarray`, it is
                returned as such.
        
                >>> x = np.ma.array(np.matrix([[1, 2], [3, 4]]), mask=[[0, 1], [1, 0]])
                >>> x.data
                matrix([[1, 2],
                        [3, 4]])
        
                The type of the data can be accessed through the :attr:`baseclass`
                attribute.
                
        """
    @property
    def fill_value(self):
        """
        
                The filling value of the masked array is a scalar. When setting, None
                will set to a default based on the data type.
        
                Examples
                --------
                >>> for dt in [np.int32, np.int64, np.float64, np.complex128]:
                ...     np.ma.array([0, 1], dtype=dt).get_fill_value()
                ...
                999999
                999999
                1e+20
                (1e+20+0j)
        
                >>> x = np.ma.array([0, 1.], fill_value=-np.inf)
                >>> x.fill_value
                -inf
                >>> x.fill_value = np.pi
                >>> x.fill_value
                3.1415926535897931 # may vary
        
                Reset to default:
        
                >>> x.fill_value = None
                >>> x.fill_value
                1e+20
        
                
        """
    @fill_value.setter
    def fill_value(self, value = None):
        ...
    @property
    def flat(self):
        """
         Return a flat iterator, or set a flattened version of self to value. 
        """
    @flat.setter
    def flat(self, value):
        ...
    @property
    def hardmask(self):
        """
        
                Specifies whether values can be unmasked through assignments.
        
                By default, assigning definite values to masked array entries will
                unmask them.  When `hardmask` is ``True``, the mask will not change
                through assignments.
        
                See Also
                --------
                ma.MaskedArray.harden_mask
                ma.MaskedArray.soften_mask
        
                Examples
                --------
                >>> x = np.arange(10)
                >>> m = np.ma.masked_array(x, x>5)
                >>> assert not m.hardmask
        
                Since `m` has a soft mask, assigning an element value unmasks that
                element:
        
                >>> m[8] = 42
                >>> m
                masked_array(data=[0, 1, 2, 3, 4, 5, --, --, 42, --],
                             mask=[False, False, False, False, False, False,
                                   True, True, False, True],
                       fill_value=999999)
        
                After hardening, the mask is not affected by assignments:
        
                >>> hardened = np.ma.harden_mask(m)
                >>> assert m.hardmask and hardened is m
                >>> m[:] = 23
                >>> m
                masked_array(data=[23, 23, 23, 23, 23, 23, --, --, 23, --],
                             mask=[False, False, False, False, False, False,
                                   True, True, False, True],
                       fill_value=999999)
        
                
        """
    @property
    def imag(self):
        """
        
                The imaginary part of the masked array.
        
                This property is a view on the imaginary part of this `MaskedArray`.
        
                See Also
                --------
                real
        
                Examples
                --------
                >>> x = np.ma.array([1+1.j, -2j, 3.45+1.6j], mask=[False, True, False])
                >>> x.imag
                masked_array(data=[1.0, --, 1.6],
                             mask=[False,  True, False],
                       fill_value=1e+20)
        
                
        """
    @property
    def mask(self):
        """
         Current mask. 
        """
    @mask.setter
    def mask(self, value):
        ...
    @property
    def real(self):
        """
        
                The real part of the masked array.
        
                This property is a view on the real part of this `MaskedArray`.
        
                See Also
                --------
                imag
        
                Examples
                --------
                >>> x = np.ma.array([1+1.j, -2j, 3.45+1.6j], mask=[False, True, False])
                >>> x.real
                masked_array(data=[1.0, --, 3.45],
                             mask=[False,  True, False],
                       fill_value=1e+20)
        
                
        """
    @property
    def recordmask(self):
        """
        
                Get or set the mask of the array if it has no named fields. For
                structured arrays, returns a ndarray of booleans where entries are
                ``True`` if **all** the fields are masked, ``False`` otherwise:
        
                >>> x = np.ma.array([(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)],
                ...         mask=[(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)],
                ...        dtype=[('a', int), ('b', int)])
                >>> x.recordmask
                array([False, False,  True, False, False])
                
        """
    @recordmask.setter
    def recordmask(self, mask):
        ...
    @property
    def sharedmask(self):
        """
         Share status of the mask (read-only). 
        """
class MaskedArrayFutureWarning(FutureWarning):
    pass
class MaskedConstant(MaskedArray):
    _MaskedConstant__singleton: typing.ClassVar[MaskedConstant]  # value = masked
    @classmethod
    def _MaskedConstant__has_singleton(cls):
        ...
    @classmethod
    def __new__(cls):
        ...
    def __array_finalize__(self, obj):
        ...
    def __array_prepare__(self, obj, context = None):
        ...
    def __array_wrap__(self, obj, context = None):
        ...
    def __copy__(self):
        ...
    def __deepcopy__(self, memo):
        ...
    def __format__(self, format_spec):
        ...
    def __iadd__(self, other):
        ...
    def __ifloordiv__(self, other):
        ...
    def __imul__(self, other):
        ...
    def __ipow__(self, other):
        ...
    def __isub__(self, other):
        ...
    def __itruediv__(self, other):
        ...
    def __reduce__(self):
        """
        Override of MaskedArray's __reduce__.
                
        """
    def __repr__(self):
        ...
    def __setattr__(self, attr, value):
        ...
    def __str__(self):
        ...
    def copy(self, *args, **kwargs):
        """
         Copy is a no-op on the maskedconstant, as it is a scalar 
        """
class MaskedIterator:
    """
    
        Flat iterator object to iterate over masked arrays.
    
        A `MaskedIterator` iterator is returned by ``x.flat`` for any masked array
        `x`. It allows iterating over the array as if it were a 1-D array,
        either in a for-loop or by calling its `next` method.
    
        Iteration is done in C-contiguous style, with the last index varying the
        fastest. The iterator can also be indexed using basic slicing or
        advanced indexing.
    
        See Also
        --------
        MaskedArray.flat : Return a flat iterator over an array.
        MaskedArray.flatten : Returns a flattened copy of an array.
    
        Notes
        -----
        `MaskedIterator` is not exported by the `ma` module. Instead of
        instantiating a `MaskedIterator` directly, use `MaskedArray.flat`.
    
        Examples
        --------
        >>> x = np.ma.array(arange(6).reshape(2, 3))
        >>> fl = x.flat
        >>> type(fl)
        <class 'numpy.ma.core.MaskedIterator'>
        >>> for item in fl:
        ...     print(item)
        ...
        0
        1
        2
        3
        4
        5
    
        Extracting more than a single element b indexing the `MaskedIterator`
        returns a masked array:
    
        >>> fl[2:4]
        masked_array(data = [2 3],
                     mask = False,
               fill_value = 999999)
    
        
    """
    def __getitem__(self, indx):
        ...
    def __init__(self, ma):
        ...
    def __iter__(self):
        ...
    def __next__(self):
        """
        
                Return the next value, or raise StopIteration.
        
                Examples
                --------
                >>> x = np.ma.array([3, 2], mask=[0, 1])
                >>> fl = x.flat
                >>> next(fl)
                3
                >>> next(fl)
                masked
                >>> next(fl)
                Traceback (most recent call last):
                  ...
                StopIteration
        
                
        """
    def __setitem__(self, index, value):
        ...
class _DomainCheckInterval:
    """
    
        Define a valid interval, so that :
    
        ``domain_check_interval(a,b)(x) == True`` where
        ``x < a`` or ``x > b``.
    
        
    """
    def __call__(self, x):
        """
        Execute the call behavior.
        """
    def __init__(self, a, b):
        """
        domain_check_interval(a,b)(x) = true where x < a or y > b
        """
class _DomainGreater:
    """
    
        DomainGreater(v)(x) is True where x <= v.
    
        
    """
    def __call__(self, x):
        """
        Executes the call behavior.
        """
    def __init__(self, critical_value):
        """
        DomainGreater(v)(x) = true where x <= v
        """
class _DomainGreaterEqual:
    """
    
        DomainGreaterEqual(v)(x) is True where x < v.
    
        
    """
    def __call__(self, x):
        """
        Executes the call behavior.
        """
    def __init__(self, critical_value):
        """
        DomainGreaterEqual(v)(x) = true where x < v
        """
class _DomainSafeDivide:
    """
    
        Define a domain for safe division.
    
        
    """
    def __call__(self, a, b):
        ...
    def __init__(self, tolerance = None):
        ...
class _DomainTan:
    """
    
        Define a valid interval for the `tan` function, so that:
    
        ``domain_tan(eps) = True`` where ``abs(cos(x)) < eps``
    
        
    """
    def __call__(self, x):
        """
        Executes the call behavior.
        """
    def __init__(self, eps):
        """
        domain_tan(eps) = true where abs(cos(x)) < eps)
        """
class _DomainedBinaryOperation(_MaskedUFunc):
    """
    
        Define binary operations that have a domain, like divide.
    
        They have no reduce, outer or accumulate.
    
        Parameters
        ----------
        mbfunc : function
            The function for which to define a masked version. Made available
            as ``_DomainedBinaryOperation.f``.
        domain : class instance
            Default domain for the function. Should be one of the ``_Domain*``
            classes.
        fillx : scalar, optional
            Filling value for the first argument, default is 0.
        filly : scalar, optional
            Filling value for the second argument, default is 0.
    
        
    """
    def __call__(self, a, b, *args, **kwargs):
        """
        Execute the call behavior.
        """
    def __init__(self, dbfunc, domain, fillx = 0, filly = 0):
        """
        abfunc(fillx, filly) must be defined.
                   abfunc(x, filly) = x for all x to enable reduce.
                
        """
class _MaskedBinaryOperation(_MaskedUFunc):
    """
    
        Define masked version of binary operations, where invalid
        values are pre-masked.
    
        Parameters
        ----------
        mbfunc : function
            The function for which to define a masked version. Made available
            as ``_MaskedBinaryOperation.f``.
        domain : class instance
            Default domain for the function. Should be one of the ``_Domain*``
            classes. Default is None.
        fillx : scalar, optional
            Filling value for the first argument, default is 0.
        filly : scalar, optional
            Filling value for the second argument, default is 0.
    
        
    """
    def __call__(self, a, b, *args, **kwargs):
        """
        
                Execute the call behavior.
        
                
        """
    def __init__(self, mbfunc, fillx = 0, filly = 0):
        """
        
                abfunc(fillx, filly) must be defined.
        
                abfunc(x, filly) = x for all x to enable reduce.
        
                
        """
    def accumulate(self, target, axis = 0):
        """
        Accumulate `target` along `axis` after filling with y fill
                value.
        
                
        """
    def outer(self, a, b):
        """
        
                Return the function applied to the outer product of a and b.
        
                
        """
    def reduce(self, target, axis = 0, dtype = None):
        """
        
                Reduce `target` along the given `axis`.
        
                
        """
class _MaskedPrintOption:
    """
    
        Handle the string used to represent missing data in a masked array.
    
        
    """
    def __init__(self, display):
        """
        
                Create the masked_print_option object.
        
                
        """
    def __repr__(self):
        ...
    def __str__(self):
        ...
    def display(self):
        """
        
                Display the string to print for masked values.
        
                
        """
    def enable(self, shrink = 1):
        """
        
                Set the enabling shrink to `shrink`.
        
                
        """
    def enabled(self):
        """
        
                Is the use of the display value enabled?
        
                
        """
    def set_display(self, s):
        """
        
                Set the string to print for masked values.
        
                
        """
class _MaskedUFunc:
    def __init__(self, ufunc):
        ...
    def __str__(self):
        ...
class _MaskedUnaryOperation(_MaskedUFunc):
    """
    
        Defines masked version of unary operations, where invalid values are
        pre-masked.
    
        Parameters
        ----------
        mufunc : callable
            The function for which to define a masked version. Made available
            as ``_MaskedUnaryOperation.f``.
        fill : scalar, optional
            Filling value, default is 0.
        domain : class instance
            Domain for the function. Should be one of the ``_Domain*``
            classes. Default is None.
    
        
    """
    def __call__(self, a, *args, **kwargs):
        """
        
                Execute the call behavior.
        
                
        """
    def __init__(self, mufunc, fill = 0, domain = None):
        ...
class _convert2ma:
    def __call__(self, *args, **params):
        ...
    def __init__(self, funcname, np_ret, np_ma_ret, params = None):
        ...
    def _replace_return_type(self, doc, np_ret, np_ma_ret):
        """
        
                Replace documentation of ``np`` function's return type.
        
                Replaces it with the proper type for the ``np.ma`` function.
        
                Parameters
                ----------
                doc : str
                    The documentation of the ``np`` method.
                np_ret : str
                    The return type string of the ``np`` method that we want to
                    replace. (e.g. "out : ndarray")
                np_ma_ret : str
                    The return type string of the ``np.ma`` method.
                    (e.g. "out : MaskedArray")
                
        """
    def getdoc(self, np_ret, np_ma_ret):
        """
        Return the doc of the function (from the doc of the method).
        """
class _extrema_operation(_MaskedUFunc):
    """
    
        Generic class for maximum/minimum functions.
    
        .. note::
          This is the base class for `_maximum_operation` and
          `_minimum_operation`.
    
        
    """
    def __call__(self, a, b):
        """
        Executes the call behavior.
        """
    def __init__(self, ufunc, compare, fill_value):
        ...
    def outer(self, a, b):
        """
        Return the function applied to the outer product of a and b.
        """
    def reduce(self, target, axis = ...):
        """
        Reduce target along the given axis.
        """
class _frommethod:
    """
    
        Define functions from existing MaskedArray methods.
    
        Parameters
        ----------
        methodname : str
            Name of the method to transform.
    
        
    """
    def __call__(self, a, *args, **params):
        ...
    def __init__(self, methodname, reversed = False):
        ...
    def getdoc(self):
        """
        Return the doc of the function (from the doc of the method).
        """
class mvoid(MaskedArray):
    """
    
        Fake a 'void' object to use for masked array with structured dtypes.
        
    """
    def __getitem__(self, indx):
        """
        
                Get the index.
        
                
        """
    def __iter__(self):
        """
        Defines an iterator for mvoid
        """
    def __len__(self):
        ...
    def __new__(self, data, mask = ..., dtype = None, fill_value = None, hardmask = False, copy = False, subok = True):
        ...
    def __repr__(self):
        ...
    def __setitem__(self, indx, value):
        ...
    def __str__(self):
        ...
    def filled(self, fill_value = None):
        """
        
                Return a copy with masked fields filled with a given value.
        
                Parameters
                ----------
                fill_value : array_like, optional
                    The value to use for invalid entries. Can be scalar or
                    non-scalar. If latter is the case, the filled array should
                    be broadcastable over input array. Default is None, in
                    which case the `fill_value` attribute is used instead.
        
                Returns
                -------
                filled_void
                    A `np.void` object
        
                See Also
                --------
                MaskedArray.filled
        
                
        """
    def tolist(self):
        """
        
            Transforms the mvoid object into a tuple.
        
            Masked fields are replaced by None.
        
            Returns
            -------
            returned_tuple
                Tuple of fields
                
        """
    @property
    def _data(self):
        ...
def _arraymethod(funcname, onmask = True):
    """
    
        Return a class method wrapper around a basic array method.
    
        Creates a class method which returns a masked array, where the new
        ``_data`` array is the output of the corresponding basic method called
        on the original ``_data``.
    
        If `onmask` is True, the new mask is the output of the method called
        on the initial mask. Otherwise, the new mask is just a reference
        to the initial mask.
    
        Parameters
        ----------
        funcname : str
            Name of the function to apply on data.
        onmask : bool
            Whether the mask must be processed also (True) or left
            alone (False). Default is True. Make available as `_onmask`
            attribute.
    
        Returns
        -------
        method : instancemethod
            Class method wrapper of the specified basic array method.
    
        
    """
def _check_fill_value(fill_value, ndtype):
    """
    
        Private function validating the given `fill_value` for the given dtype.
    
        If fill_value is None, it is set to the default corresponding to the dtype.
    
        If fill_value is not None, its value is forced to the given dtype.
    
        The result is always a 0d array.
    
        
    """
def _check_mask_axis(mask, axis, keepdims = ...):
    """
    Check whether there are masked values along the given axis
    """
def _convolve_or_correlate(f, a, v, mode, propagate_mask):
    """
    
        Helper function for ma.correlate and ma.convolve
        
    """
def _deprecate_argsort_axis(arr):
    """
    
        Adjust the axis passed to argsort, warning if necessary
    
        Parameters
        ----------
        arr
            The array which argsort was called on
    
        np.ma.argsort has a long-term bug where the default of the axis argument
        is wrong (gh-8701), which now must be kept for backwards compatibility.
        Thankfully, this only makes a difference when arrays are 2- or more-
        dimensional, so we only need a warning then.
        
    """
def _extremum_fill_value(obj, extremum, extremum_name):
    ...
def _get_dtype_of(obj):
    """
     Convert the argument for *_fill_value into a dtype 
    """
def _mareconstruct(subtype, baseclass, baseshape, basetype):
    """
    Internal function that builds a new MaskedArray from the
        information stored in a pickle.
    
        
    """
def _mask_propagate(a, axis):
    """
    
        Mask whole 1-d vectors of an array that contain masked values.
        
    """
def _recursive_fill_value(dtype, f):
    """
    
        Recursively produce a fill value for `dtype`, calling f on scalar dtypes
        
    """
def _recursive_filled(a, mask, fill_value):
    """
    
        Recursively fill `a` with `fill_value`.
    
        
    """
def _recursive_mask_or(m1, m2, newmask):
    ...
def _recursive_printoption(result, mask, printopt):
    """
    
        Puts printoptions in result where mask is True.
    
        Private function allowing for recursion
    
        
    """
def _recursive_set_fill_value(fillvalue, dt):
    """
    
        Create a fill value for a structured dtype.
    
        Parameters
        ----------
        fillvalue : scalar or array_like
            Scalar or array representing the fill value. If it is of shorter
            length than the number of fields in dt, it will be resized.
        dt : dtype
            The structured dtype for which to create the fill value.
    
        Returns
        -------
        val : tuple
            A tuple of values corresponding to the structured fill value.
    
        
    """
def _replace_dtype_fields(dtype, primitive_dtype):
    """
    
        Construct a dtype description list from a given dtype.
    
        Returns a new dtype object, with all fields and subtypes in the given type
        recursively replaced with `primitive_dtype`.
    
        Arguments are coerced to dtypes first.
        
    """
def _replace_dtype_fields_recursive(dtype, primitive_dtype):
    """
    Private function allowing recursion in _replace_dtype_fields.
    """
def _shrink_mask(m):
    """
    
        Shrink a mask to nomask if possible
        
    """
def allclose(a, b, masked_equal = True, rtol = 1e-05, atol = 1e-08):
    """
    
        Returns True if two arrays are element-wise equal within a tolerance.
    
        This function is equivalent to `allclose` except that masked values
        are treated as equal (default) or unequal, depending on the `masked_equal`
        argument.
    
        Parameters
        ----------
        a, b : array_like
            Input arrays to compare.
        masked_equal : bool, optional
            Whether masked values in `a` and `b` are considered equal (True) or not
            (False). They are considered equal by default.
        rtol : float, optional
            Relative tolerance. The relative difference is equal to ``rtol * b``.
            Default is 1e-5.
        atol : float, optional
            Absolute tolerance. The absolute difference is equal to `atol`.
            Default is 1e-8.
    
        Returns
        -------
        y : bool
            Returns True if the two arrays are equal within the given
            tolerance, False otherwise. If either array contains NaN, then
            False is returned.
    
        See Also
        --------
        all, any
        numpy.allclose : the non-masked `allclose`.
    
        Notes
        -----
        If the following equation is element-wise True, then `allclose` returns
        True::
    
          absolute(`a` - `b`) <= (`atol` + `rtol` * absolute(`b`))
    
        Return True if all elements of `a` and `b` are equal subject to
        given tolerances.
    
        Examples
        --------
        >>> a = np.ma.array([1e10, 1e-7, 42.0], mask=[0, 0, 1])
        >>> a
        masked_array(data=[10000000000.0, 1e-07, --],
                     mask=[False, False,  True],
               fill_value=1e+20)
        >>> b = np.ma.array([1e10, 1e-8, -42.0], mask=[0, 0, 1])
        >>> np.ma.allclose(a, b)
        False
    
        >>> a = np.ma.array([1e10, 1e-8, 42.0], mask=[0, 0, 1])
        >>> b = np.ma.array([1.00001e10, 1e-9, -42.0], mask=[0, 0, 1])
        >>> np.ma.allclose(a, b)
        True
        >>> np.ma.allclose(a, b, masked_equal=False)
        False
    
        Masked values are not compared directly.
    
        >>> a = np.ma.array([1e10, 1e-8, 42.0], mask=[0, 0, 1])
        >>> b = np.ma.array([1.00001e10, 1e-9, 42.0], mask=[0, 0, 1])
        >>> np.ma.allclose(a, b)
        True
        >>> np.ma.allclose(a, b, masked_equal=False)
        False
    
        
    """
def allequal(a, b, fill_value = True):
    """
    
        Return True if all entries of a and b are equal, using
        fill_value as a truth value where either or both are masked.
    
        Parameters
        ----------
        a, b : array_like
            Input arrays to compare.
        fill_value : bool, optional
            Whether masked values in a or b are considered equal (True) or not
            (False).
    
        Returns
        -------
        y : bool
            Returns True if the two arrays are equal within the given
            tolerance, False otherwise. If either array contains NaN,
            then False is returned.
    
        See Also
        --------
        all, any
        numpy.ma.allclose
    
        Examples
        --------
        >>> a = np.ma.array([1e10, 1e-7, 42.0], mask=[0, 0, 1])
        >>> a
        masked_array(data=[10000000000.0, 1e-07, --],
                     mask=[False, False,  True],
               fill_value=1e+20)
    
        >>> b = np.array([1e10, 1e-7, -42.0])
        >>> b
        array([  1.00000000e+10,   1.00000000e-07,  -4.20000000e+01])
        >>> np.ma.allequal(a, b, fill_value=False)
        False
        >>> np.ma.allequal(a, b)
        True
    
        
    """
def append(a, b, axis = None):
    """
    Append values to the end of an array.
    
        .. versionadded:: 1.9.0
    
        Parameters
        ----------
        a : array_like
            Values are appended to a copy of this array.
        b : array_like
            These values are appended to a copy of `a`.  It must be of the
            correct shape (the same shape as `a`, excluding `axis`).  If `axis`
            is not specified, `b` can be any shape and will be flattened
            before use.
        axis : int, optional
            The axis along which `v` are appended.  If `axis` is not given,
            both `a` and `b` are flattened before use.
    
        Returns
        -------
        append : MaskedArray
            A copy of `a` with `b` appended to `axis`.  Note that `append`
            does not occur in-place: a new array is allocated and filled.  If
            `axis` is None, the result is a flattened array.
    
        See Also
        --------
        numpy.append : Equivalent function in the top-level NumPy module.
    
        Examples
        --------
        >>> import numpy.ma as ma
        >>> a = ma.masked_values([1, 2, 3], 2)
        >>> b = ma.masked_values([[4, 5, 6], [7, 8, 9]], 7)
        >>> ma.append(a, b)
        masked_array(data=[1, --, 3, 4, 5, 6, --, 8, 9],
                     mask=[False,  True, False, False, False, False,  True, False,
                           False],
               fill_value=999999)
        
    """
def argsort(a, axis = ..., kind = None, order = None, endwith = True, fill_value = None):
    """
    
            Return an ndarray of indices that sort the array along the
            specified axis.  Masked values are filled beforehand to
            `fill_value`.
    
            Parameters
            ----------
            axis : int, optional
                Axis along which to sort. If None, the default, the flattened array
                is used.
    
                ..  versionchanged:: 1.13.0
                    Previously, the default was documented to be -1, but that was
                    in error. At some future date, the default will change to -1, as
                    originally intended.
                    Until then, the axis should be given explicitly when
                    ``arr.ndim > 1``, to avoid a FutureWarning.
            kind : {'quicksort', 'mergesort', 'heapsort', 'stable'}, optional
                The sorting algorithm used.
            order : list, optional
                When `a` is an array with fields defined, this argument specifies
                which fields to compare first, second, etc.  Not all fields need be
                specified.
            endwith : {True, False}, optional
                Whether missing values (if any) should be treated as the largest values
                (True) or the smallest values (False)
                When the array contains unmasked values at the same extremes of the
                datatype, the ordering of these values and the masked values is
                undefined.
            fill_value : scalar or None, optional
                Value used internally for the masked values.
                If ``fill_value`` is not None, it supersedes ``endwith``.
    
            Returns
            -------
            index_array : ndarray, int
                Array of indices that sort `a` along the specified axis.
                In other words, ``a[index_array]`` yields a sorted `a`.
    
            See Also
            --------
            ma.MaskedArray.sort : Describes sorting algorithms used.
            lexsort : Indirect stable sort with multiple keys.
            numpy.ndarray.sort : Inplace sort.
    
            Notes
            -----
            See `sort` for notes on the different sorting algorithms.
    
            Examples
            --------
            >>> a = np.ma.array([3,2,1], mask=[False, False, True])
            >>> a
            masked_array(data=[3, 2, --],
                         mask=[False, False,  True],
                   fill_value=999999)
            >>> a.argsort()
            array([1, 0, 2])
    
            
    """
def array(data, dtype = None, copy = False, order = None, mask = ..., fill_value = None, keep_mask = True, hard_mask = False, shrink = True, subok = True, ndmin = 0):
    """
    
        An array class with possibly masked values.
    
        Masked values of True exclude the corresponding element from any
        computation.
    
        Construction::
    
          x = MaskedArray(data, mask=nomask, dtype=None, copy=False, subok=True,
                          ndmin=0, fill_value=None, keep_mask=True, hard_mask=None,
                          shrink=True, order=None)
    
        Parameters
        ----------
        data : array_like
            Input data.
        mask : sequence, optional
            Mask. Must be convertible to an array of booleans with the same
            shape as `data`. True indicates a masked (i.e. invalid) data.
        dtype : dtype, optional
            Data type of the output.
            If `dtype` is None, the type of the data argument (``data.dtype``)
            is used. If `dtype` is not None and different from ``data.dtype``,
            a copy is performed.
        copy : bool, optional
            Whether to copy the input data (True), or to use a reference instead.
            Default is False.
        subok : bool, optional
            Whether to return a subclass of `MaskedArray` if possible (True) or a
            plain `MaskedArray`. Default is True.
        ndmin : int, optional
            Minimum number of dimensions. Default is 0.
        fill_value : scalar, optional
            Value used to fill in the masked values when necessary.
            If None, a default based on the data-type is used.
        keep_mask : bool, optional
            Whether to combine `mask` with the mask of the input data, if any
            (True), or to use only `mask` for the output (False). Default is True.
        hard_mask : bool, optional
            Whether to use a hard mask or not. With a hard mask, masked values
            cannot be unmasked. Default is False.
        shrink : bool, optional
            Whether to force compression of an empty mask. Default is True.
        order : {'C', 'F', 'A'}, optional
            Specify the order of the array.  If order is 'C', then the array
            will be in C-contiguous order (last-index varies the fastest).
            If order is 'F', then the returned array will be in
            Fortran-contiguous order (first-index varies the fastest).
            If order is 'A' (default), then the returned array may be
            in any order (either C-, Fortran-contiguous, or even discontiguous),
            unless a copy is required, in which case it will be C-contiguous.
    
        Examples
        --------
    
        The ``mask`` can be initialized with an array of boolean values
        with the same shape as ``data``.
    
        >>> data = np.arange(6).reshape((2, 3))
        >>> np.ma.MaskedArray(data, mask=[[False, True, False],
        ...                               [False, False, True]])
        masked_array(
          data=[[0, --, 2],
                [3, 4, --]],
          mask=[[False,  True, False],
                [False, False,  True]],
          fill_value=999999)
    
        Alternatively, the ``mask`` can be initialized to homogeneous boolean
        array with the same shape as ``data`` by passing in a scalar
        boolean value:
    
        >>> np.ma.MaskedArray(data, mask=False)
        masked_array(
          data=[[0, 1, 2],
                [3, 4, 5]],
          mask=[[False, False, False],
                [False, False, False]],
          fill_value=999999)
    
        >>> np.ma.MaskedArray(data, mask=True)
        masked_array(
          data=[[--, --, --],
                [--, --, --]],
          mask=[[ True,  True,  True],
                [ True,  True,  True]],
          fill_value=999999,
          dtype=int64)
    
        .. note::
            The recommended practice for initializing ``mask`` with a scalar
            boolean value is to use ``True``/``False`` rather than
            ``np.True_``/``np.False_``. The reason is :attr:`nomask`
            is represented internally as ``np.False_``.
    
            >>> np.False_ is np.ma.nomask
            True
    
        
    """
def asanyarray(a, dtype = None):
    """
    
        Convert the input to a masked array, conserving subclasses.
    
        If `a` is a subclass of `MaskedArray`, its class is conserved.
        No copy is performed if the input is already an `ndarray`.
    
        Parameters
        ----------
        a : array_like
            Input data, in any form that can be converted to an array.
        dtype : dtype, optional
            By default, the data-type is inferred from the input data.
        order : {'C', 'F'}, optional
            Whether to use row-major ('C') or column-major ('FORTRAN') memory
            representation.  Default is 'C'.
    
        Returns
        -------
        out : MaskedArray
            MaskedArray interpretation of `a`.
    
        See Also
        --------
        asarray : Similar to `asanyarray`, but does not conserve subclass.
    
        Examples
        --------
        >>> x = np.arange(10.).reshape(2, 5)
        >>> x
        array([[0., 1., 2., 3., 4.],
               [5., 6., 7., 8., 9.]])
        >>> np.ma.asanyarray(x)
        masked_array(
          data=[[0., 1., 2., 3., 4.],
                [5., 6., 7., 8., 9.]],
          mask=False,
          fill_value=1e+20)
        >>> type(np.ma.asanyarray(x))
        <class 'numpy.ma.core.MaskedArray'>
    
        
    """
def asarray(a, dtype = None, order = None):
    """
    
        Convert the input to a masked array of the given data-type.
    
        No copy is performed if the input is already an `ndarray`. If `a` is
        a subclass of `MaskedArray`, a base class `MaskedArray` is returned.
    
        Parameters
        ----------
        a : array_like
            Input data, in any form that can be converted to a masked array. This
            includes lists, lists of tuples, tuples, tuples of tuples, tuples
            of lists, ndarrays and masked arrays.
        dtype : dtype, optional
            By default, the data-type is inferred from the input data.
        order : {'C', 'F'}, optional
            Whether to use row-major ('C') or column-major ('FORTRAN') memory
            representation.  Default is 'C'.
    
        Returns
        -------
        out : MaskedArray
            Masked array interpretation of `a`.
    
        See Also
        --------
        asanyarray : Similar to `asarray`, but conserves subclasses.
    
        Examples
        --------
        >>> x = np.arange(10.).reshape(2, 5)
        >>> x
        array([[0., 1., 2., 3., 4.],
               [5., 6., 7., 8., 9.]])
        >>> np.ma.asarray(x)
        masked_array(
          data=[[0., 1., 2., 3., 4.],
                [5., 6., 7., 8., 9.]],
          mask=False,
          fill_value=1e+20)
        >>> type(np.ma.asarray(x))
        <class 'numpy.ma.core.MaskedArray'>
    
        
    """
def choose(indices, choices, out = None, mode = 'raise'):
    """
    
        Use an index array to construct a new array from a list of choices.
    
        Given an array of integers and a list of n choice arrays, this method
        will create a new array that merges each of the choice arrays.  Where a
        value in `index` is i, the new array will have the value that choices[i]
        contains in the same place.
    
        Parameters
        ----------
        indices : ndarray of ints
            This array must contain integers in ``[0, n-1]``, where n is the
            number of choices.
        choices : sequence of arrays
            Choice arrays. The index array and all of the choices should be
            broadcastable to the same shape.
        out : array, optional
            If provided, the result will be inserted into this array. It should
            be of the appropriate shape and `dtype`.
        mode : {'raise', 'wrap', 'clip'}, optional
            Specifies how out-of-bounds indices will behave.
    
            * 'raise' : raise an error
            * 'wrap' : wrap around
            * 'clip' : clip to the range
    
        Returns
        -------
        merged_array : array
    
        See Also
        --------
        choose : equivalent function
    
        Examples
        --------
        >>> choice = np.array([[1,1,1], [2,2,2], [3,3,3]])
        >>> a = np.array([2, 1, 0])
        >>> np.ma.choose(a, choice)
        masked_array(data=[3, 2, 1],
                     mask=False,
               fill_value=999999)
    
        
    """
def common_fill_value(a, b):
    """
    
        Return the common filling value of two masked arrays, if any.
    
        If ``a.fill_value == b.fill_value``, return the fill value,
        otherwise return None.
    
        Parameters
        ----------
        a, b : MaskedArray
            The masked arrays for which to compare fill values.
    
        Returns
        -------
        fill_value : scalar or None
            The common fill value, or None.
    
        Examples
        --------
        >>> x = np.ma.array([0, 1.], fill_value=3)
        >>> y = np.ma.array([0, 1.], fill_value=3)
        >>> np.ma.common_fill_value(x, y)
        3.0
    
        
    """
def compressed(x):
    """
    
        Return all the non-masked data as a 1-D array.
    
        This function is equivalent to calling the "compressed" method of a
        `ma.MaskedArray`, see `ma.MaskedArray.compressed` for details.
    
        See Also
        --------
        ma.MaskedArray.compressed : Equivalent method.
    
        Examples
        --------
        
        Create an array with negative values masked:
    
        >>> import numpy as np
        >>> x = np.array([[1, -1, 0], [2, -1, 3], [7, 4, -1]])
        >>> masked_x = np.ma.masked_array(x, mask=x < 0)
        >>> masked_x
        masked_array(
          data=[[1, --, 0],
                [2, --, 3],
                [7, 4, --]],
          mask=[[False,  True, False],
                [False,  True, False],
                [False, False,  True]],
          fill_value=999999)
    
        Compress the masked array into a 1-D array of non-masked values:
    
        >>> np.ma.compressed(masked_x)
        array([1, 0, 2, 3, 7, 4])
    
        
    """
def concatenate(arrays, axis = 0):
    """
    
        Concatenate a sequence of arrays along the given axis.
    
        Parameters
        ----------
        arrays : sequence of array_like
            The arrays must have the same shape, except in the dimension
            corresponding to `axis` (the first, by default).
        axis : int, optional
            The axis along which the arrays will be joined. Default is 0.
    
        Returns
        -------
        result : MaskedArray
            The concatenated array with any masked entries preserved.
    
        See Also
        --------
        numpy.concatenate : Equivalent function in the top-level NumPy module.
    
        Examples
        --------
        >>> import numpy.ma as ma
        >>> a = ma.arange(3)
        >>> a[1] = ma.masked
        >>> b = ma.arange(2, 5)
        >>> a
        masked_array(data=[0, --, 2],
                     mask=[False,  True, False],
               fill_value=999999)
        >>> b
        masked_array(data=[2, 3, 4],
                     mask=False,
               fill_value=999999)
        >>> ma.concatenate([a, b])
        masked_array(data=[0, --, 2, 2, 3, 4],
                     mask=[False,  True, False, False, False, False],
               fill_value=999999)
    
        
    """
def convolve(a, v, mode = 'full', propagate_mask = True):
    """
    
        Returns the discrete, linear convolution of two one-dimensional sequences.
    
        Parameters
        ----------
        a, v : array_like
            Input sequences.
        mode : {'valid', 'same', 'full'}, optional
            Refer to the `np.convolve` docstring.
        propagate_mask : bool
            If True, then if any masked element is included in the sum for a result
            element, then the result is masked.
            If False, then the result element is only masked if no non-masked cells
            contribute towards it
    
        Returns
        -------
        out : MaskedArray
            Discrete, linear convolution of `a` and `v`.
    
        See Also
        --------
        numpy.convolve : Equivalent function in the top-level NumPy module.
        
    """
def correlate(a, v, mode = 'valid', propagate_mask = True):
    """
    
        Cross-correlation of two 1-dimensional sequences.
    
        Parameters
        ----------
        a, v : array_like
            Input sequences.
        mode : {'valid', 'same', 'full'}, optional
            Refer to the `np.convolve` docstring.  Note that the default
            is 'valid', unlike `convolve`, which uses 'full'.
        propagate_mask : bool
            If True, then a result element is masked if any masked element contributes towards it.
            If False, then a result element is only masked if no non-masked element
            contribute towards it
    
        Returns
        -------
        out : MaskedArray
            Discrete cross-correlation of `a` and `v`.
    
        See Also
        --------
        numpy.correlate : Equivalent function in the top-level NumPy module.
        
    """
def default_fill_value(obj):
    """
    
        Return the default fill value for the argument object.
    
        The default filling value depends on the datatype of the input
        array or the type of the input scalar:
    
           ========  ========
           datatype  default
           ========  ========
           bool      True
           int       999999
           float     1.e20
           complex   1.e20+0j
           object    '?'
           string    'N/A'
           ========  ========
    
        For structured types, a structured scalar is returned, with each field the
        default fill value for its type.
    
        For subarray types, the fill value is an array of the same size containing
        the default scalar fill value.
    
        Parameters
        ----------
        obj : ndarray, dtype or scalar
            The array data-type or scalar for which the default fill value
            is returned.
    
        Returns
        -------
        fill_value : scalar
            The default fill value.
    
        Examples
        --------
        >>> np.ma.default_fill_value(1)
        999999
        >>> np.ma.default_fill_value(np.array([1.1, 2., np.pi]))
        1e+20
        >>> np.ma.default_fill_value(np.dtype(complex))
        (1e+20+0j)
    
        
    """
def diag(v, k = 0):
    """
    
        Extract a diagonal or construct a diagonal array.
    
        This function is the equivalent of `numpy.diag` that takes masked
        values into account, see `numpy.diag` for details.
    
        See Also
        --------
        numpy.diag : Equivalent function for ndarrays.
    
        Examples
        --------
    
        Create an array with negative values masked:
    
        >>> import numpy as np
        >>> x = np.array([[11.2, -3.973, 18], [0.801, -1.41, 12], [7, 33, -12]])
        >>> masked_x = np.ma.masked_array(x, mask=x < 0)
        >>> masked_x
        masked_array(
          data=[[11.2, --, 18.0],
                [0.801, --, 12.0],
                [7.0, 33.0, --]],
          mask=[[False,  True, False],
                [False,  True, False],
                [False, False,  True]],
          fill_value=1e+20)
    
        Isolate the main diagonal from the masked array:
    
        >>> np.ma.diag(masked_x)
        masked_array(data=[11.2, --, --],
                     mask=[False,  True,  True],
               fill_value=1e+20)
    
        Isolate the first diagonal below the main diagonal:
    
        >>> np.ma.diag(masked_x, -1)
        masked_array(data=[0.801, 33.0],
                     mask=[False, False],
               fill_value=1e+20)
    
        
    """
def diff(a, n = 1, axis = -1, prepend = ..., append = ...):
    """
    
        Calculate the n-th discrete difference along the given axis.
        The first difference is given by ``out[i] = a[i+1] - a[i]`` along
        the given axis, higher differences are calculated by using `diff`
        recursively.
        Preserves the input mask.
    
        Parameters
        ----------
        a : array_like
            Input array
        n : int, optional
            The number of times values are differenced. If zero, the input
            is returned as-is.
        axis : int, optional
            The axis along which the difference is taken, default is the
            last axis.
        prepend, append : array_like, optional
            Values to prepend or append to `a` along axis prior to
            performing the difference.  Scalar values are expanded to
            arrays with length 1 in the direction of axis and the shape
            of the input array in along all other axes.  Otherwise the
            dimension and shape must match `a` except along axis.
    
        Returns
        -------
        diff : MaskedArray
            The n-th differences. The shape of the output is the same as `a`
            except along `axis` where the dimension is smaller by `n`. The
            type of the output is the same as the type of the difference
            between any two elements of `a`. This is the same as the type of
            `a` in most cases. A notable exception is `datetime64`, which
            results in a `timedelta64` output array.
    
        See Also
        --------
        numpy.diff : Equivalent function in the top-level NumPy module.
    
        Notes
        -----
        Type is preserved for boolean arrays, so the result will contain
        `False` when consecutive elements are the same and `True` when they
        differ.
    
        For unsigned integer arrays, the results will also be unsigned. This
        should not be surprising, as the result is consistent with
        calculating the difference directly:
    
        >>> u8_arr = np.array([1, 0], dtype=np.uint8)
        >>> np.ma.diff(u8_arr)
        masked_array(data=[255],
                     mask=False,
               fill_value=999999,
                    dtype=uint8)
        >>> u8_arr[1,...] - u8_arr[0,...]
        255
    
        If this is not desirable, then the array should be cast to a larger
        integer type first:
    
        >>> i16_arr = u8_arr.astype(np.int16)
        >>> np.ma.diff(i16_arr)
        masked_array(data=[-1],
                     mask=False,
               fill_value=999999,
                    dtype=int16)
    
        Examples
        --------
        >>> a = np.array([1, 2, 3, 4, 7, 0, 2, 3])
        >>> x = np.ma.masked_where(a < 2, a)
        >>> np.ma.diff(x)
        masked_array(data=[--, 1, 1, 3, --, --, 1],
                mask=[ True, False, False, False,  True,  True, False],
            fill_value=999999)
    
        >>> np.ma.diff(x, n=2)
        masked_array(data=[--, 0, 2, --, --, --],
                    mask=[ True, False, False,  True,  True,  True],
            fill_value=999999)
    
        >>> a = np.array([[1, 3, 1, 5, 10], [0, 1, 5, 6, 8]])
        >>> x = np.ma.masked_equal(a, value=1)
        >>> np.ma.diff(x)
        masked_array(
            data=[[--, --, --, 5],
                    [--, --, 1, 2]],
            mask=[[ True,  True,  True, False],
                    [ True,  True, False, False]],
            fill_value=1)
    
        >>> np.ma.diff(x, axis=0)
        masked_array(data=[[--, --, --, 1, -2]],
                mask=[[ True,  True,  True, False, False]],
            fill_value=1)
    
        
    """
def doc_note(initialdoc, note):
    """
    
        Adds a Notes section to an existing docstring.
    
        
    """
def dot(a, b, strict = False, out = None):
    """
    
        Return the dot product of two arrays.
    
        This function is the equivalent of `numpy.dot` that takes masked values
        into account. Note that `strict` and `out` are in different position
        than in the method version. In order to maintain compatibility with the
        corresponding method, it is recommended that the optional arguments be
        treated as keyword only.  At some point that may be mandatory.
    
        Parameters
        ----------
        a, b : masked_array_like
            Inputs arrays.
        strict : bool, optional
            Whether masked data are propagated (True) or set to 0 (False) for
            the computation. Default is False.  Propagating the mask means that
            if a masked value appears in a row or column, the whole row or
            column is considered masked.
        out : masked_array, optional
            Output argument. This must have the exact kind that would be returned
            if it was not used. In particular, it must have the right type, must be
            C-contiguous, and its dtype must be the dtype that would be returned
            for `dot(a,b)`. This is a performance feature. Therefore, if these
            conditions are not met, an exception is raised, instead of attempting
            to be flexible.
    
            .. versionadded:: 1.10.2
    
        See Also
        --------
        numpy.dot : Equivalent function for ndarrays.
    
        Examples
        --------
        >>> a = np.ma.array([[1, 2, 3], [4, 5, 6]], mask=[[1, 0, 0], [0, 0, 0]])
        >>> b = np.ma.array([[1, 2], [3, 4], [5, 6]], mask=[[1, 0], [0, 0], [0, 0]])
        >>> np.ma.dot(a, b)
        masked_array(
          data=[[21, 26],
                [45, 64]],
          mask=[[False, False],
                [False, False]],
          fill_value=999999)
        >>> np.ma.dot(a, b, strict=True)
        masked_array(
          data=[[--, --],
                [--, 64]],
          mask=[[ True,  True],
                [ True, False]],
          fill_value=999999)
    
        
    """
def filled(a, fill_value = None):
    """
    
        Return input as an array with masked data replaced by a fill value.
    
        If `a` is not a `MaskedArray`, `a` itself is returned.
        If `a` is a `MaskedArray` and `fill_value` is None, `fill_value` is set to
        ``a.fill_value``.
    
        Parameters
        ----------
        a : MaskedArray or array_like
            An input object.
        fill_value : array_like, optional.
            Can be scalar or non-scalar. If non-scalar, the
            resulting filled array should be broadcastable
            over input array. Default is None.
    
        Returns
        -------
        a : ndarray
            The filled array.
    
        See Also
        --------
        compressed
    
        Examples
        --------
        >>> x = np.ma.array(np.arange(9).reshape(3, 3), mask=[[1, 0, 0],
        ...                                                   [1, 0, 0],
        ...                                                   [0, 0, 0]])
        >>> x.filled()
        array([[999999,      1,      2],
               [999999,      4,      5],
               [     6,      7,      8]])
        >>> x.filled(fill_value=333)
        array([[333,   1,   2],
               [333,   4,   5],
               [  6,   7,   8]])
        >>> x.filled(fill_value=np.arange(3))
        array([[0, 1, 2],
               [0, 4, 5],
               [6, 7, 8]])
    
        
    """
def fix_invalid(a, mask = ..., copy = True, fill_value = None):
    """
    
        Return input with invalid data masked and replaced by a fill value.
    
        Invalid data means values of `nan`, `inf`, etc.
    
        Parameters
        ----------
        a : array_like
            Input array, a (subclass of) ndarray.
        mask : sequence, optional
            Mask. Must be convertible to an array of booleans with the same
            shape as `data`. True indicates a masked (i.e. invalid) data.
        copy : bool, optional
            Whether to use a copy of `a` (True) or to fix `a` in place (False).
            Default is True.
        fill_value : scalar, optional
            Value used for fixing invalid data. Default is None, in which case
            the ``a.fill_value`` is used.
    
        Returns
        -------
        b : MaskedArray
            The input array with invalid entries fixed.
    
        Notes
        -----
        A copy is performed by default.
    
        Examples
        --------
        >>> x = np.ma.array([1., -1, np.nan, np.inf], mask=[1] + [0]*3)
        >>> x
        masked_array(data=[--, -1.0, nan, inf],
                     mask=[ True, False, False, False],
               fill_value=1e+20)
        >>> np.ma.fix_invalid(x)
        masked_array(data=[--, -1.0, --, --],
                     mask=[ True, False,  True,  True],
               fill_value=1e+20)
    
        >>> fixed = np.ma.fix_invalid(x)
        >>> fixed.data
        array([ 1.e+00, -1.e+00,  1.e+20,  1.e+20])
        >>> x.data
        array([ 1., -1., nan, inf])
    
        
    """
def flatten_mask(mask):
    """
    
        Returns a completely flattened version of the mask, where nested fields
        are collapsed.
    
        Parameters
        ----------
        mask : array_like
            Input array, which will be interpreted as booleans.
    
        Returns
        -------
        flattened_mask : ndarray of bools
            The flattened input.
    
        Examples
        --------
        >>> mask = np.array([0, 0, 1])
        >>> np.ma.flatten_mask(mask)
        array([False, False,  True])
    
        >>> mask = np.array([(0, 0), (0, 1)], dtype=[('a', bool), ('b', bool)])
        >>> np.ma.flatten_mask(mask)
        array([False, False, False,  True])
    
        >>> mdtype = [('a', bool), ('b', [('ba', bool), ('bb', bool)])]
        >>> mask = np.array([(0, (0, 0)), (0, (0, 1))], dtype=mdtype)
        >>> np.ma.flatten_mask(mask)
        array([False, False, False, False, False,  True])
    
        
    """
def flatten_structured_array(a):
    """
    
        Flatten a structured array.
    
        The data type of the output is chosen such that it can represent all of the
        (nested) fields.
    
        Parameters
        ----------
        a : structured array
    
        Returns
        -------
        output : masked array or ndarray
            A flattened masked array if the input is a masked array, otherwise a
            standard ndarray.
    
        Examples
        --------
        >>> ndtype = [('a', int), ('b', float)]
        >>> a = np.array([(1, 1), (2, 2)], dtype=ndtype)
        >>> np.ma.flatten_structured_array(a)
        array([[1., 1.],
               [2., 2.]])
    
        
    """
def fromfile(file, dtype = float, count = -1, sep = ''):
    ...
def fromflex(fxarray):
    """
    
        Build a masked array from a suitable flexible-type array.
    
        The input array has to have a data-type with ``_data`` and ``_mask``
        fields. This type of array is output by `MaskedArray.toflex`.
    
        Parameters
        ----------
        fxarray : ndarray
            The structured input array, containing ``_data`` and ``_mask``
            fields. If present, other fields are discarded.
    
        Returns
        -------
        result : MaskedArray
            The constructed masked array.
    
        See Also
        --------
        MaskedArray.toflex : Build a flexible-type array from a masked array.
    
        Examples
        --------
        >>> x = np.ma.array(np.arange(9).reshape(3, 3), mask=[0] + [1, 0] * 4)
        >>> rec = x.toflex()
        >>> rec
        array([[(0, False), (1,  True), (2, False)],
               [(3,  True), (4, False), (5,  True)],
               [(6, False), (7,  True), (8, False)]],
              dtype=[('_data', '<i8'), ('_mask', '?')])
        >>> x2 = np.ma.fromflex(rec)
        >>> x2
        masked_array(
          data=[[0, --, 2],
                [--, 4, --],
                [6, --, 8]],
          mask=[[False,  True, False],
                [ True, False,  True],
                [False,  True, False]],
          fill_value=999999)
    
        Extra fields can be present in the structured array but are discarded:
    
        >>> dt = [('_data', '<i4'), ('_mask', '|b1'), ('field3', '<f4')]
        >>> rec2 = np.zeros((2, 2), dtype=dt)
        >>> rec2
        array([[(0, False, 0.), (0, False, 0.)],
               [(0, False, 0.), (0, False, 0.)]],
              dtype=[('_data', '<i4'), ('_mask', '?'), ('field3', '<f4')])
        >>> y = np.ma.fromflex(rec2)
        >>> y
        masked_array(
          data=[[0, 0],
                [0, 0]],
          mask=[[False, False],
                [False, False]],
          fill_value=999999,
          dtype=int32)
    
        
    """
def get_fill_value(a):
    """
    
        Return the filling value of a, if any.  Otherwise, returns the
        default filling value for that type.
    
        
    """
def get_masked_subclass(*arrays):
    """
    
        Return the youngest subclass of MaskedArray from a list of (masked) arrays.
    
        In case of siblings, the first listed takes over.
    
        
    """
def get_object_signature(obj):
    """
    
        Get the signature from obj
    
        
    """
def getdata(a, subok = True):
    """
    
        Return the data of a masked array as an ndarray.
    
        Return the data of `a` (if any) as an ndarray if `a` is a ``MaskedArray``,
        else return `a` as a ndarray or subclass (depending on `subok`) if not.
    
        Parameters
        ----------
        a : array_like
            Input ``MaskedArray``, alternatively a ndarray or a subclass thereof.
        subok : bool
            Whether to force the output to be a `pure` ndarray (False) or to
            return a subclass of ndarray if appropriate (True, default).
    
        See Also
        --------
        getmask : Return the mask of a masked array, or nomask.
        getmaskarray : Return the mask of a masked array, or full array of False.
    
        Examples
        --------
        >>> import numpy.ma as ma
        >>> a = ma.masked_equal([[1,2],[3,4]], 2)
        >>> a
        masked_array(
          data=[[1, --],
                [3, 4]],
          mask=[[False,  True],
                [False, False]],
          fill_value=2)
        >>> ma.getdata(a)
        array([[1, 2],
               [3, 4]])
    
        Equivalently use the ``MaskedArray`` `data` attribute.
    
        >>> a.data
        array([[1, 2],
               [3, 4]])
    
        
    """
def getmask(a):
    """
    
        Return the mask of a masked array, or nomask.
    
        Return the mask of `a` as an ndarray if `a` is a `MaskedArray` and the
        mask is not `nomask`, else return `nomask`. To guarantee a full array
        of booleans of the same shape as a, use `getmaskarray`.
    
        Parameters
        ----------
        a : array_like
            Input `MaskedArray` for which the mask is required.
    
        See Also
        --------
        getdata : Return the data of a masked array as an ndarray.
        getmaskarray : Return the mask of a masked array, or full array of False.
    
        Examples
        --------
        >>> import numpy.ma as ma
        >>> a = ma.masked_equal([[1,2],[3,4]], 2)
        >>> a
        masked_array(
          data=[[1, --],
                [3, 4]],
          mask=[[False,  True],
                [False, False]],
          fill_value=2)
        >>> ma.getmask(a)
        array([[False,  True],
               [False, False]])
    
        Equivalently use the `MaskedArray` `mask` attribute.
    
        >>> a.mask
        array([[False,  True],
               [False, False]])
    
        Result when mask == `nomask`
    
        >>> b = ma.masked_array([[1,2],[3,4]])
        >>> b
        masked_array(
          data=[[1, 2],
                [3, 4]],
          mask=False,
          fill_value=999999)
        >>> ma.nomask
        False
        >>> ma.getmask(b) == ma.nomask
        True
        >>> b.mask == ma.nomask
        True
    
        
    """
def getmaskarray(arr):
    """
    
        Return the mask of a masked array, or full boolean array of False.
    
        Return the mask of `arr` as an ndarray if `arr` is a `MaskedArray` and
        the mask is not `nomask`, else return a full boolean array of False of
        the same shape as `arr`.
    
        Parameters
        ----------
        arr : array_like
            Input `MaskedArray` for which the mask is required.
    
        See Also
        --------
        getmask : Return the mask of a masked array, or nomask.
        getdata : Return the data of a masked array as an ndarray.
    
        Examples
        --------
        >>> import numpy.ma as ma
        >>> a = ma.masked_equal([[1,2],[3,4]], 2)
        >>> a
        masked_array(
          data=[[1, --],
                [3, 4]],
          mask=[[False,  True],
                [False, False]],
          fill_value=2)
        >>> ma.getmaskarray(a)
        array([[False,  True],
               [False, False]])
    
        Result when mask == ``nomask``
    
        >>> b = ma.masked_array([[1,2],[3,4]])
        >>> b
        masked_array(
          data=[[1, 2],
                [3, 4]],
          mask=False,
          fill_value=999999)
        >>> ma.getmaskarray(b)
        array([[False, False],
               [False, False]])
    
        
    """
def inner(a, b):
    """
    inner(a, b, /)
    
    Inner product of two arrays.
    
    Ordinary inner product of vectors for 1-D arrays (without complex
    conjugation), in higher dimensions a sum product over the last axes.
    
    Parameters
    ----------
    a, b : array_like
        If `a` and `b` are nonscalar, their last dimensions must match.
    
    Returns
    -------
    out : ndarray
        If `a` and `b` are both
        scalars or both 1-D arrays then a scalar is returned; otherwise
        an array is returned.
        ``out.shape = (*a.shape[:-1], *b.shape[:-1])``
    
    Raises
    ------
    ValueError
        If both `a` and `b` are nonscalar and their last dimensions have
        different sizes.
    
    See Also
    --------
    tensordot : Sum products over arbitrary axes.
    dot : Generalised matrix product, using second last dimension of `b`.
    einsum : Einstein summation convention.
    
    Notes
    -----
    Masked values are replaced by 0.
    
    For vectors (1-D arrays) it computes the ordinary inner-product::
    
        np.inner(a, b) = sum(a[:]*b[:])
    
    More generally, if ``ndim(a) = r > 0`` and ``ndim(b) = s > 0``::
    
        np.inner(a, b) = np.tensordot(a, b, axes=(-1,-1))
    
    or explicitly::
    
        np.inner(a, b)[i0,...,ir-2,j0,...,js-2]
             = sum(a[i0,...,ir-2,:]*b[j0,...,js-2,:])
    
    In addition `a` or `b` may be scalars, in which case::
    
       np.inner(a,b) = a*b
    
    Examples
    --------
    Ordinary inner product for vectors:
    
    >>> a = np.array([1,2,3])
    >>> b = np.array([0,1,0])
    >>> np.inner(a, b)
    2
    
    Some multidimensional examples:
    
    >>> a = np.arange(24).reshape((2,3,4))
    >>> b = np.arange(4)
    >>> c = np.inner(a, b)
    >>> c.shape
    (2, 3)
    >>> c
    array([[ 14,  38,  62],
           [ 86, 110, 134]])
    
    >>> a = np.arange(2).reshape((1,1,2))
    >>> b = np.arange(6).reshape((3,2))
    >>> c = np.inner(a, b)
    >>> c.shape
    (1, 1, 3)
    >>> c
    array([[[1, 3, 5]]])
    
    An example where `b` is a scalar:
    
    >>> np.inner(np.eye(2), 7)
    array([[7., 0.],
           [0., 7.]])
    """
def isMaskedArray(x):
    """
    
        Test whether input is an instance of MaskedArray.
    
        This function returns True if `x` is an instance of MaskedArray
        and returns False otherwise.  Any object is accepted as input.
    
        Parameters
        ----------
        x : object
            Object to test.
    
        Returns
        -------
        result : bool
            True if `x` is a MaskedArray.
    
        See Also
        --------
        isMA : Alias to isMaskedArray.
        isarray : Alias to isMaskedArray.
    
        Examples
        --------
        >>> import numpy.ma as ma
        >>> a = np.eye(3, 3)
        >>> a
        array([[ 1.,  0.,  0.],
               [ 0.,  1.,  0.],
               [ 0.,  0.,  1.]])
        >>> m = ma.masked_values(a, 0)
        >>> m
        masked_array(
          data=[[1.0, --, --],
                [--, 1.0, --],
                [--, --, 1.0]],
          mask=[[False,  True,  True],
                [ True, False,  True],
                [ True,  True, False]],
          fill_value=0.0)
        >>> ma.isMaskedArray(a)
        False
        >>> ma.isMaskedArray(m)
        True
        >>> ma.isMaskedArray([0, 1, 2])
        False
    
        
    """
def is_mask(m):
    """
    
        Return True if m is a valid, standard mask.
    
        This function does not check the contents of the input, only that the
        type is MaskType. In particular, this function returns False if the
        mask has a flexible dtype.
    
        Parameters
        ----------
        m : array_like
            Array to test.
    
        Returns
        -------
        result : bool
            True if `m.dtype.type` is MaskType, False otherwise.
    
        See Also
        --------
        ma.isMaskedArray : Test whether input is an instance of MaskedArray.
    
        Examples
        --------
        >>> import numpy.ma as ma
        >>> m = ma.masked_equal([0, 1, 0, 2, 3], 0)
        >>> m
        masked_array(data=[--, 1, --, 2, 3],
                     mask=[ True, False,  True, False, False],
               fill_value=0)
        >>> ma.is_mask(m)
        False
        >>> ma.is_mask(m.mask)
        True
    
        Input must be an ndarray (or have similar attributes)
        for it to be considered a valid mask.
    
        >>> m = [False, True, False]
        >>> ma.is_mask(m)
        False
        >>> m = np.array([False, True, False])
        >>> m
        array([False,  True, False])
        >>> ma.is_mask(m)
        True
    
        Arrays with complex dtypes don't return True.
    
        >>> dtype = np.dtype({'names':['monty', 'pithon'],
        ...                   'formats':[bool, bool]})
        >>> dtype
        dtype([('monty', '|b1'), ('pithon', '|b1')])
        >>> m = np.array([(True, False), (False, True), (True, False)],
        ...              dtype=dtype)
        >>> m
        array([( True, False), (False,  True), ( True, False)],
              dtype=[('monty', '?'), ('pithon', '?')])
        >>> ma.is_mask(m)
        False
    
        
    """
def is_masked(x):
    """
    
        Determine whether input has masked values.
    
        Accepts any object as input, but always returns False unless the
        input is a MaskedArray containing masked values.
    
        Parameters
        ----------
        x : array_like
            Array to check for masked values.
    
        Returns
        -------
        result : bool
            True if `x` is a MaskedArray with masked values, False otherwise.
    
        Examples
        --------
        >>> import numpy.ma as ma
        >>> x = ma.masked_equal([0, 1, 0, 2, 3], 0)
        >>> x
        masked_array(data=[--, 1, --, 2, 3],
                     mask=[ True, False,  True, False, False],
               fill_value=0)
        >>> ma.is_masked(x)
        True
        >>> x = ma.masked_equal([0, 1, 0, 2, 3], 42)
        >>> x
        masked_array(data=[0, 1, 0, 2, 3],
                     mask=False,
               fill_value=42)
        >>> ma.is_masked(x)
        False
    
        Always returns False if `x` isn't a MaskedArray.
    
        >>> x = [False, True, False]
        >>> ma.is_masked(x)
        False
        >>> x = 'a string'
        >>> ma.is_masked(x)
        False
    
        
    """
def is_string_or_list_of_strings(val):
    ...
def left_shift(a, n):
    """
    
        Shift the bits of an integer to the left.
    
        This is the masked array version of `numpy.left_shift`, for details
        see that function.
    
        See Also
        --------
        numpy.left_shift
    
        
    """
def make_mask(m, copy = False, shrink = True, dtype = numpy.bool_):
    """
    
        Create a boolean mask from an array.
    
        Return `m` as a boolean mask, creating a copy if necessary or requested.
        The function can accept any sequence that is convertible to integers,
        or ``nomask``.  Does not require that contents must be 0s and 1s, values
        of 0 are interpreted as False, everything else as True.
    
        Parameters
        ----------
        m : array_like
            Potential mask.
        copy : bool, optional
            Whether to return a copy of `m` (True) or `m` itself (False).
        shrink : bool, optional
            Whether to shrink `m` to ``nomask`` if all its values are False.
        dtype : dtype, optional
            Data-type of the output mask. By default, the output mask has a
            dtype of MaskType (bool). If the dtype is flexible, each field has
            a boolean dtype. This is ignored when `m` is ``nomask``, in which
            case ``nomask`` is always returned.
    
        Returns
        -------
        result : ndarray
            A boolean mask derived from `m`.
    
        Examples
        --------
        >>> import numpy.ma as ma
        >>> m = [True, False, True, True]
        >>> ma.make_mask(m)
        array([ True, False,  True,  True])
        >>> m = [1, 0, 1, 1]
        >>> ma.make_mask(m)
        array([ True, False,  True,  True])
        >>> m = [1, 0, 2, -3]
        >>> ma.make_mask(m)
        array([ True, False,  True,  True])
    
        Effect of the `shrink` parameter.
    
        >>> m = np.zeros(4)
        >>> m
        array([0., 0., 0., 0.])
        >>> ma.make_mask(m)
        False
        >>> ma.make_mask(m, shrink=False)
        array([False, False, False, False])
    
        Using a flexible `dtype`.
    
        >>> m = [1, 0, 1, 1]
        >>> n = [0, 1, 0, 0]
        >>> arr = []
        >>> for man, mouse in zip(m, n):
        ...     arr.append((man, mouse))
        >>> arr
        [(1, 0), (0, 1), (1, 0), (1, 0)]
        >>> dtype = np.dtype({'names':['man', 'mouse'],
        ...                   'formats':[np.int64, np.int64]})
        >>> arr = np.array(arr, dtype=dtype)
        >>> arr
        array([(1, 0), (0, 1), (1, 0), (1, 0)],
              dtype=[('man', '<i8'), ('mouse', '<i8')])
        >>> ma.make_mask(arr, dtype=dtype)
        array([(True, False), (False, True), (True, False), (True, False)],
              dtype=[('man', '|b1'), ('mouse', '|b1')])
    
        
    """
def make_mask_descr(ndtype):
    """
    
        Construct a dtype description list from a given dtype.
    
        Returns a new dtype object, with the type of all fields in `ndtype` to a
        boolean type. Field names are not altered.
    
        Parameters
        ----------
        ndtype : dtype
            The dtype to convert.
    
        Returns
        -------
        result : dtype
            A dtype that looks like `ndtype`, the type of all fields is boolean.
    
        Examples
        --------
        >>> import numpy.ma as ma
        >>> dtype = np.dtype({'names':['foo', 'bar'],
        ...                   'formats':[np.float32, np.int64]})
        >>> dtype
        dtype([('foo', '<f4'), ('bar', '<i8')])
        >>> ma.make_mask_descr(dtype)
        dtype([('foo', '|b1'), ('bar', '|b1')])
        >>> ma.make_mask_descr(np.float32)
        dtype('bool')
    
        
    """
def make_mask_none(newshape, dtype = None):
    """
    
        Return a boolean mask of the given shape, filled with False.
    
        This function returns a boolean ndarray with all entries False, that can
        be used in common mask manipulations. If a complex dtype is specified, the
        type of each field is converted to a boolean type.
    
        Parameters
        ----------
        newshape : tuple
            A tuple indicating the shape of the mask.
        dtype : {None, dtype}, optional
            If None, use a MaskType instance. Otherwise, use a new datatype with
            the same fields as `dtype`, converted to boolean types.
    
        Returns
        -------
        result : ndarray
            An ndarray of appropriate shape and dtype, filled with False.
    
        See Also
        --------
        make_mask : Create a boolean mask from an array.
        make_mask_descr : Construct a dtype description list from a given dtype.
    
        Examples
        --------
        >>> import numpy.ma as ma
        >>> ma.make_mask_none((3,))
        array([False, False, False])
    
        Defining a more complex dtype.
    
        >>> dtype = np.dtype({'names':['foo', 'bar'],
        ...                   'formats':[np.float32, np.int64]})
        >>> dtype
        dtype([('foo', '<f4'), ('bar', '<i8')])
        >>> ma.make_mask_none((3,), dtype=dtype)
        array([(False, False), (False, False), (False, False)],
              dtype=[('foo', '|b1'), ('bar', '|b1')])
    
        
    """
def mask_or(m1, m2, copy = False, shrink = True):
    """
    
        Combine two masks with the ``logical_or`` operator.
    
        The result may be a view on `m1` or `m2` if the other is `nomask`
        (i.e. False).
    
        Parameters
        ----------
        m1, m2 : array_like
            Input masks.
        copy : bool, optional
            If copy is False and one of the inputs is `nomask`, return a view
            of the other input mask. Defaults to False.
        shrink : bool, optional
            Whether to shrink the output to `nomask` if all its values are
            False. Defaults to True.
    
        Returns
        -------
        mask : output mask
            The result masks values that are masked in either `m1` or `m2`.
    
        Raises
        ------
        ValueError
            If `m1` and `m2` have different flexible dtypes.
    
        Examples
        --------
        >>> m1 = np.ma.make_mask([0, 1, 1, 0])
        >>> m2 = np.ma.make_mask([1, 0, 0, 0])
        >>> np.ma.mask_or(m1, m2)
        array([ True,  True,  True, False])
    
        
    """
def masked_equal(x, value, copy = True):
    """
    
        Mask an array where equal to a given value.
    
        Return a MaskedArray, masked where the data in array `x` are
        equal to `value`. The fill_value of the returned MaskedArray
        is set to `value`.
    
        For floating point arrays, consider using ``masked_values(x, value)``.
    
        See Also
        --------
        masked_where : Mask where a condition is met.
        masked_values : Mask using floating point equality.
    
        Examples
        --------
        >>> import numpy.ma as ma
        >>> a = np.arange(4)
        >>> a
        array([0, 1, 2, 3])
        >>> ma.masked_equal(a, 2)
        masked_array(data=[0, 1, --, 3],
                     mask=[False, False,  True, False],
               fill_value=2)
    
        
    """
def masked_greater(x, value, copy = True):
    """
    
        Mask an array where greater than a given value.
    
        This function is a shortcut to ``masked_where``, with
        `condition` = (x > value).
    
        See Also
        --------
        masked_where : Mask where a condition is met.
    
        Examples
        --------
        >>> import numpy.ma as ma
        >>> a = np.arange(4)
        >>> a
        array([0, 1, 2, 3])
        >>> ma.masked_greater(a, 2)
        masked_array(data=[0, 1, 2, --],
                     mask=[False, False, False,  True],
               fill_value=999999)
    
        
    """
def masked_greater_equal(x, value, copy = True):
    """
    
        Mask an array where greater than or equal to a given value.
    
        This function is a shortcut to ``masked_where``, with
        `condition` = (x >= value).
    
        See Also
        --------
        masked_where : Mask where a condition is met.
    
        Examples
        --------
        >>> import numpy.ma as ma
        >>> a = np.arange(4)
        >>> a
        array([0, 1, 2, 3])
        >>> ma.masked_greater_equal(a, 2)
        masked_array(data=[0, 1, --, --],
                     mask=[False, False,  True,  True],
               fill_value=999999)
    
        
    """
def masked_inside(x, v1, v2, copy = True):
    """
    
        Mask an array inside a given interval.
    
        Shortcut to ``masked_where``, where `condition` is True for `x` inside
        the interval [v1,v2] (v1 <= x <= v2).  The boundaries `v1` and `v2`
        can be given in either order.
    
        See Also
        --------
        masked_where : Mask where a condition is met.
    
        Notes
        -----
        The array `x` is prefilled with its filling value.
    
        Examples
        --------
        >>> import numpy.ma as ma
        >>> x = [0.31, 1.2, 0.01, 0.2, -0.4, -1.1]
        >>> ma.masked_inside(x, -0.3, 0.3)
        masked_array(data=[0.31, 1.2, --, --, -0.4, -1.1],
                     mask=[False, False,  True,  True, False, False],
               fill_value=1e+20)
    
        The order of `v1` and `v2` doesn't matter.
    
        >>> ma.masked_inside(x, 0.3, -0.3)
        masked_array(data=[0.31, 1.2, --, --, -0.4, -1.1],
                     mask=[False, False,  True,  True, False, False],
               fill_value=1e+20)
    
        
    """
def masked_invalid(a, copy = True):
    """
    
        Mask an array where invalid values occur (NaNs or infs).
    
        This function is a shortcut to ``masked_where``, with
        `condition` = ~(np.isfinite(a)). Any pre-existing mask is conserved.
        Only applies to arrays with a dtype where NaNs or infs make sense
        (i.e. floating point types), but accepts any array_like object.
    
        See Also
        --------
        masked_where : Mask where a condition is met.
    
        Examples
        --------
        >>> import numpy.ma as ma
        >>> a = np.arange(5, dtype=float)
        >>> a[2] = np.NaN
        >>> a[3] = np.PINF
        >>> a
        array([ 0.,  1., nan, inf,  4.])
        >>> ma.masked_invalid(a)
        masked_array(data=[0.0, 1.0, --, --, 4.0],
                     mask=[False, False,  True,  True, False],
               fill_value=1e+20)
    
        
    """
def masked_less(x, value, copy = True):
    """
    
        Mask an array where less than a given value.
    
        This function is a shortcut to ``masked_where``, with
        `condition` = (x < value).
    
        See Also
        --------
        masked_where : Mask where a condition is met.
    
        Examples
        --------
        >>> import numpy.ma as ma
        >>> a = np.arange(4)
        >>> a
        array([0, 1, 2, 3])
        >>> ma.masked_less(a, 2)
        masked_array(data=[--, --, 2, 3],
                     mask=[ True,  True, False, False],
               fill_value=999999)
    
        
    """
def masked_less_equal(x, value, copy = True):
    """
    
        Mask an array where less than or equal to a given value.
    
        This function is a shortcut to ``masked_where``, with
        `condition` = (x <= value).
    
        See Also
        --------
        masked_where : Mask where a condition is met.
    
        Examples
        --------
        >>> import numpy.ma as ma
        >>> a = np.arange(4)
        >>> a
        array([0, 1, 2, 3])
        >>> ma.masked_less_equal(a, 2)
        masked_array(data=[--, --, --, 3],
                     mask=[ True,  True,  True, False],
               fill_value=999999)
    
        
    """
def masked_not_equal(x, value, copy = True):
    """
    
        Mask an array where `not` equal to a given value.
    
        This function is a shortcut to ``masked_where``, with
        `condition` = (x != value).
    
        See Also
        --------
        masked_where : Mask where a condition is met.
    
        Examples
        --------
        >>> import numpy.ma as ma
        >>> a = np.arange(4)
        >>> a
        array([0, 1, 2, 3])
        >>> ma.masked_not_equal(a, 2)
        masked_array(data=[--, --, 2, --],
                     mask=[ True,  True, False,  True],
               fill_value=999999)
    
        
    """
def masked_object(x, value, copy = True, shrink = True):
    """
    
        Mask the array `x` where the data are exactly equal to value.
    
        This function is similar to `masked_values`, but only suitable
        for object arrays: for floating point, use `masked_values` instead.
    
        Parameters
        ----------
        x : array_like
            Array to mask
        value : object
            Comparison value
        copy : {True, False}, optional
            Whether to return a copy of `x`.
        shrink : {True, False}, optional
            Whether to collapse a mask full of False to nomask
    
        Returns
        -------
        result : MaskedArray
            The result of masking `x` where equal to `value`.
    
        See Also
        --------
        masked_where : Mask where a condition is met.
        masked_equal : Mask where equal to a given value (integers).
        masked_values : Mask using floating point equality.
    
        Examples
        --------
        >>> import numpy.ma as ma
        >>> food = np.array(['green_eggs', 'ham'], dtype=object)
        >>> # don't eat spoiled food
        >>> eat = ma.masked_object(food, 'green_eggs')
        >>> eat
        masked_array(data=[--, 'ham'],
                     mask=[ True, False],
               fill_value='green_eggs',
                    dtype=object)
        >>> # plain ol` ham is boring
        >>> fresh_food = np.array(['cheese', 'ham', 'pineapple'], dtype=object)
        >>> eat = ma.masked_object(fresh_food, 'green_eggs')
        >>> eat
        masked_array(data=['cheese', 'ham', 'pineapple'],
                     mask=False,
               fill_value='green_eggs',
                    dtype=object)
    
        Note that `mask` is set to ``nomask`` if possible.
    
        >>> eat
        masked_array(data=['cheese', 'ham', 'pineapple'],
                     mask=False,
               fill_value='green_eggs',
                    dtype=object)
    
        
    """
def masked_outside(x, v1, v2, copy = True):
    """
    
        Mask an array outside a given interval.
    
        Shortcut to ``masked_where``, where `condition` is True for `x` outside
        the interval [v1,v2] (x < v1)|(x > v2).
        The boundaries `v1` and `v2` can be given in either order.
    
        See Also
        --------
        masked_where : Mask where a condition is met.
    
        Notes
        -----
        The array `x` is prefilled with its filling value.
    
        Examples
        --------
        >>> import numpy.ma as ma
        >>> x = [0.31, 1.2, 0.01, 0.2, -0.4, -1.1]
        >>> ma.masked_outside(x, -0.3, 0.3)
        masked_array(data=[--, --, 0.01, 0.2, --, --],
                     mask=[ True,  True, False, False,  True,  True],
               fill_value=1e+20)
    
        The order of `v1` and `v2` doesn't matter.
    
        >>> ma.masked_outside(x, 0.3, -0.3)
        masked_array(data=[--, --, 0.01, 0.2, --, --],
                     mask=[ True,  True, False, False,  True,  True],
               fill_value=1e+20)
    
        
    """
def masked_values(x, value, rtol = 1e-05, atol = 1e-08, copy = True, shrink = True):
    """
    
        Mask using floating point equality.
    
        Return a MaskedArray, masked where the data in array `x` are approximately
        equal to `value`, determined using `isclose`. The default tolerances for
        `masked_values` are the same as those for `isclose`.
    
        For integer types, exact equality is used, in the same way as
        `masked_equal`.
    
        The fill_value is set to `value` and the mask is set to ``nomask`` if
        possible.
    
        Parameters
        ----------
        x : array_like
            Array to mask.
        value : float
            Masking value.
        rtol, atol : float, optional
            Tolerance parameters passed on to `isclose`
        copy : bool, optional
            Whether to return a copy of `x`.
        shrink : bool, optional
            Whether to collapse a mask full of False to ``nomask``.
    
        Returns
        -------
        result : MaskedArray
            The result of masking `x` where approximately equal to `value`.
    
        See Also
        --------
        masked_where : Mask where a condition is met.
        masked_equal : Mask where equal to a given value (integers).
    
        Examples
        --------
        >>> import numpy.ma as ma
        >>> x = np.array([1, 1.1, 2, 1.1, 3])
        >>> ma.masked_values(x, 1.1)
        masked_array(data=[1.0, --, 2.0, --, 3.0],
                     mask=[False,  True, False,  True, False],
               fill_value=1.1)
    
        Note that `mask` is set to ``nomask`` if possible.
    
        >>> ma.masked_values(x, 2.1)
        masked_array(data=[1. , 1.1, 2. , 1.1, 3. ],
                     mask=False,
               fill_value=2.1)
    
        Unlike `masked_equal`, `masked_values` can perform approximate equalities.
    
        >>> ma.masked_values(x, 2.1, atol=1e-1)
        masked_array(data=[1.0, 1.1, --, 1.1, 3.0],
                     mask=[False, False,  True, False, False],
               fill_value=2.1)
    
        
    """
def masked_where(condition, a, copy = True):
    """
    
        Mask an array where a condition is met.
    
        Return `a` as an array masked where `condition` is True.
        Any masked values of `a` or `condition` are also masked in the output.
    
        Parameters
        ----------
        condition : array_like
            Masking condition.  When `condition` tests floating point values for
            equality, consider using ``masked_values`` instead.
        a : array_like
            Array to mask.
        copy : bool
            If True (default) make a copy of `a` in the result.  If False modify
            `a` in place and return a view.
    
        Returns
        -------
        result : MaskedArray
            The result of masking `a` where `condition` is True.
    
        See Also
        --------
        masked_values : Mask using floating point equality.
        masked_equal : Mask where equal to a given value.
        masked_not_equal : Mask where `not` equal to a given value.
        masked_less_equal : Mask where less than or equal to a given value.
        masked_greater_equal : Mask where greater than or equal to a given value.
        masked_less : Mask where less than a given value.
        masked_greater : Mask where greater than a given value.
        masked_inside : Mask inside a given interval.
        masked_outside : Mask outside a given interval.
        masked_invalid : Mask invalid values (NaNs or infs).
    
        Examples
        --------
        >>> import numpy.ma as ma
        >>> a = np.arange(4)
        >>> a
        array([0, 1, 2, 3])
        >>> ma.masked_where(a <= 2, a)
        masked_array(data=[--, --, --, 3],
                     mask=[ True,  True,  True, False],
               fill_value=999999)
    
        Mask array `b` conditional on `a`.
    
        >>> b = ['a', 'b', 'c', 'd']
        >>> ma.masked_where(a == 2, b)
        masked_array(data=['a', 'b', --, 'd'],
                     mask=[False, False,  True, False],
               fill_value='N/A',
                    dtype='<U1')
    
        Effect of the `copy` argument.
    
        >>> c = ma.masked_where(a <= 2, a)
        >>> c
        masked_array(data=[--, --, --, 3],
                     mask=[ True,  True,  True, False],
               fill_value=999999)
        >>> c[0] = 99
        >>> c
        masked_array(data=[99, --, --, 3],
                     mask=[False,  True,  True, False],
               fill_value=999999)
        >>> a
        array([0, 1, 2, 3])
        >>> c = ma.masked_where(a <= 2, a, copy=False)
        >>> c[0] = 99
        >>> c
        masked_array(data=[99, --, --, 3],
                     mask=[False,  True,  True, False],
               fill_value=999999)
        >>> a
        array([99,  1,  2,  3])
    
        When `condition` or `a` contain masked values.
    
        >>> a = np.arange(4)
        >>> a = ma.masked_where(a == 2, a)
        >>> a
        masked_array(data=[0, 1, --, 3],
                     mask=[False, False,  True, False],
               fill_value=999999)
        >>> b = np.arange(4)
        >>> b = ma.masked_where(b == 0, b)
        >>> b
        masked_array(data=[--, 1, 2, 3],
                     mask=[ True, False, False, False],
               fill_value=999999)
        >>> ma.masked_where(a == 3, b)
        masked_array(data=[--, 1, --, --],
                     mask=[ True, False,  True,  True],
               fill_value=999999)
    
        
    """
def max(obj, axis = None, out = None, fill_value = None, keepdims = ...):
    """
    
            Return the maximum along a given axis.
    
            Parameters
            ----------
            axis : None or int or tuple of ints, optional
                Axis along which to operate.  By default, ``axis`` is None and the
                flattened input is used.
                .. versionadded:: 1.7.0
                If this is a tuple of ints, the maximum is selected over multiple
                axes, instead of a single axis or all the axes as before.
            out : array_like, optional
                Alternative output array in which to place the result.  Must
                be of the same shape and buffer length as the expected output.
            fill_value : scalar or None, optional
                Value used to fill in the masked values.
                If None, use the output of maximum_fill_value().
            keepdims : bool, optional
                If this is set to True, the axes which are reduced are left
                in the result as dimensions with size one. With this option,
                the result will broadcast correctly against the array.
    
            Returns
            -------
            amax : array_like
                New array holding the result.
                If ``out`` was specified, ``out`` is returned.
    
            See Also
            --------
            ma.maximum_fill_value
                Returns the maximum filling value for a given datatype.
    
            Examples
            --------
            >>> import numpy.ma as ma
            >>> x = [[-1., 2.5], [4., -2.], [3., 0.]]
            >>> mask = [[0, 0], [1, 0], [1, 0]]
            >>> masked_x = ma.masked_array(x, mask)
            >>> masked_x
            masked_array(
              data=[[-1.0, 2.5],
                    [--, -2.0],
                    [--, 0.0]],
              mask=[[False, False],
                    [ True, False],
                    [ True, False]],
              fill_value=1e+20)
            >>> ma.max(masked_x)
            2.5
            >>> ma.max(masked_x, axis=0)
            masked_array(data=[-1.0, 2.5],
                         mask=[False, False],
                   fill_value=1e+20)
            >>> ma.max(masked_x, axis=1, keepdims=True)
            masked_array(
              data=[[2.5],
                    [-2.0],
                    [0.0]],
              mask=[[False],
                    [False],
                    [False]],
              fill_value=1e+20)
            >>> mask = [[1, 1], [1, 1], [1, 1]]
            >>> masked_x = ma.masked_array(x, mask)
            >>> ma.max(masked_x, axis=1)
            masked_array(data=[--, --, --],
                         mask=[ True,  True,  True],
                   fill_value=1e+20,
                        dtype=float64)
            
    """
def maximum_fill_value(obj):
    """
    
        Return the minimum value that can be represented by the dtype of an object.
    
        This function is useful for calculating a fill value suitable for
        taking the maximum of an array with a given dtype.
    
        Parameters
        ----------
        obj : ndarray, dtype or scalar
            An object that can be queried for it's numeric type.
    
        Returns
        -------
        val : scalar
            The minimum representable value.
    
        Raises
        ------
        TypeError
            If `obj` isn't a suitable numeric type.
    
        See Also
        --------
        minimum_fill_value : The inverse function.
        set_fill_value : Set the filling value of a masked array.
        MaskedArray.fill_value : Return current fill value.
    
        Examples
        --------
        >>> import numpy.ma as ma
        >>> a = np.int8()
        >>> ma.maximum_fill_value(a)
        -128
        >>> a = np.int32()
        >>> ma.maximum_fill_value(a)
        -2147483648
    
        An array of numeric data can also be passed.
    
        >>> a = np.array([1, 2, 3], dtype=np.int8)
        >>> ma.maximum_fill_value(a)
        -128
        >>> a = np.array([1, 2, 3], dtype=np.float32)
        >>> ma.maximum_fill_value(a)
        -inf
    
        
    """
def min(obj, axis = None, out = None, fill_value = None, keepdims = ...):
    """
    
            Return the minimum along a given axis.
    
            Parameters
            ----------
            axis : None or int or tuple of ints, optional
                Axis along which to operate.  By default, ``axis`` is None and the
                flattened input is used.
                .. versionadded:: 1.7.0
                If this is a tuple of ints, the minimum is selected over multiple
                axes, instead of a single axis or all the axes as before.
            out : array_like, optional
                Alternative output array in which to place the result.  Must be of
                the same shape and buffer length as the expected output.
            fill_value : scalar or None, optional
                Value used to fill in the masked values.
                If None, use the output of `minimum_fill_value`.
            keepdims : bool, optional
                If this is set to True, the axes which are reduced are left
                in the result as dimensions with size one. With this option,
                the result will broadcast correctly against the array.
    
            Returns
            -------
            amin : array_like
                New array holding the result.
                If ``out`` was specified, ``out`` is returned.
    
            See Also
            --------
            ma.minimum_fill_value
                Returns the minimum filling value for a given datatype.
    
            Examples
            --------
            >>> import numpy.ma as ma
            >>> x = [[1., -2., 3.], [0.2, -0.7, 0.1]]
            >>> mask = [[1, 1, 0], [0, 0, 1]]
            >>> masked_x = ma.masked_array(x, mask)
            >>> masked_x
            masked_array(
              data=[[--, --, 3.0],
                    [0.2, -0.7, --]],
              mask=[[ True,  True, False],
                    [False, False,  True]],
              fill_value=1e+20)
            >>> ma.min(masked_x)
            -0.7
            >>> ma.min(masked_x, axis=-1)
            masked_array(data=[3.0, -0.7],
                         mask=[False, False],
                    fill_value=1e+20)
            >>> ma.min(masked_x, axis=0, keepdims=True)
            masked_array(data=[[0.2, -0.7, 3.0]],
                         mask=[[False, False, False]],
                    fill_value=1e+20)
            >>> mask = [[1, 1, 1,], [1, 1, 1]]
            >>> masked_x = ma.masked_array(x, mask)
            >>> ma.min(masked_x, axis=0)
            masked_array(data=[--, --, --],
                         mask=[ True,  True,  True],
                    fill_value=1e+20,
                        dtype=float64)
            
    """
def minimum_fill_value(obj):
    """
    
        Return the maximum value that can be represented by the dtype of an object.
    
        This function is useful for calculating a fill value suitable for
        taking the minimum of an array with a given dtype.
    
        Parameters
        ----------
        obj : ndarray, dtype or scalar
            An object that can be queried for it's numeric type.
    
        Returns
        -------
        val : scalar
            The maximum representable value.
    
        Raises
        ------
        TypeError
            If `obj` isn't a suitable numeric type.
    
        See Also
        --------
        maximum_fill_value : The inverse function.
        set_fill_value : Set the filling value of a masked array.
        MaskedArray.fill_value : Return current fill value.
    
        Examples
        --------
        >>> import numpy.ma as ma
        >>> a = np.int8()
        >>> ma.minimum_fill_value(a)
        127
        >>> a = np.int32()
        >>> ma.minimum_fill_value(a)
        2147483647
    
        An array of numeric data can also be passed.
    
        >>> a = np.array([1, 2, 3], dtype=np.int8)
        >>> ma.minimum_fill_value(a)
        127
        >>> a = np.array([1, 2, 3], dtype=np.float32)
        >>> ma.minimum_fill_value(a)
        inf
    
        
    """
def ndim(obj):
    """
    
        Return the number of dimensions of an array.
    
        Parameters
        ----------
        a : array_like
            Input array.  If it is not already an ndarray, a conversion is
            attempted.
    
        Returns
        -------
        number_of_dimensions : int
            The number of dimensions in `a`.  Scalars are zero-dimensional.
    
        See Also
        --------
        ndarray.ndim : equivalent method
        shape : dimensions of array
        ndarray.shape : dimensions of array
    
        Examples
        --------
        >>> np.ndim([[1,2,3],[4,5,6]])
        2
        >>> np.ndim(np.array([[1,2,3],[4,5,6]]))
        2
        >>> np.ndim(1)
        0
    
        
    """
def outer(a, b):
    """
    Compute the outer product of two vectors.
    
    Given two vectors `a` and `b` of length ``M`` and ``N``, repsectively,
    the outer product [1]_ is::
    
      [[a_0*b_0  a_0*b_1 ... a_0*b_{N-1} ]
       [a_1*b_0    .
       [ ...          .
       [a_{M-1}*b_0            a_{M-1}*b_{N-1} ]]
    
    Parameters
    ----------
    a : (M,) array_like
        First input vector.  Input is flattened if
        not already 1-dimensional.
    b : (N,) array_like
        Second input vector.  Input is flattened if
        not already 1-dimensional.
    out : (M, N) ndarray, optional
        A location where the result is stored
    
        .. versionadded:: 1.9.0
    
    Returns
    -------
    out : (M, N) ndarray
        ``out[i, j] = a[i] * b[j]``
    
    See also
    --------
    inner
    einsum : ``einsum('i,j->ij', a.ravel(), b.ravel())`` is the equivalent.
    ufunc.outer : A generalization to dimensions other than 1D and other
                  operations. ``np.multiply.outer(a.ravel(), b.ravel())``
                  is the equivalent.
    tensordot : ``np.tensordot(a.ravel(), b.ravel(), axes=((), ()))``
                is the equivalent.
    
    References
    ----------
    .. [1] G. H. Golub and C. F. Van Loan, *Matrix Computations*, 3rd
           ed., Baltimore, MD, Johns Hopkins University Press, 1996,
           pg. 8.
    
    Examples
    --------
    Make a (*very* coarse) grid for computing a Mandelbrot set:
    
    >>> rl = np.outer(np.ones((5,)), np.linspace(-2, 2, 5))
    >>> rl
    array([[-2., -1.,  0.,  1.,  2.],
           [-2., -1.,  0.,  1.,  2.],
           [-2., -1.,  0.,  1.,  2.],
           [-2., -1.,  0.,  1.,  2.],
           [-2., -1.,  0.,  1.,  2.]])
    >>> im = np.outer(1j*np.linspace(2, -2, 5), np.ones((5,)))
    >>> im
    array([[0.+2.j, 0.+2.j, 0.+2.j, 0.+2.j, 0.+2.j],
           [0.+1.j, 0.+1.j, 0.+1.j, 0.+1.j, 0.+1.j],
           [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
           [0.-1.j, 0.-1.j, 0.-1.j, 0.-1.j, 0.-1.j],
           [0.-2.j, 0.-2.j, 0.-2.j, 0.-2.j, 0.-2.j]])
    >>> grid = rl + im
    >>> grid
    array([[-2.+2.j, -1.+2.j,  0.+2.j,  1.+2.j,  2.+2.j],
           [-2.+1.j, -1.+1.j,  0.+1.j,  1.+1.j,  2.+1.j],
           [-2.+0.j, -1.+0.j,  0.+0.j,  1.+0.j,  2.+0.j],
           [-2.-1.j, -1.-1.j,  0.-1.j,  1.-1.j,  2.-1.j],
           [-2.-2.j, -1.-2.j,  0.-2.j,  1.-2.j,  2.-2.j]])
    
    An example using a "vector" of letters:
    
    >>> x = np.array(['a', 'b', 'c'], dtype=object)
    >>> np.outer(x, [1, 2, 3])
    array([['a', 'aa', 'aaa'],
           ['b', 'bb', 'bbb'],
           ['c', 'cc', 'ccc']], dtype=object)
    
    Notes
    -----
    Masked values are replaced by 0.
    """
def power(a, b, third = None):
    """
    
        Returns element-wise base array raised to power from second array.
    
        This is the masked array version of `numpy.power`. For details see
        `numpy.power`.
    
        See Also
        --------
        numpy.power
    
        Notes
        -----
        The *out* argument to `numpy.power` is not supported, `third` has to be
        None.
    
        Examples
        --------
        >>> import numpy.ma as ma
        >>> x = [11.2, -3.973, 0.801, -1.41]
        >>> mask = [0, 0, 0, 1]
        >>> masked_x = ma.masked_array(x, mask)
        >>> masked_x
        masked_array(data=[11.2, -3.973, 0.801, --],
                 mask=[False, False, False,  True],
           fill_value=1e+20)
        >>> ma.power(masked_x, 2)
        masked_array(data=[125.43999999999998, 15.784728999999999,
                       0.6416010000000001, --],
                 mask=[False, False, False,  True],
           fill_value=1e+20)
        >>> y = [-0.5, 2, 0, 17]
        >>> masked_y = ma.masked_array(y, mask)
        >>> masked_y
        masked_array(data=[-0.5, 2.0, 0.0, --],
                 mask=[False, False, False,  True],
           fill_value=1e+20)
        >>> ma.power(masked_x, masked_y)
        masked_array(data=[0.29880715233359845, 15.784728999999999, 1.0, --],
                 mask=[False, False, False,  True],
           fill_value=1e+20)
    
        
    """
def ptp(obj, axis = None, out = None, fill_value = None, keepdims = ...):
    """
    
            Return (maximum - minimum) along the given dimension
            (i.e. peak-to-peak value).
    
            .. warning::
                `ptp` preserves the data type of the array. This means the
                return value for an input of signed integers with n bits
                (e.g. `np.int8`, `np.int16`, etc) is also a signed integer
                with n bits.  In that case, peak-to-peak values greater than
                ``2**(n-1)-1`` will be returned as negative values. An example
                with a work-around is shown below.
    
            Parameters
            ----------
            axis : {None, int}, optional
                Axis along which to find the peaks.  If None (default) the
                flattened array is used.
            out : {None, array_like}, optional
                Alternative output array in which to place the result. It must
                have the same shape and buffer length as the expected output
                but the type will be cast if necessary.
            fill_value : scalar or None, optional
                Value used to fill in the masked values.
            keepdims : bool, optional
                If this is set to True, the axes which are reduced are left
                in the result as dimensions with size one. With this option,
                the result will broadcast correctly against the array.
    
            Returns
            -------
            ptp : ndarray.
                A new array holding the result, unless ``out`` was
                specified, in which case a reference to ``out`` is returned.
    
            Examples
            --------
            >>> x = np.ma.MaskedArray([[4, 9, 2, 10],
            ...                        [6, 9, 7, 12]])
    
            >>> x.ptp(axis=1)
            masked_array(data=[8, 6],
                         mask=False,
                   fill_value=999999)
    
            >>> x.ptp(axis=0)
            masked_array(data=[2, 0, 5, 2],
                         mask=False,
                   fill_value=999999)
    
            >>> x.ptp()
            10
    
            This example shows that a negative value can be returned when
            the input is an array of signed integers.
    
            >>> y = np.ma.MaskedArray([[1, 127],
            ...                        [0, 127],
            ...                        [-1, 127],
            ...                        [-2, 127]], dtype=np.int8)
            >>> y.ptp(axis=1)
            masked_array(data=[ 126,  127, -128, -127],
                         mask=False,
                   fill_value=999999,
                        dtype=int8)
    
            A work-around is to use the `view()` method to view the result as
            unsigned integers with the same bit width:
    
            >>> y.ptp(axis=1).view(np.uint8)
            masked_array(data=[126, 127, 128, 129],
                         mask=False,
                   fill_value=999999,
                        dtype=uint8)
            
    """
def put(a, indices, values, mode = 'raise'):
    """
    
        Set storage-indexed locations to corresponding values.
    
        This function is equivalent to `MaskedArray.put`, see that method
        for details.
    
        See Also
        --------
        MaskedArray.put
    
        
    """
def putmask(a, mask, values):
    """
    
        Changes elements of an array based on conditional and input values.
    
        This is the masked array version of `numpy.putmask`, for details see
        `numpy.putmask`.
    
        See Also
        --------
        numpy.putmask
    
        Notes
        -----
        Using a masked array as `values` will **not** transform a `ndarray` into
        a `MaskedArray`.
    
        
    """
def reshape(a, new_shape, order = 'C'):
    """
    
        Returns an array containing the same data with a new shape.
    
        Refer to `MaskedArray.reshape` for full documentation.
    
        See Also
        --------
        MaskedArray.reshape : equivalent function
    
        
    """
def resize(x, new_shape):
    """
    
        Return a new masked array with the specified size and shape.
    
        This is the masked equivalent of the `numpy.resize` function. The new
        array is filled with repeated copies of `x` (in the order that the
        data are stored in memory). If `x` is masked, the new array will be
        masked, and the new mask will be a repetition of the old one.
    
        See Also
        --------
        numpy.resize : Equivalent function in the top level NumPy module.
    
        Examples
        --------
        >>> import numpy.ma as ma
        >>> a = ma.array([[1, 2] ,[3, 4]])
        >>> a[0, 1] = ma.masked
        >>> a
        masked_array(
          data=[[1, --],
                [3, 4]],
          mask=[[False,  True],
                [False, False]],
          fill_value=999999)
        >>> np.resize(a, (3, 3))
        masked_array(
          data=[[1, 2, 3],
                [4, 1, 2],
                [3, 4, 1]],
          mask=False,
          fill_value=999999)
        >>> ma.resize(a, (3, 3))
        masked_array(
          data=[[1, --, 3],
                [4, 1, --],
                [3, 4, 1]],
          mask=[[False,  True, False],
                [False, False,  True],
                [False, False, False]],
          fill_value=999999)
    
        A MaskedArray is always returned, regardless of the input type.
    
        >>> a = np.array([[1, 2] ,[3, 4]])
        >>> ma.resize(a, (3, 3))
        masked_array(
          data=[[1, 2, 3],
                [4, 1, 2],
                [3, 4, 1]],
          mask=False,
          fill_value=999999)
    
        
    """
def right_shift(a, n):
    """
    
        Shift the bits of an integer to the right.
    
        This is the masked array version of `numpy.right_shift`, for details
        see that function.
    
        See Also
        --------
        numpy.right_shift
    
        Examples
        --------
        >>> import numpy.ma as ma
        >>> x = [11, 3, 8, 1]
        >>> mask = [0, 0, 0, 1]
        >>> masked_x = ma.masked_array(x, mask)
        >>> masked_x
        masked_array(data=[11, 3, 8, --],
                     mask=[False, False, False,  True],
               fill_value=999999)
        >>> ma.right_shift(masked_x,1)
        masked_array(data=[5, 1, 4, --],
                     mask=[False, False, False,  True],
               fill_value=999999)
    
        
    """
def round_(a, decimals = 0, out = None):
    """
    
        Return a copy of a, rounded to 'decimals' places.
    
        When 'decimals' is negative, it specifies the number of positions
        to the left of the decimal point.  The real and imaginary parts of
        complex numbers are rounded separately. Nothing is done if the
        array is not of float type and 'decimals' is greater than or equal
        to 0.
    
        Parameters
        ----------
        decimals : int
            Number of decimals to round to. May be negative.
        out : array_like
            Existing array to use for output.
            If not given, returns a default copy of a.
    
        Notes
        -----
        If out is given and does not have a mask attribute, the mask of a
        is lost!
    
        Examples
        --------
        >>> import numpy.ma as ma
        >>> x = [11.2, -3.973, 0.801, -1.41]
        >>> mask = [0, 0, 0, 1]
        >>> masked_x = ma.masked_array(x, mask)
        >>> masked_x
        masked_array(data=[11.2, -3.973, 0.801, --],
                     mask=[False, False, False, True],
            fill_value=1e+20)
        >>> ma.round_(masked_x)
        masked_array(data=[11.0, -4.0, 1.0, --],
                     mask=[False, False, False, True],
            fill_value=1e+20)
        >>> ma.round(masked_x, decimals=1)
        masked_array(data=[11.2, -4.0, 0.8, --],
                     mask=[False, False, False, True],
            fill_value=1e+20)
        >>> ma.round_(masked_x, decimals=-1)
        masked_array(data=[10.0, -0.0, 0.0, --],
                     mask=[False, False, False, True],
            fill_value=1e+20)
        
    """
def set_fill_value(a, fill_value):
    """
    
        Set the filling value of a, if a is a masked array.
    
        This function changes the fill value of the masked array `a` in place.
        If `a` is not a masked array, the function returns silently, without
        doing anything.
    
        Parameters
        ----------
        a : array_like
            Input array.
        fill_value : dtype
            Filling value. A consistency test is performed to make sure
            the value is compatible with the dtype of `a`.
    
        Returns
        -------
        None
            Nothing returned by this function.
    
        See Also
        --------
        maximum_fill_value : Return the default fill value for a dtype.
        MaskedArray.fill_value : Return current fill value.
        MaskedArray.set_fill_value : Equivalent method.
    
        Examples
        --------
        >>> import numpy.ma as ma
        >>> a = np.arange(5)
        >>> a
        array([0, 1, 2, 3, 4])
        >>> a = ma.masked_where(a < 3, a)
        >>> a
        masked_array(data=[--, --, --, 3, 4],
                     mask=[ True,  True,  True, False, False],
               fill_value=999999)
        >>> ma.set_fill_value(a, -999)
        >>> a
        masked_array(data=[--, --, --, 3, 4],
                     mask=[ True,  True,  True, False, False],
               fill_value=-999)
    
        Nothing happens if `a` is not a masked array.
    
        >>> a = list(range(5))
        >>> a
        [0, 1, 2, 3, 4]
        >>> ma.set_fill_value(a, 100)
        >>> a
        [0, 1, 2, 3, 4]
        >>> a = np.arange(5)
        >>> a
        array([0, 1, 2, 3, 4])
        >>> ma.set_fill_value(a, 100)
        >>> a
        array([0, 1, 2, 3, 4])
    
        
    """
def shape(obj):
    """
    
        Return the shape of an array.
    
        Parameters
        ----------
        a : array_like
            Input array.
    
        Returns
        -------
        shape : tuple of ints
            The elements of the shape tuple give the lengths of the
            corresponding array dimensions.
    
        See Also
        --------
        len : ``len(a)`` is equivalent to ``np.shape(a)[0]`` for N-D arrays with
              ``N>=1``.
        ndarray.shape : Equivalent array method.
    
        Examples
        --------
        >>> np.shape(np.eye(3))
        (3, 3)
        >>> np.shape([[1, 3]])
        (1, 2)
        >>> np.shape([0])
        (1,)
        >>> np.shape(0)
        ()
    
        >>> a = np.array([(1, 2), (3, 4), (5, 6)],
        ...              dtype=[('x', 'i4'), ('y', 'i4')])
        >>> np.shape(a)
        (3,)
        >>> a.shape
        (3,)
    
        
    """
def size(obj, axis = None):
    """
    
        Return the number of elements along a given axis.
    
        Parameters
        ----------
        a : array_like
            Input data.
        axis : int, optional
            Axis along which the elements are counted.  By default, give
            the total number of elements.
    
        Returns
        -------
        element_count : int
            Number of elements along the specified axis.
    
        See Also
        --------
        shape : dimensions of array
        ndarray.shape : dimensions of array
        ndarray.size : number of elements in array
    
        Examples
        --------
        >>> a = np.array([[1,2,3],[4,5,6]])
        >>> np.size(a)
        6
        >>> np.size(a,1)
        3
        >>> np.size(a,0)
        2
    
        
    """
def sort(a, axis = -1, kind = None, order = None, endwith = True, fill_value = None):
    """
    
        Return a sorted copy of the masked array.
    
        Equivalent to creating a copy of the array
        and applying the  MaskedArray ``sort()`` method.
    
        Refer to ``MaskedArray.sort`` for the full documentation
    
        See Also
        --------
        MaskedArray.sort : equivalent method
    
        Examples
        --------
        >>> import numpy.ma as ma
        >>> x = [11.2, -3.973, 0.801, -1.41]
        >>> mask = [0, 0, 0, 1]
        >>> masked_x = ma.masked_array(x, mask)
        >>> masked_x
        masked_array(data=[11.2, -3.973, 0.801, --],
                     mask=[False, False, False,  True],
               fill_value=1e+20)
        >>> ma.sort(masked_x)
        masked_array(data=[-3.973, 0.801, 11.2, --],
                     mask=[False, False, False,  True],
               fill_value=1e+20)
        
    """
def take(a, indices, axis = None, out = None, mode = 'raise'):
    """
    
        
    """
def transpose(a, axes = None):
    """
    
        Permute the dimensions of an array.
    
        This function is exactly equivalent to `numpy.transpose`.
    
        See Also
        --------
        numpy.transpose : Equivalent function in top-level NumPy module.
    
        Examples
        --------
        >>> import numpy.ma as ma
        >>> x = ma.arange(4).reshape((2,2))
        >>> x[1, 1] = ma.masked
        >>> x
        masked_array(
          data=[[0, 1],
                [2, --]],
          mask=[[False, False],
                [False,  True]],
          fill_value=999999)
    
        >>> ma.transpose(x)
        masked_array(
          data=[[0, 2],
                [1, --]],
          mask=[[False, False],
                [False,  True]],
          fill_value=999999)
        
    """
def where(condition, x = ..., y = ...):
    """
    
        Return a masked array with elements from `x` or `y`, depending on condition.
    
        .. note::
            When only `condition` is provided, this function is identical to
            `nonzero`. The rest of this documentation covers only the case where
            all three arguments are provided.
    
        Parameters
        ----------
        condition : array_like, bool
            Where True, yield `x`, otherwise yield `y`.
        x, y : array_like, optional
            Values from which to choose. `x`, `y` and `condition` need to be
            broadcastable to some shape.
    
        Returns
        -------
        out : MaskedArray
            An masked array with `masked` elements where the condition is masked,
            elements from `x` where `condition` is True, and elements from `y`
            elsewhere.
    
        See Also
        --------
        numpy.where : Equivalent function in the top-level NumPy module.
        nonzero : The function that is called when x and y are omitted
    
        Examples
        --------
        >>> x = np.ma.array(np.arange(9.).reshape(3, 3), mask=[[0, 1, 0],
        ...                                                    [1, 0, 1],
        ...                                                    [0, 1, 0]])
        >>> x
        masked_array(
          data=[[0.0, --, 2.0],
                [--, 4.0, --],
                [6.0, --, 8.0]],
          mask=[[False,  True, False],
                [ True, False,  True],
                [False,  True, False]],
          fill_value=1e+20)
        >>> np.ma.where(x > 5, x, -3.1416)
        masked_array(
          data=[[-3.1416, --, -3.1416],
                [--, -3.1416, --],
                [6.0, --, 8.0]],
          mask=[[False,  True, False],
                [ True, False,  True],
                [False,  True, False]],
          fill_value=1e+20)
    
        
    """
_NoValue: numpy._globals._NoValueType  # value = <no value>
_legacy_print_templates: dict = {'long_std': 'masked_%(name)s(data =\n %(data)s,\n%(nlen)s        mask =\n %(mask)s,\n%(nlen)s  fill_value = %(fill)s)\n', 'long_flx': 'masked_%(name)s(data =\n %(data)s,\n%(nlen)s        mask =\n %(mask)s,\n%(nlen)s  fill_value = %(fill)s,\n%(nlen)s       dtype = %(dtype)s)\n', 'short_std': 'masked_%(name)s(data = %(data)s,\n%(nlen)s        mask = %(mask)s,\n%(nlen)s  fill_value = %(fill)s)\n', 'short_flx': 'masked_%(name)s(data = %(data)s,\n%(nlen)s        mask = %(mask)s,\n%(nlen)s  fill_value = %(fill)s,\n%(nlen)s       dtype = %(dtype)s)\n'}
abs: _MaskedUnaryOperation  # value = <numpy.ma.core._MaskedUnaryOperation object>
absolute: _MaskedUnaryOperation  # value = <numpy.ma.core._MaskedUnaryOperation object>
add: _MaskedBinaryOperation  # value = <numpy.ma.core._MaskedBinaryOperation object>
all: _frommethod  # value = <numpy.ma.core._frommethod object>
angle: _MaskedUnaryOperation  # value = <numpy.ma.core._MaskedUnaryOperation object>
anom: _frommethod  # value = <numpy.ma.core._frommethod object>
anomalies: _frommethod  # value = <numpy.ma.core._frommethod object>
any: _frommethod  # value = <numpy.ma.core._frommethod object>
arange: _convert2ma  # value = <numpy.ma.core._convert2ma object>
arccos: _MaskedUnaryOperation  # value = <numpy.ma.core._MaskedUnaryOperation object>
arccosh: _MaskedUnaryOperation  # value = <numpy.ma.core._MaskedUnaryOperation object>
arcsin: _MaskedUnaryOperation  # value = <numpy.ma.core._MaskedUnaryOperation object>
arcsinh: _MaskedUnaryOperation  # value = <numpy.ma.core._MaskedUnaryOperation object>
arctan: _MaskedUnaryOperation  # value = <numpy.ma.core._MaskedUnaryOperation object>
arctan2: _MaskedBinaryOperation  # value = <numpy.ma.core._MaskedBinaryOperation object>
arctanh: _MaskedUnaryOperation  # value = <numpy.ma.core._MaskedUnaryOperation object>
argmax: _frommethod  # value = <numpy.ma.core._frommethod object>
argmin: _frommethod  # value = <numpy.ma.core._frommethod object>
around: _MaskedUnaryOperation  # value = <numpy.ma.core._MaskedUnaryOperation object>
bitwise_and: _MaskedBinaryOperation  # value = <numpy.ma.core._MaskedBinaryOperation object>
bitwise_or: _MaskedBinaryOperation  # value = <numpy.ma.core._MaskedBinaryOperation object>
bitwise_xor: _MaskedBinaryOperation  # value = <numpy.ma.core._MaskedBinaryOperation object>
ceil: _MaskedUnaryOperation  # value = <numpy.ma.core._MaskedUnaryOperation object>
clip: _convert2ma  # value = <numpy.ma.core._convert2ma object>
compress: _frommethod  # value = <numpy.ma.core._frommethod object>
conjugate: _MaskedUnaryOperation  # value = <numpy.ma.core._MaskedUnaryOperation object>
copy: _frommethod  # value = <numpy.ma.core._frommethod object>
cos: _MaskedUnaryOperation  # value = <numpy.ma.core._MaskedUnaryOperation object>
cosh: _MaskedUnaryOperation  # value = <numpy.ma.core._MaskedUnaryOperation object>
count: _frommethod  # value = <numpy.ma.core._frommethod object>
cumprod: _frommethod  # value = <numpy.ma.core._frommethod object>
cumsum: _frommethod  # value = <numpy.ma.core._frommethod object>
default_filler: dict  # value = {'b': True, 'c': (1e+20+0j), 'f': 1e+20, 'i': 999999, 'O': '?', 'S': b'N/A', 'u': 999999, 'V': b'???', 'U': 'N/A', 'M8[Y]': numpy.datetime64('NaT'), 'm8[Y]': numpy.timedelta64('NaT','Y'), 'M8[M]': numpy.datetime64('NaT'), 'm8[M]': numpy.timedelta64('NaT','M'), 'M8[W]': numpy.datetime64('NaT'), 'm8[W]': numpy.timedelta64('NaT','W'), 'M8[D]': numpy.datetime64('NaT'), 'm8[D]': numpy.timedelta64('NaT','D'), 'M8[h]': numpy.datetime64('NaT','h'), 'm8[h]': numpy.timedelta64('NaT','h'), 'M8[m]': numpy.datetime64('NaT'), 'm8[m]': numpy.timedelta64('NaT','m'), 'M8[s]': numpy.datetime64('NaT'), 'm8[s]': numpy.timedelta64('NaT','s'), 'M8[ms]': numpy.datetime64('NaT'), 'm8[ms]': numpy.timedelta64('NaT','ms'), 'M8[us]': numpy.datetime64('NaT'), 'm8[us]': numpy.timedelta64('NaT','us'), 'M8[ns]': numpy.datetime64('NaT'), 'm8[ns]': numpy.timedelta64('NaT','ns'), 'M8[ps]': numpy.datetime64('NaT'), 'm8[ps]': numpy.timedelta64('NaT','ps'), 'M8[fs]': numpy.datetime64('NaT'), 'm8[fs]': numpy.timedelta64('NaT','fs'), 'M8[as]': numpy.datetime64('NaT'), 'm8[as]': numpy.timedelta64('NaT','as')}
diagonal: _frommethod  # value = <numpy.ma.core._frommethod object>
divide: _DomainedBinaryOperation  # value = <numpy.ma.core._DomainedBinaryOperation object>
empty: _convert2ma  # value = <numpy.ma.core._convert2ma object>
empty_like: _convert2ma  # value = <numpy.ma.core._convert2ma object>
equal: _MaskedBinaryOperation  # value = <numpy.ma.core._MaskedBinaryOperation object>
exp: _MaskedUnaryOperation  # value = <numpy.ma.core._MaskedUnaryOperation object>
fabs: _MaskedUnaryOperation  # value = <numpy.ma.core._MaskedUnaryOperation object>
floor: _MaskedUnaryOperation  # value = <numpy.ma.core._MaskedUnaryOperation object>
floor_divide: _DomainedBinaryOperation  # value = <numpy.ma.core._DomainedBinaryOperation object>
fmod: _DomainedBinaryOperation  # value = <numpy.ma.core._DomainedBinaryOperation object>
frombuffer: _convert2ma  # value = <numpy.ma.core._convert2ma object>
fromfunction: _convert2ma  # value = <numpy.ma.core._convert2ma object>
greater: _MaskedBinaryOperation  # value = <numpy.ma.core._MaskedBinaryOperation object>
greater_equal: _MaskedBinaryOperation  # value = <numpy.ma.core._MaskedBinaryOperation object>
harden_mask: _frommethod  # value = <numpy.ma.core._frommethod object>
hypot: _MaskedBinaryOperation  # value = <numpy.ma.core._MaskedBinaryOperation object>
identity: _convert2ma  # value = <numpy.ma.core._convert2ma object>
ids: _frommethod  # value = <numpy.ma.core._frommethod object>
indices: _convert2ma  # value = <numpy.ma.core._convert2ma object>
less: _MaskedBinaryOperation  # value = <numpy.ma.core._MaskedBinaryOperation object>
less_equal: _MaskedBinaryOperation  # value = <numpy.ma.core._MaskedBinaryOperation object>
log: _MaskedUnaryOperation  # value = <numpy.ma.core._MaskedUnaryOperation object>
log10: _MaskedUnaryOperation  # value = <numpy.ma.core._MaskedUnaryOperation object>
log2: _MaskedUnaryOperation  # value = <numpy.ma.core._MaskedUnaryOperation object>
logical_and: _MaskedBinaryOperation  # value = <numpy.ma.core._MaskedBinaryOperation object>
logical_not: _MaskedUnaryOperation  # value = <numpy.ma.core._MaskedUnaryOperation object>
logical_or: _MaskedBinaryOperation  # value = <numpy.ma.core._MaskedBinaryOperation object>
logical_xor: _MaskedBinaryOperation  # value = <numpy.ma.core._MaskedBinaryOperation object>
masked: MaskedConstant  # value = masked
masked_print_option: _MaskedPrintOption  # value = --
masked_singleton: MaskedConstant  # value = masked
max_filler: numpy.core.numerictypes._typedict  # value = {<class 'numpy.bool_'>: 0, <class 'numpy.int8'>: -128, <class 'numpy.uint8'>: 0, <class 'numpy.int16'>: -32768, <class 'numpy.uint16'>: 0, <class 'numpy.intc'>: -2147483648, <class 'numpy.uintc'>: 0, <class 'numpy.int64'>: -9223372036854775808, <class 'numpy.uint64'>: 0, <class 'numpy.int32'>: -2147483648, <class 'numpy.uint32'>: 0, <class 'numpy.float16'>: -inf, <class 'numpy.float32'>: -inf, <class 'numpy.float64'>: -inf, <class 'numpy.longdouble'>: -inf, <class 'numpy.complex64'>: (-inf-infj), <class 'numpy.complex128'>: (-inf-infj), <class 'numpy.clongdouble'>: (-inf-infj), <class 'numpy.object_'>: None, <class 'numpy.bytes_'>: None, <class 'numpy.str_'>: None, <class 'numpy.void'>: None, <class 'numpy.datetime64'>: -9223372036854775808, <class 'numpy.timedelta64'>: -9223372036854775808}
maximum: _extrema_operation  # value = <numpy.ma.core._extrema_operation object>
mean: _frommethod  # value = <numpy.ma.core._frommethod object>
min_filler: numpy.core.numerictypes._typedict  # value = {<class 'numpy.bool_'>: 1, <class 'numpy.int8'>: 127, <class 'numpy.uint8'>: 255, <class 'numpy.int16'>: 32767, <class 'numpy.uint16'>: 65535, <class 'numpy.intc'>: 2147483647, <class 'numpy.uintc'>: 4294967295, <class 'numpy.int64'>: 9223372036854775807, <class 'numpy.uint64'>: 18446744073709551615, <class 'numpy.int32'>: 2147483647, <class 'numpy.uint32'>: 4294967295, <class 'numpy.float16'>: inf, <class 'numpy.float32'>: inf, <class 'numpy.float64'>: inf, <class 'numpy.longdouble'>: inf, <class 'numpy.complex64'>: (inf+infj), <class 'numpy.complex128'>: (inf+infj), <class 'numpy.clongdouble'>: (inf+infj), <class 'numpy.object_'>: None, <class 'numpy.bytes_'>: None, <class 'numpy.str_'>: None, <class 'numpy.void'>: None, <class 'numpy.datetime64'>: 9223372036854775807, <class 'numpy.timedelta64'>: 9223372036854775807}
minimum: _extrema_operation  # value = <numpy.ma.core._extrema_operation object>
mod: _DomainedBinaryOperation  # value = <numpy.ma.core._DomainedBinaryOperation object>
multiply: _MaskedBinaryOperation  # value = <numpy.ma.core._MaskedBinaryOperation object>
negative: _MaskedUnaryOperation  # value = <numpy.ma.core._MaskedUnaryOperation object>
nomask: numpy.bool_  # value = False
nonzero: _frommethod  # value = <numpy.ma.core._frommethod object>
not_equal: _MaskedBinaryOperation  # value = <numpy.ma.core._MaskedBinaryOperation object>
ones: _convert2ma  # value = <numpy.ma.core._convert2ma object>
ones_like: _convert2ma  # value = <numpy.ma.core._convert2ma object>
prod: _frommethod  # value = <numpy.ma.core._frommethod object>
product: _frommethod  # value = <numpy.ma.core._frommethod object>
ravel: _frommethod  # value = <numpy.ma.core._frommethod object>
remainder: _DomainedBinaryOperation  # value = <numpy.ma.core._DomainedBinaryOperation object>
repeat: _frommethod  # value = <numpy.ma.core._frommethod object>
shrink_mask: _frommethod  # value = <numpy.ma.core._frommethod object>
sin: _MaskedUnaryOperation  # value = <numpy.ma.core._MaskedUnaryOperation object>
sinh: _MaskedUnaryOperation  # value = <numpy.ma.core._MaskedUnaryOperation object>
soften_mask: _frommethod  # value = <numpy.ma.core._frommethod object>
sqrt: _MaskedUnaryOperation  # value = <numpy.ma.core._MaskedUnaryOperation object>
squeeze: _convert2ma  # value = <numpy.ma.core._convert2ma object>
std: _frommethod  # value = <numpy.ma.core._frommethod object>
subtract: _MaskedBinaryOperation  # value = <numpy.ma.core._MaskedBinaryOperation object>
sum: _frommethod  # value = <numpy.ma.core._frommethod object>
swapaxes: _frommethod  # value = <numpy.ma.core._frommethod object>
tan: _MaskedUnaryOperation  # value = <numpy.ma.core._MaskedUnaryOperation object>
tanh: _MaskedUnaryOperation  # value = <numpy.ma.core._MaskedUnaryOperation object>
trace: _frommethod  # value = <numpy.ma.core._frommethod object>
true_divide: _DomainedBinaryOperation  # value = <numpy.ma.core._DomainedBinaryOperation object>
ufunc_domain: dict  # value = {<ufunc 'exp'>: None, <ufunc 'conjugate'>: None, <ufunc 'sin'>: None, <ufunc 'cos'>: None, <ufunc 'arctan'>: None, <ufunc 'arcsinh'>: None, <ufunc 'sinh'>: None, <ufunc 'cosh'>: None, <ufunc 'tanh'>: None, <ufunc 'absolute'>: None, numpy.angle: None, <ufunc 'fabs'>: None, <ufunc 'negative'>: None, <ufunc 'floor'>: None, <ufunc 'ceil'>: None, numpy.round_: None, <ufunc 'logical_not'>: None, <ufunc 'sqrt'>: <numpy.ma.core._DomainGreaterEqual object>, <ufunc 'log'>: <numpy.ma.core._DomainGreater object>, <ufunc 'log2'>: <numpy.ma.core._DomainGreater object>, <ufunc 'log10'>: <numpy.ma.core._DomainGreater object>, <ufunc 'tan'>: <numpy.ma.core._DomainTan object>, <ufunc 'arcsin'>: <numpy.ma.core._DomainCheckInterval object>, <ufunc 'arccos'>: <numpy.ma.core._DomainCheckInterval object>, <ufunc 'arccosh'>: <numpy.ma.core._DomainGreaterEqual object>, <ufunc 'arctanh'>: <numpy.ma.core._DomainCheckInterval object>, <ufunc 'add'>: None, <ufunc 'subtract'>: None, <ufunc 'multiply'>: None, <ufunc 'arctan2'>: None, <ufunc 'equal'>: None, <ufunc 'not_equal'>: None, <ufunc 'less_equal'>: None, <ufunc 'greater_equal'>: None, <ufunc 'less'>: None, <ufunc 'greater'>: None, <ufunc 'logical_and'>: None, <ufunc 'logical_or'>: None, <ufunc 'logical_xor'>: None, <ufunc 'bitwise_and'>: None, <ufunc 'bitwise_or'>: None, <ufunc 'bitwise_xor'>: None, <ufunc 'hypot'>: None, <ufunc 'divide'>: <numpy.ma.core._DomainSafeDivide object>, <ufunc 'floor_divide'>: <numpy.ma.core._DomainSafeDivide object>, <ufunc 'remainder'>: <numpy.ma.core._DomainSafeDivide object>, <ufunc 'fmod'>: <numpy.ma.core._DomainSafeDivide object>}
ufunc_fills: dict  # value = {<ufunc 'exp'>: 0, <ufunc 'conjugate'>: 0, <ufunc 'sin'>: 0, <ufunc 'cos'>: 0, <ufunc 'arctan'>: 0, <ufunc 'arcsinh'>: 0, <ufunc 'sinh'>: 0, <ufunc 'cosh'>: 0, <ufunc 'tanh'>: 0, <ufunc 'absolute'>: 0, numpy.angle: 0, <ufunc 'fabs'>: 0, <ufunc 'negative'>: 0, <ufunc 'floor'>: 0, <ufunc 'ceil'>: 0, numpy.round_: 0, <ufunc 'logical_not'>: 0, <ufunc 'sqrt'>: 0.0, <ufunc 'log'>: 1.0, <ufunc 'log2'>: 1.0, <ufunc 'log10'>: 1.0, <ufunc 'tan'>: 0.0, <ufunc 'arcsin'>: 0.0, <ufunc 'arccos'>: 0.0, <ufunc 'arccosh'>: 1.0, <ufunc 'arctanh'>: 0.0, <ufunc 'add'>: (0, 0), <ufunc 'subtract'>: (0, 0), <ufunc 'multiply'>: (1, 1), <ufunc 'arctan2'>: (0.0, 1.0), <ufunc 'equal'>: (0, 0), <ufunc 'not_equal'>: (0, 0), <ufunc 'less_equal'>: (0, 0), <ufunc 'greater_equal'>: (0, 0), <ufunc 'less'>: (0, 0), <ufunc 'greater'>: (0, 0), <ufunc 'logical_and'>: (1, 1), <ufunc 'logical_or'>: (0, 0), <ufunc 'logical_xor'>: (0, 0), <ufunc 'bitwise_and'>: (0, 0), <ufunc 'bitwise_or'>: (0, 0), <ufunc 'bitwise_xor'>: (0, 0), <ufunc 'hypot'>: (0, 0), <ufunc 'divide'>: (0, 1), <ufunc 'floor_divide'>: (0, 1), <ufunc 'remainder'>: (0, 1), <ufunc 'fmod'>: (0, 1)}
v: str = 'as'
var: _frommethod  # value = <numpy.ma.core._frommethod object>
zeros: _convert2ma  # value = <numpy.ma.core._convert2ma object>
zeros_like: _convert2ma  # value = <numpy.ma.core._convert2ma object>
alltrue = _MaskedBinaryOperation.reduce
get_data = getdata
get_mask = getmask
innerproduct = inner
isMA = isMaskedArray
isarray = isMaskedArray
masked_array = MaskedArray
outerproduct = outer
round = round_
sometrue = _MaskedBinaryOperation.reduce
