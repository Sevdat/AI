"""

Masked arrays add-ons.

A collection of utilities for `numpy.ma`.

:author: Pierre Gerard-Marchant
:contact: pierregm_at_uga_dot_edu
:version: $Id: extras.py 3473 2007-10-29 15:18:13Z jarrod.millman $

"""
from __future__ import annotations
import itertools as itertools
import numpy
import numpy as np
from numpy import array as nxarray
from numpy.core._multiarray_umath import normalize_axis_index
from numpy.core.numeric import normalize_axis_tuple
from numpy.lib.function_base import _ureduce
import numpy.lib.index_tricks
from numpy.lib.index_tricks import AxisConcatenator
import numpy.ma.core
from numpy.ma import core as ma
from numpy.ma.core import MAError
from numpy.ma.core import MaskedArray as masked_array
from numpy.ma.core import MaskedArray
from numpy.ma.core import array
from numpy.ma.core import asarray
from numpy.ma.core import concatenate
from numpy.ma.core import dot
from numpy.ma.core import filled
from numpy.ma.core import get_masked_subclass
from numpy.ma.core import getdata
from numpy.ma.core import getmask
from numpy.ma.core import getmaskarray
from numpy.ma.core import make_mask_descr
from numpy.ma.core import mask_or
from numpy.ma.core import sort
from numpy import ndarray
import warnings as warnings
__all__: list = ['apply_along_axis', 'apply_over_axes', 'atleast_1d', 'atleast_2d', 'atleast_3d', 'average', 'clump_masked', 'clump_unmasked', 'column_stack', 'compress_cols', 'compress_nd', 'compress_rowcols', 'compress_rows', 'count_masked', 'corrcoef', 'cov', 'diagflat', 'dot', 'dstack', 'ediff1d', 'flatnotmasked_contiguous', 'flatnotmasked_edges', 'hsplit', 'hstack', 'isin', 'in1d', 'intersect1d', 'mask_cols', 'mask_rowcols', 'mask_rows', 'masked_all', 'masked_all_like', 'median', 'mr_', 'ndenumerate', 'notmasked_contiguous', 'notmasked_edges', 'polyfit', 'row_stack', 'setdiff1d', 'setxor1d', 'stack', 'unique', 'union1d', 'vander', 'vstack']
class MAxisConcatenator(numpy.lib.index_tricks.AxisConcatenator):
    """
    
        Translate slice objects to concatenation along an axis.
    
        For documentation on usage, see `mr_class`.
    
        See Also
        --------
        mr_class
    
        
    """
    @staticmethod
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
    @classmethod
    def makemat(cls, arr):
        ...
    def __getitem__(self, key):
        ...
class _fromnxfunction:
    """
    
        Defines a wrapper to adapt NumPy functions to masked arrays.
    
    
        An instance of `_fromnxfunction` can be called with the same parameters
        as the wrapped NumPy function. The docstring of `newfunc` is adapted from
        the wrapped function as well, see `getdoc`.
    
        This class should not be used directly. Instead, one of its extensions that
        provides support for a specific type of input should be used.
    
        Parameters
        ----------
        funcname : str
            The name of the function to be adapted. The function should be
            in the NumPy namespace (i.e. ``np.funcname``).
    
        
    """
    def __call__(self, *args, **params):
        ...
    def __init__(self, funcname):
        ...
    def getdoc(self):
        """
        
                Retrieve the docstring and signature from the function.
        
                The ``__doc__`` attribute of the function is used as the docstring for
                the new masked array version of the function. A note on application
                of the function to the mask is appended.
        
                Parameters
                ----------
                None
        
                
        """
class _fromnxfunction_allargs(_fromnxfunction):
    """
    
        A version of `_fromnxfunction` that is called with multiple array
        arguments. Similar to `_fromnxfunction_args` except that all args
        are converted to arrays even if they are not so already. This makes
        it possible to process scalars as 1-D arrays. Only keyword arguments
        are passed through verbatim for the data and mask calls. Arrays
        arguments are processed independently and the results are returned
        in a list. If only one arg is present, the return value is just the
        processed array instead of a list.
        
    """
    def __call__(self, *args, **params):
        ...
class _fromnxfunction_args(_fromnxfunction):
    """
    
        A version of `_fromnxfunction` that is called with multiple array
        arguments. The first non-array-like input marks the beginning of the
        arguments that are passed verbatim for both the data and mask calls.
        Array arguments are processed independently and the results are
        returned in a list. If only one array is found, the return value is
        just the processed array instead of a list.
        
    """
    def __call__(self, *args, **params):
        ...
class _fromnxfunction_seq(_fromnxfunction):
    """
    
        A version of `_fromnxfunction` that is called with a single sequence
        of arrays followed by auxiliary args that are passed verbatim for
        both the data and mask calls.
        
    """
    def __call__(self, x, *args, **params):
        ...
class _fromnxfunction_single(_fromnxfunction):
    """
    
        A version of `_fromnxfunction` that is called with a single array
        argument followed by auxiliary args that are passed verbatim for
        both the data and mask calls.
        
    """
    def __call__(self, x, *args, **params):
        ...
class mr_class(MAxisConcatenator):
    """
    
        Translate slice objects to concatenation along the first axis.
    
        This is the masked array version of `lib.index_tricks.RClass`.
    
        See Also
        --------
        lib.index_tricks.RClass
    
        Examples
        --------
        >>> np.ma.mr_[np.ma.array([1,2,3]), 0, 0, np.ma.array([4,5,6])]
        masked_array(data=[1, 2, 3, ..., 4, 5, 6],
                     mask=False,
               fill_value=999999)
    
        
    """
    def __init__(self):
        ...
def _covhelper(x, y = None, rowvar = True, allow_masked = True):
    """
    
        Private function for the computation of covariance and correlation
        coefficients.
    
        
    """
def _ezclump(mask):
    """
    
        Finds the clumps (groups of data with the same values) for a 1D bool array.
    
        Returns a series of slices.
        
    """
def _median(a, axis = None, out = None, overwrite_input = False):
    ...
def apply_along_axis(func1d, axis, arr, *args, **kwargs):
    """
    
        Apply a function to 1-D slices along the given axis.
    
        Execute `func1d(a, *args, **kwargs)` where `func1d` operates on 1-D arrays
        and `a` is a 1-D slice of `arr` along `axis`.
    
        This is equivalent to (but faster than) the following use of `ndindex` and
        `s_`, which sets each of ``ii``, ``jj``, and ``kk`` to a tuple of indices::
    
            Ni, Nk = a.shape[:axis], a.shape[axis+1:]
            for ii in ndindex(Ni):
                for kk in ndindex(Nk):
                    f = func1d(arr[ii + s_[:,] + kk])
                    Nj = f.shape
                    for jj in ndindex(Nj):
                        out[ii + jj + kk] = f[jj]
    
        Equivalently, eliminating the inner loop, this can be expressed as::
    
            Ni, Nk = a.shape[:axis], a.shape[axis+1:]
            for ii in ndindex(Ni):
                for kk in ndindex(Nk):
                    out[ii + s_[...,] + kk] = func1d(arr[ii + s_[:,] + kk])
    
        Parameters
        ----------
        func1d : function (M,) -> (Nj...)
            This function should accept 1-D arrays. It is applied to 1-D
            slices of `arr` along the specified axis.
        axis : integer
            Axis along which `arr` is sliced.
        arr : ndarray (Ni..., M, Nk...)
            Input array.
        args : any
            Additional arguments to `func1d`.
        kwargs : any
            Additional named arguments to `func1d`.
    
            .. versionadded:: 1.9.0
    
    
        Returns
        -------
        out : ndarray  (Ni..., Nj..., Nk...)
            The output array. The shape of `out` is identical to the shape of
            `arr`, except along the `axis` dimension. This axis is removed, and
            replaced with new dimensions equal to the shape of the return value
            of `func1d`. So if `func1d` returns a scalar `out` will have one
            fewer dimensions than `arr`.
    
        See Also
        --------
        apply_over_axes : Apply a function repeatedly over multiple axes.
    
        Examples
        --------
        >>> def my_func(a):
        ...     \"\"\"Average first and last element of a 1-D array\"\"\"
        ...     return (a[0] + a[-1]) * 0.5
        >>> b = np.array([[1,2,3], [4,5,6], [7,8,9]])
        >>> np.apply_along_axis(my_func, 0, b)
        array([4., 5., 6.])
        >>> np.apply_along_axis(my_func, 1, b)
        array([2.,  5.,  8.])
    
        For a function that returns a 1D array, the number of dimensions in
        `outarr` is the same as `arr`.
    
        >>> b = np.array([[8,1,7], [4,3,9], [5,2,6]])
        >>> np.apply_along_axis(sorted, 1, b)
        array([[1, 7, 8],
               [3, 4, 9],
               [2, 5, 6]])
    
        For a function that returns a higher dimensional array, those dimensions
        are inserted in place of the `axis` dimension.
    
        >>> b = np.array([[1,2,3], [4,5,6], [7,8,9]])
        >>> np.apply_along_axis(np.diag, -1, b)
        array([[[1, 0, 0],
                [0, 2, 0],
                [0, 0, 3]],
               [[4, 0, 0],
                [0, 5, 0],
                [0, 0, 6]],
               [[7, 0, 0],
                [0, 8, 0],
                [0, 0, 9]]])
        
    """
def apply_over_axes(func, a, axes):
    """
    
        Apply a function repeatedly over multiple axes.
    
        `func` is called as `res = func(a, axis)`, where `axis` is the first
        element of `axes`.  The result `res` of the function call must have
        either the same dimensions as `a` or one less dimension.  If `res`
        has one less dimension than `a`, a dimension is inserted before
        `axis`.  The call to `func` is then repeated for each axis in `axes`,
        with `res` as the first argument.
    
        Parameters
        ----------
        func : function
            This function must take two arguments, `func(a, axis)`.
        a : array_like
            Input array.
        axes : array_like
            Axes over which `func` is applied; the elements must be integers.
    
        Returns
        -------
        apply_over_axis : ndarray
            The output array.  The number of dimensions is the same as `a`,
            but the shape can be different.  This depends on whether `func`
            changes the shape of its output with respect to its input.
    
        See Also
        --------
        apply_along_axis :
            Apply a function to 1-D slices of an array along the given axis.
    
        Examples
        --------
        >>> a = np.ma.arange(24).reshape(2,3,4)
        >>> a[:,0,1] = np.ma.masked
        >>> a[:,1,:] = np.ma.masked
        >>> a
        masked_array(
          data=[[[0, --, 2, 3],
                 [--, --, --, --],
                 [8, 9, 10, 11]],
                [[12, --, 14, 15],
                 [--, --, --, --],
                 [20, 21, 22, 23]]],
          mask=[[[False,  True, False, False],
                 [ True,  True,  True,  True],
                 [False, False, False, False]],
                [[False,  True, False, False],
                 [ True,  True,  True,  True],
                 [False, False, False, False]]],
          fill_value=999999)
        >>> np.ma.apply_over_axes(np.ma.sum, a, [0,2])
        masked_array(
          data=[[[46],
                 [--],
                 [124]]],
          mask=[[[False],
                 [ True],
                 [False]]],
          fill_value=999999)
    
        Tuple axis arguments to ufuncs are equivalent:
    
        >>> np.ma.sum(a, axis=(0,2)).reshape((1,-1,1))
        masked_array(
          data=[[[46],
                 [--],
                 [124]]],
          mask=[[[False],
                 [ True],
                 [False]]],
          fill_value=999999)
        
    """
def average(a, axis = None, weights = None, returned = False, *, keepdims = ...):
    """
    
        Return the weighted average of array over the given axis.
    
        Parameters
        ----------
        a : array_like
            Data to be averaged.
            Masked entries are not taken into account in the computation.
        axis : int, optional
            Axis along which to average `a`. If None, averaging is done over
            the flattened array.
        weights : array_like, optional
            The importance that each element has in the computation of the average.
            The weights array can either be 1-D (in which case its length must be
            the size of `a` along the given axis) or of the same shape as `a`.
            If ``weights=None``, then all data in `a` are assumed to have a
            weight equal to one.  The 1-D calculation is::
    
                avg = sum(a * weights) / sum(weights)
    
            The only constraint on `weights` is that `sum(weights)` must not be 0.
        returned : bool, optional
            Flag indicating whether a tuple ``(result, sum of weights)``
            should be returned as output (True), or just the result (False).
            Default is False.
        keepdims : bool, optional
            If this is set to True, the axes which are reduced are left
            in the result as dimensions with size one. With this option,
            the result will broadcast correctly against the original `a`.
            *Note:* `keepdims` will not work with instances of `numpy.matrix`
            or other classes whose methods do not support `keepdims`.
    
            .. versionadded:: 1.23.0
    
        Returns
        -------
        average, [sum_of_weights] : (tuple of) scalar or MaskedArray
            The average along the specified axis. When returned is `True`,
            return a tuple with the average as the first element and the sum
            of the weights as the second element. The return type is `np.float64`
            if `a` is of integer type and floats smaller than `float64`, or the
            input data-type, otherwise. If returned, `sum_of_weights` is always
            `float64`.
    
        Examples
        --------
        >>> a = np.ma.array([1., 2., 3., 4.], mask=[False, False, True, True])
        >>> np.ma.average(a, weights=[3, 1, 0, 0])
        1.25
    
        >>> x = np.ma.arange(6.).reshape(3, 2)
        >>> x
        masked_array(
          data=[[0., 1.],
                [2., 3.],
                [4., 5.]],
          mask=False,
          fill_value=1e+20)
        >>> avg, sumweights = np.ma.average(x, axis=0, weights=[1, 2, 3],
        ...                                 returned=True)
        >>> avg
        masked_array(data=[2.6666666666666665, 3.6666666666666665],
                     mask=[False, False],
               fill_value=1e+20)
    
        With ``keepdims=True``, the following result has shape (3, 1).
    
        >>> np.ma.average(x, axis=1, keepdims=True)
        masked_array(
          data=[[0.5],
                [2.5],
                [4.5]],
          mask=False,
          fill_value=1e+20)
        
    """
def clump_masked(a):
    """
    
        Returns a list of slices corresponding to the masked clumps of a 1-D array.
        (A "clump" is defined as a contiguous region of the array).
    
        Parameters
        ----------
        a : ndarray
            A one-dimensional masked array.
    
        Returns
        -------
        slices : list of slice
            The list of slices, one for each continuous region of masked elements
            in `a`.
    
        Notes
        -----
        .. versionadded:: 1.4.0
    
        See Also
        --------
        flatnotmasked_edges, flatnotmasked_contiguous, notmasked_edges
        notmasked_contiguous, clump_unmasked
    
        Examples
        --------
        >>> a = np.ma.masked_array(np.arange(10))
        >>> a[[0, 1, 2, 6, 8, 9]] = np.ma.masked
        >>> np.ma.clump_masked(a)
        [slice(0, 3, None), slice(6, 7, None), slice(8, 10, None)]
    
        
    """
def clump_unmasked(a):
    """
    
        Return list of slices corresponding to the unmasked clumps of a 1-D array.
        (A "clump" is defined as a contiguous region of the array).
    
        Parameters
        ----------
        a : ndarray
            A one-dimensional masked array.
    
        Returns
        -------
        slices : list of slice
            The list of slices, one for each continuous region of unmasked
            elements in `a`.
    
        Notes
        -----
        .. versionadded:: 1.4.0
    
        See Also
        --------
        flatnotmasked_edges, flatnotmasked_contiguous, notmasked_edges
        notmasked_contiguous, clump_masked
    
        Examples
        --------
        >>> a = np.ma.masked_array(np.arange(10))
        >>> a[[0, 1, 2, 6, 8, 9]] = np.ma.masked
        >>> np.ma.clump_unmasked(a)
        [slice(3, 6, None), slice(7, 8, None)]
    
        
    """
def compress_cols(a):
    """
    
        Suppress whole columns of a 2-D array that contain masked values.
    
        This is equivalent to ``np.ma.compress_rowcols(a, 1)``, see
        `compress_rowcols` for details.
    
        See Also
        --------
        compress_rowcols
    
        
    """
def compress_nd(x, axis = None):
    """
    Suppress slices from multiple dimensions which contain masked values.
    
        Parameters
        ----------
        x : array_like, MaskedArray
            The array to operate on. If not a MaskedArray instance (or if no array
            elements are masked), `x` is interpreted as a MaskedArray with `mask`
            set to `nomask`.
        axis : tuple of ints or int, optional
            Which dimensions to suppress slices from can be configured with this
            parameter.
            - If axis is a tuple of ints, those are the axes to suppress slices from.
            - If axis is an int, then that is the only axis to suppress slices from.
            - If axis is None, all axis are selected.
    
        Returns
        -------
        compress_array : ndarray
            The compressed array.
        
    """
def compress_rowcols(x, axis = None):
    """
    
        Suppress the rows and/or columns of a 2-D array that contain
        masked values.
    
        The suppression behavior is selected with the `axis` parameter.
    
        - If axis is None, both rows and columns are suppressed.
        - If axis is 0, only rows are suppressed.
        - If axis is 1 or -1, only columns are suppressed.
    
        Parameters
        ----------
        x : array_like, MaskedArray
            The array to operate on.  If not a MaskedArray instance (or if no array
            elements are masked), `x` is interpreted as a MaskedArray with
            `mask` set to `nomask`. Must be a 2D array.
        axis : int, optional
            Axis along which to perform the operation. Default is None.
    
        Returns
        -------
        compressed_array : ndarray
            The compressed array.
    
        Examples
        --------
        >>> x = np.ma.array(np.arange(9).reshape(3, 3), mask=[[1, 0, 0],
        ...                                                   [1, 0, 0],
        ...                                                   [0, 0, 0]])
        >>> x
        masked_array(
          data=[[--, 1, 2],
                [--, 4, 5],
                [6, 7, 8]],
          mask=[[ True, False, False],
                [ True, False, False],
                [False, False, False]],
          fill_value=999999)
    
        >>> np.ma.compress_rowcols(x)
        array([[7, 8]])
        >>> np.ma.compress_rowcols(x, 0)
        array([[6, 7, 8]])
        >>> np.ma.compress_rowcols(x, 1)
        array([[1, 2],
               [4, 5],
               [7, 8]])
    
        
    """
def compress_rows(a):
    """
    
        Suppress whole rows of a 2-D array that contain masked values.
    
        This is equivalent to ``np.ma.compress_rowcols(a, 0)``, see
        `compress_rowcols` for details.
    
        See Also
        --------
        compress_rowcols
    
        
    """
def corrcoef(x, y = None, rowvar = True, bias = ..., allow_masked = True, ddof = ...):
    """
    
        Return Pearson product-moment correlation coefficients.
    
        Except for the handling of missing data this function does the same as
        `numpy.corrcoef`. For more details and examples, see `numpy.corrcoef`.
    
        Parameters
        ----------
        x : array_like
            A 1-D or 2-D array containing multiple variables and observations.
            Each row of `x` represents a variable, and each column a single
            observation of all those variables. Also see `rowvar` below.
        y : array_like, optional
            An additional set of variables and observations. `y` has the same
            shape as `x`.
        rowvar : bool, optional
            If `rowvar` is True (default), then each row represents a
            variable, with observations in the columns. Otherwise, the relationship
            is transposed: each column represents a variable, while the rows
            contain observations.
        bias : _NoValue, optional
            Has no effect, do not use.
    
            .. deprecated:: 1.10.0
        allow_masked : bool, optional
            If True, masked values are propagated pair-wise: if a value is masked
            in `x`, the corresponding value is masked in `y`.
            If False, raises an exception.  Because `bias` is deprecated, this
            argument needs to be treated as keyword only to avoid a warning.
        ddof : _NoValue, optional
            Has no effect, do not use.
    
            .. deprecated:: 1.10.0
    
        See Also
        --------
        numpy.corrcoef : Equivalent function in top-level NumPy module.
        cov : Estimate the covariance matrix.
    
        Notes
        -----
        This function accepts but discards arguments `bias` and `ddof`.  This is
        for backwards compatibility with previous versions of this function.  These
        arguments had no effect on the return values of the function and can be
        safely ignored in this and previous versions of numpy.
        
    """
def count_masked(arr, axis = None):
    """
    
        Count the number of masked elements along the given axis.
    
        Parameters
        ----------
        arr : array_like
            An array with (possibly) masked elements.
        axis : int, optional
            Axis along which to count. If None (default), a flattened
            version of the array is used.
    
        Returns
        -------
        count : int, ndarray
            The total number of masked elements (axis=None) or the number
            of masked elements along each slice of the given axis.
    
        See Also
        --------
        MaskedArray.count : Count non-masked elements.
    
        Examples
        --------
        >>> import numpy.ma as ma
        >>> a = np.arange(9).reshape((3,3))
        >>> a = ma.array(a)
        >>> a[1, 0] = ma.masked
        >>> a[1, 2] = ma.masked
        >>> a[2, 1] = ma.masked
        >>> a
        masked_array(
          data=[[0, 1, 2],
                [--, 4, --],
                [6, --, 8]],
          mask=[[False, False, False],
                [ True, False,  True],
                [False,  True, False]],
          fill_value=999999)
        >>> ma.count_masked(a)
        3
    
        When the `axis` keyword is used an array is returned.
    
        >>> ma.count_masked(a, axis=0)
        array([1, 1, 1])
        >>> ma.count_masked(a, axis=1)
        array([0, 2, 1])
    
        
    """
def cov(x, y = None, rowvar = True, bias = False, allow_masked = True, ddof = None):
    """
    
        Estimate the covariance matrix.
    
        Except for the handling of missing data this function does the same as
        `numpy.cov`. For more details and examples, see `numpy.cov`.
    
        By default, masked values are recognized as such. If `x` and `y` have the
        same shape, a common mask is allocated: if ``x[i,j]`` is masked, then
        ``y[i,j]`` will also be masked.
        Setting `allow_masked` to False will raise an exception if values are
        missing in either of the input arrays.
    
        Parameters
        ----------
        x : array_like
            A 1-D or 2-D array containing multiple variables and observations.
            Each row of `x` represents a variable, and each column a single
            observation of all those variables. Also see `rowvar` below.
        y : array_like, optional
            An additional set of variables and observations. `y` has the same
            shape as `x`.
        rowvar : bool, optional
            If `rowvar` is True (default), then each row represents a
            variable, with observations in the columns. Otherwise, the relationship
            is transposed: each column represents a variable, while the rows
            contain observations.
        bias : bool, optional
            Default normalization (False) is by ``(N-1)``, where ``N`` is the
            number of observations given (unbiased estimate). If `bias` is True,
            then normalization is by ``N``. This keyword can be overridden by
            the keyword ``ddof`` in numpy versions >= 1.5.
        allow_masked : bool, optional
            If True, masked values are propagated pair-wise: if a value is masked
            in `x`, the corresponding value is masked in `y`.
            If False, raises a `ValueError` exception when some values are missing.
        ddof : {None, int}, optional
            If not ``None`` normalization is by ``(N - ddof)``, where ``N`` is
            the number of observations; this overrides the value implied by
            ``bias``. The default value is ``None``.
    
            .. versionadded:: 1.5
    
        Raises
        ------
        ValueError
            Raised if some values are missing and `allow_masked` is False.
    
        See Also
        --------
        numpy.cov
    
        
    """
def ediff1d(arr, to_end = None, to_begin = None):
    """
    
        Compute the differences between consecutive elements of an array.
    
        This function is the equivalent of `numpy.ediff1d` that takes masked
        values into account, see `numpy.ediff1d` for details.
    
        See Also
        --------
        numpy.ediff1d : Equivalent function for ndarrays.
    
        
    """
def flatnotmasked_contiguous(a):
    """
    
        Find contiguous unmasked data in a masked array.
    
        Parameters
        ----------
        a : array_like
            The input array.
    
        Returns
        -------
        slice_list : list
            A sorted sequence of `slice` objects (start index, end index).
    
            .. versionchanged:: 1.15.0
                Now returns an empty list instead of None for a fully masked array
    
        See Also
        --------
        flatnotmasked_edges, notmasked_contiguous, notmasked_edges
        clump_masked, clump_unmasked
    
        Notes
        -----
        Only accepts 2-D arrays at most.
    
        Examples
        --------
        >>> a = np.ma.arange(10)
        >>> np.ma.flatnotmasked_contiguous(a)
        [slice(0, 10, None)]
    
        >>> mask = (a < 3) | (a > 8) | (a == 5)
        >>> a[mask] = np.ma.masked
        >>> np.array(a[~a.mask])
        array([3, 4, 6, 7, 8])
    
        >>> np.ma.flatnotmasked_contiguous(a)
        [slice(3, 5, None), slice(6, 9, None)]
        >>> a[:] = np.ma.masked
        >>> np.ma.flatnotmasked_contiguous(a)
        []
    
        
    """
def flatnotmasked_edges(a):
    """
    
        Find the indices of the first and last unmasked values.
    
        Expects a 1-D `MaskedArray`, returns None if all values are masked.
    
        Parameters
        ----------
        a : array_like
            Input 1-D `MaskedArray`
    
        Returns
        -------
        edges : ndarray or None
            The indices of first and last non-masked value in the array.
            Returns None if all values are masked.
    
        See Also
        --------
        flatnotmasked_contiguous, notmasked_contiguous, notmasked_edges
        clump_masked, clump_unmasked
    
        Notes
        -----
        Only accepts 1-D arrays.
    
        Examples
        --------
        >>> a = np.ma.arange(10)
        >>> np.ma.flatnotmasked_edges(a)
        array([0, 9])
    
        >>> mask = (a < 3) | (a > 8) | (a == 5)
        >>> a[mask] = np.ma.masked
        >>> np.array(a[~a.mask])
        array([3, 4, 6, 7, 8])
    
        >>> np.ma.flatnotmasked_edges(a)
        array([3, 8])
    
        >>> a[:] = np.ma.masked
        >>> print(np.ma.flatnotmasked_edges(a))
        None
    
        
    """
def flatten_inplace(seq):
    """
    Flatten a sequence in place.
    """
def in1d(ar1, ar2, assume_unique = False, invert = False):
    """
    
        Test whether each element of an array is also present in a second
        array.
    
        The output is always a masked array. See `numpy.in1d` for more details.
    
        We recommend using :func:`isin` instead of `in1d` for new code.
    
        See Also
        --------
        isin       : Version of this function that preserves the shape of ar1.
        numpy.in1d : Equivalent function for ndarrays.
    
        Notes
        -----
        .. versionadded:: 1.4.0
    
        
    """
def intersect1d(ar1, ar2, assume_unique = False):
    """
    
        Returns the unique elements common to both arrays.
    
        Masked values are considered equal one to the other.
        The output is always a masked array.
    
        See `numpy.intersect1d` for more details.
    
        See Also
        --------
        numpy.intersect1d : Equivalent function for ndarrays.
    
        Examples
        --------
        >>> x = np.ma.array([1, 3, 3, 3], mask=[0, 0, 0, 1])
        >>> y = np.ma.array([3, 1, 1, 1], mask=[0, 0, 0, 1])
        >>> np.ma.intersect1d(x, y)
        masked_array(data=[1, 3, --],
                     mask=[False, False,  True],
               fill_value=999999)
    
        
    """
def isin(element, test_elements, assume_unique = False, invert = False):
    """
    
        Calculates `element in test_elements`, broadcasting over
        `element` only.
    
        The output is always a masked array of the same shape as `element`.
        See `numpy.isin` for more details.
    
        See Also
        --------
        in1d       : Flattened version of this function.
        numpy.isin : Equivalent function for ndarrays.
    
        Notes
        -----
        .. versionadded:: 1.13.0
    
        
    """
def issequence(seq):
    """
    
        Is seq a sequence (ndarray, list or tuple)?
    
        
    """
def mask_cols(a, axis = ...):
    """
    
        Mask columns of a 2D array that contain masked values.
    
        This function is a shortcut to ``mask_rowcols`` with `axis` equal to 1.
    
        See Also
        --------
        mask_rowcols : Mask rows and/or columns of a 2D array.
        masked_where : Mask where a condition is met.
    
        Examples
        --------
        >>> import numpy.ma as ma
        >>> a = np.zeros((3, 3), dtype=int)
        >>> a[1, 1] = 1
        >>> a
        array([[0, 0, 0],
               [0, 1, 0],
               [0, 0, 0]])
        >>> a = ma.masked_equal(a, 1)
        >>> a
        masked_array(
          data=[[0, 0, 0],
                [0, --, 0],
                [0, 0, 0]],
          mask=[[False, False, False],
                [False,  True, False],
                [False, False, False]],
          fill_value=1)
        >>> ma.mask_cols(a)
        masked_array(
          data=[[0, --, 0],
                [0, --, 0],
                [0, --, 0]],
          mask=[[False,  True, False],
                [False,  True, False],
                [False,  True, False]],
          fill_value=1)
    
        
    """
def mask_rowcols(a, axis = None):
    """
    
        Mask rows and/or columns of a 2D array that contain masked values.
    
        Mask whole rows and/or columns of a 2D array that contain
        masked values.  The masking behavior is selected using the
        `axis` parameter.
    
          - If `axis` is None, rows *and* columns are masked.
          - If `axis` is 0, only rows are masked.
          - If `axis` is 1 or -1, only columns are masked.
    
        Parameters
        ----------
        a : array_like, MaskedArray
            The array to mask.  If not a MaskedArray instance (or if no array
            elements are masked), the result is a MaskedArray with `mask` set
            to `nomask` (False). Must be a 2D array.
        axis : int, optional
            Axis along which to perform the operation. If None, applies to a
            flattened version of the array.
    
        Returns
        -------
        a : MaskedArray
            A modified version of the input array, masked depending on the value
            of the `axis` parameter.
    
        Raises
        ------
        NotImplementedError
            If input array `a` is not 2D.
    
        See Also
        --------
        mask_rows : Mask rows of a 2D array that contain masked values.
        mask_cols : Mask cols of a 2D array that contain masked values.
        masked_where : Mask where a condition is met.
    
        Notes
        -----
        The input array's mask is modified by this function.
    
        Examples
        --------
        >>> import numpy.ma as ma
        >>> a = np.zeros((3, 3), dtype=int)
        >>> a[1, 1] = 1
        >>> a
        array([[0, 0, 0],
               [0, 1, 0],
               [0, 0, 0]])
        >>> a = ma.masked_equal(a, 1)
        >>> a
        masked_array(
          data=[[0, 0, 0],
                [0, --, 0],
                [0, 0, 0]],
          mask=[[False, False, False],
                [False,  True, False],
                [False, False, False]],
          fill_value=1)
        >>> ma.mask_rowcols(a)
        masked_array(
          data=[[0, --, 0],
                [--, --, --],
                [0, --, 0]],
          mask=[[False,  True, False],
                [ True,  True,  True],
                [False,  True, False]],
          fill_value=1)
    
        
    """
def mask_rows(a, axis = ...):
    """
    
        Mask rows of a 2D array that contain masked values.
    
        This function is a shortcut to ``mask_rowcols`` with `axis` equal to 0.
    
        See Also
        --------
        mask_rowcols : Mask rows and/or columns of a 2D array.
        masked_where : Mask where a condition is met.
    
        Examples
        --------
        >>> import numpy.ma as ma
        >>> a = np.zeros((3, 3), dtype=int)
        >>> a[1, 1] = 1
        >>> a
        array([[0, 0, 0],
               [0, 1, 0],
               [0, 0, 0]])
        >>> a = ma.masked_equal(a, 1)
        >>> a
        masked_array(
          data=[[0, 0, 0],
                [0, --, 0],
                [0, 0, 0]],
          mask=[[False, False, False],
                [False,  True, False],
                [False, False, False]],
          fill_value=1)
    
        >>> ma.mask_rows(a)
        masked_array(
          data=[[0, 0, 0],
                [--, --, --],
                [0, 0, 0]],
          mask=[[False, False, False],
                [ True,  True,  True],
                [False, False, False]],
          fill_value=1)
    
        
    """
def masked_all(shape, dtype = float):
    """
    
        Empty masked array with all elements masked.
    
        Return an empty masked array of the given shape and dtype, where all the
        data are masked.
    
        Parameters
        ----------
        shape : int or tuple of ints
            Shape of the required MaskedArray, e.g., ``(2, 3)`` or ``2``.
        dtype : dtype, optional
            Data type of the output.
    
        Returns
        -------
        a : MaskedArray
            A masked array with all data masked.
    
        See Also
        --------
        masked_all_like : Empty masked array modelled on an existing array.
    
        Examples
        --------
        >>> import numpy.ma as ma
        >>> ma.masked_all((3, 3))
        masked_array(
          data=[[--, --, --],
                [--, --, --],
                [--, --, --]],
          mask=[[ True,  True,  True],
                [ True,  True,  True],
                [ True,  True,  True]],
          fill_value=1e+20,
          dtype=float64)
    
        The `dtype` parameter defines the underlying data type.
    
        >>> a = ma.masked_all((3, 3))
        >>> a.dtype
        dtype('float64')
        >>> a = ma.masked_all((3, 3), dtype=np.int32)
        >>> a.dtype
        dtype('int32')
    
        
    """
def masked_all_like(arr):
    """
    
        Empty masked array with the properties of an existing array.
    
        Return an empty masked array of the same shape and dtype as
        the array `arr`, where all the data are masked.
    
        Parameters
        ----------
        arr : ndarray
            An array describing the shape and dtype of the required MaskedArray.
    
        Returns
        -------
        a : MaskedArray
            A masked array with all data masked.
    
        Raises
        ------
        AttributeError
            If `arr` doesn't have a shape attribute (i.e. not an ndarray)
    
        See Also
        --------
        masked_all : Empty masked array with all elements masked.
    
        Examples
        --------
        >>> import numpy.ma as ma
        >>> arr = np.zeros((2, 3), dtype=np.float32)
        >>> arr
        array([[0., 0., 0.],
               [0., 0., 0.]], dtype=float32)
        >>> ma.masked_all_like(arr)
        masked_array(
          data=[[--, --, --],
                [--, --, --]],
          mask=[[ True,  True,  True],
                [ True,  True,  True]],
          fill_value=1e+20,
          dtype=float32)
    
        The dtype of the masked array matches the dtype of `arr`.
    
        >>> arr.dtype
        dtype('float32')
        >>> ma.masked_all_like(arr).dtype
        dtype('float32')
    
        
    """
def median(a, axis = None, out = None, overwrite_input = False, keepdims = False):
    """
    
        Compute the median along the specified axis.
    
        Returns the median of the array elements.
    
        Parameters
        ----------
        a : array_like
            Input array or object that can be converted to an array.
        axis : int, optional
            Axis along which the medians are computed. The default (None) is
            to compute the median along a flattened version of the array.
        out : ndarray, optional
            Alternative output array in which to place the result. It must
            have the same shape and buffer length as the expected output
            but the type will be cast if necessary.
        overwrite_input : bool, optional
            If True, then allow use of memory of input array (a) for
            calculations. The input array will be modified by the call to
            median. This will save memory when you do not need to preserve
            the contents of the input array. Treat the input as undefined,
            but it will probably be fully or partially sorted. Default is
            False. Note that, if `overwrite_input` is True, and the input
            is not already an `ndarray`, an error will be raised.
        keepdims : bool, optional
            If this is set to True, the axes which are reduced are left
            in the result as dimensions with size one. With this option,
            the result will broadcast correctly against the input array.
    
            .. versionadded:: 1.10.0
    
        Returns
        -------
        median : ndarray
            A new array holding the result is returned unless out is
            specified, in which case a reference to out is returned.
            Return data-type is `float64` for integers and floats smaller than
            `float64`, or the input data-type, otherwise.
    
        See Also
        --------
        mean
    
        Notes
        -----
        Given a vector ``V`` with ``N`` non masked values, the median of ``V``
        is the middle value of a sorted copy of ``V`` (``Vs``) - i.e.
        ``Vs[(N-1)/2]``, when ``N`` is odd, or ``{Vs[N/2 - 1] + Vs[N/2]}/2``
        when ``N`` is even.
    
        Examples
        --------
        >>> x = np.ma.array(np.arange(8), mask=[0]*4 + [1]*4)
        >>> np.ma.median(x)
        1.5
    
        >>> x = np.ma.array(np.arange(10).reshape(2, 5), mask=[0]*6 + [1]*4)
        >>> np.ma.median(x)
        2.5
        >>> np.ma.median(x, axis=-1, overwrite_input=True)
        masked_array(data=[2.0, 5.0],
                     mask=[False, False],
               fill_value=1e+20)
    
        
    """
def ndenumerate(a, compressed = True):
    """
    
        Multidimensional index iterator.
    
        Return an iterator yielding pairs of array coordinates and values,
        skipping elements that are masked. With `compressed=False`,
        `ma.masked` is yielded as the value of masked elements. This
        behavior differs from that of `numpy.ndenumerate`, which yields the
        value of the underlying data array.
    
        Notes
        -----
        .. versionadded:: 1.23.0
    
        Parameters
        ----------
        a : array_like
            An array with (possibly) masked elements.
        compressed : bool, optional
            If True (default), masked elements are skipped.
    
        See Also
        --------
        numpy.ndenumerate : Equivalent function ignoring any mask.
    
        Examples
        --------
        >>> a = np.ma.arange(9).reshape((3, 3))
        >>> a[1, 0] = np.ma.masked
        >>> a[1, 2] = np.ma.masked
        >>> a[2, 1] = np.ma.masked
        >>> a
        masked_array(
          data=[[0, 1, 2],
                [--, 4, --],
                [6, --, 8]],
          mask=[[False, False, False],
                [ True, False,  True],
                [False,  True, False]],
          fill_value=999999)
        >>> for index, x in np.ma.ndenumerate(a):
        ...     print(index, x)
        (0, 0) 0
        (0, 1) 1
        (0, 2) 2
        (1, 1) 4
        (2, 0) 6
        (2, 2) 8
    
        >>> for index, x in np.ma.ndenumerate(a, compressed=False):
        ...     print(index, x)
        (0, 0) 0
        (0, 1) 1
        (0, 2) 2
        (1, 0) --
        (1, 1) 4
        (1, 2) --
        (2, 0) 6
        (2, 1) --
        (2, 2) 8
        
    """
def notmasked_contiguous(a, axis = None):
    """
    
        Find contiguous unmasked data in a masked array along the given axis.
    
        Parameters
        ----------
        a : array_like
            The input array.
        axis : int, optional
            Axis along which to perform the operation.
            If None (default), applies to a flattened version of the array, and this
            is the same as `flatnotmasked_contiguous`.
    
        Returns
        -------
        endpoints : list
            A list of slices (start and end indexes) of unmasked indexes
            in the array.
    
            If the input is 2d and axis is specified, the result is a list of lists.
    
        See Also
        --------
        flatnotmasked_edges, flatnotmasked_contiguous, notmasked_edges
        clump_masked, clump_unmasked
    
        Notes
        -----
        Only accepts 2-D arrays at most.
    
        Examples
        --------
        >>> a = np.arange(12).reshape((3, 4))
        >>> mask = np.zeros_like(a)
        >>> mask[1:, :-1] = 1; mask[0, 1] = 1; mask[-1, 0] = 0
        >>> ma = np.ma.array(a, mask=mask)
        >>> ma
        masked_array(
          data=[[0, --, 2, 3],
                [--, --, --, 7],
                [8, --, --, 11]],
          mask=[[False,  True, False, False],
                [ True,  True,  True, False],
                [False,  True,  True, False]],
          fill_value=999999)
        >>> np.array(ma[~ma.mask])
        array([ 0,  2,  3,  7, 8, 11])
    
        >>> np.ma.notmasked_contiguous(ma)
        [slice(0, 1, None), slice(2, 4, None), slice(7, 9, None), slice(11, 12, None)]
    
        >>> np.ma.notmasked_contiguous(ma, axis=0)
        [[slice(0, 1, None), slice(2, 3, None)], [], [slice(0, 1, None)], [slice(0, 3, None)]]
    
        >>> np.ma.notmasked_contiguous(ma, axis=1)
        [[slice(0, 1, None), slice(2, 4, None)], [slice(3, 4, None)], [slice(0, 1, None), slice(3, 4, None)]]
    
        
    """
def notmasked_edges(a, axis = None):
    """
    
        Find the indices of the first and last unmasked values along an axis.
    
        If all values are masked, return None.  Otherwise, return a list
        of two tuples, corresponding to the indices of the first and last
        unmasked values respectively.
    
        Parameters
        ----------
        a : array_like
            The input array.
        axis : int, optional
            Axis along which to perform the operation.
            If None (default), applies to a flattened version of the array.
    
        Returns
        -------
        edges : ndarray or list
            An array of start and end indexes if there are any masked data in
            the array. If there are no masked data in the array, `edges` is a
            list of the first and last index.
    
        See Also
        --------
        flatnotmasked_contiguous, flatnotmasked_edges, notmasked_contiguous
        clump_masked, clump_unmasked
    
        Examples
        --------
        >>> a = np.arange(9).reshape((3, 3))
        >>> m = np.zeros_like(a)
        >>> m[1:, 1:] = 1
    
        >>> am = np.ma.array(a, mask=m)
        >>> np.array(am[~am.mask])
        array([0, 1, 2, 3, 6])
    
        >>> np.ma.notmasked_edges(am)
        array([0, 6])
    
        
    """
def polyfit(x, y, deg, rcond = None, full = False, w = None, cov = False):
    """
    Least squares polynomial fit.
    
    .. note::
       This forms part of the old polynomial API. Since version 1.4, the
       new polynomial API defined in `numpy.polynomial` is preferred.
       A summary of the differences can be found in the
       :doc:`transition guide </reference/routines.polynomials>`.
    
    Fit a polynomial ``p(x) = p[0] * x**deg + ... + p[deg]`` of degree `deg`
    to points `(x, y)`. Returns a vector of coefficients `p` that minimises
    the squared error in the order `deg`, `deg-1`, ... `0`.
    
    The `Polynomial.fit <numpy.polynomial.polynomial.Polynomial.fit>` class
    method is recommended for new code as it is more stable numerically. See
    the documentation of the method for more information.
    
    Parameters
    ----------
    x : array_like, shape (M,)
        x-coordinates of the M sample points ``(x[i], y[i])``.
    y : array_like, shape (M,) or (M, K)
        y-coordinates of the sample points. Several data sets of sample
        points sharing the same x-coordinates can be fitted at once by
        passing in a 2D-array that contains one dataset per column.
    deg : int
        Degree of the fitting polynomial
    rcond : float, optional
        Relative condition number of the fit. Singular values smaller than
        this relative to the largest singular value will be ignored. The
        default value is len(x)*eps, where eps is the relative precision of
        the float type, about 2e-16 in most cases.
    full : bool, optional
        Switch determining nature of return value. When it is False (the
        default) just the coefficients are returned, when True diagnostic
        information from the singular value decomposition is also returned.
    w : array_like, shape (M,), optional
        Weights. If not None, the weight ``w[i]`` applies to the unsquared
        residual ``y[i] - y_hat[i]`` at ``x[i]``. Ideally the weights are
        chosen so that the errors of the products ``w[i]*y[i]`` all have the
        same variance.  When using inverse-variance weighting, use
        ``w[i] = 1/sigma(y[i])``.  The default value is None.
    cov : bool or str, optional
        If given and not `False`, return not just the estimate but also its
        covariance matrix. By default, the covariance are scaled by
        chi2/dof, where dof = M - (deg + 1), i.e., the weights are presumed
        to be unreliable except in a relative sense and everything is scaled
        such that the reduced chi2 is unity. This scaling is omitted if
        ``cov='unscaled'``, as is relevant for the case that the weights are
        w = 1/sigma, with sigma known to be a reliable estimate of the
        uncertainty.
    
    Returns
    -------
    p : ndarray, shape (deg + 1,) or (deg + 1, K)
        Polynomial coefficients, highest power first.  If `y` was 2-D, the
        coefficients for `k`-th data set are in ``p[:,k]``.
    
    residuals, rank, singular_values, rcond
        These values are only returned if ``full == True``
    
        - residuals -- sum of squared residuals of the least squares fit
        - rank -- the effective rank of the scaled Vandermonde
           coefficient matrix
        - singular_values -- singular values of the scaled Vandermonde
           coefficient matrix
        - rcond -- value of `rcond`.
    
        For more details, see `numpy.linalg.lstsq`.
    
    V : ndarray, shape (M,M) or (M,M,K)
        Present only if ``full == False`` and ``cov == True``.  The covariance
        matrix of the polynomial coefficient estimates.  The diagonal of
        this matrix are the variance estimates for each coefficient.  If y
        is a 2-D array, then the covariance matrix for the `k`-th data set
        are in ``V[:,:,k]``
    
    
    Warns
    -----
    RankWarning
        The rank of the coefficient matrix in the least-squares fit is
        deficient. The warning is only raised if ``full == False``.
    
        The warnings can be turned off by
    
        >>> import warnings
        >>> warnings.simplefilter('ignore', np.RankWarning)
    
    See Also
    --------
    polyval : Compute polynomial values.
    linalg.lstsq : Computes a least-squares fit.
    scipy.interpolate.UnivariateSpline : Computes spline fits.
    
    Notes
    -----
    Any masked values in x is propagated in y, and vice-versa.
    
    The solution minimizes the squared error
    
    .. math::
        E = \\sum_{j=0}^k |p(x_j) - y_j|^2
    
    in the equations::
    
        x[0]**n * p[0] + ... + x[0] * p[n-1] + p[n] = y[0]
        x[1]**n * p[0] + ... + x[1] * p[n-1] + p[n] = y[1]
        ...
        x[k]**n * p[0] + ... + x[k] * p[n-1] + p[n] = y[k]
    
    The coefficient matrix of the coefficients `p` is a Vandermonde matrix.
    
    `polyfit` issues a `RankWarning` when the least-squares fit is badly
    conditioned. This implies that the best fit is not well-defined due
    to numerical error. The results may be improved by lowering the polynomial
    degree or by replacing `x` by `x` - `x`.mean(). The `rcond` parameter
    can also be set to a value smaller than its default, but the resulting
    fit may be spurious: including contributions from the small singular
    values can add numerical noise to the result.
    
    Note that fitting polynomial coefficients is inherently badly conditioned
    when the degree of the polynomial is large or the interval of sample points
    is badly centered. The quality of the fit should always be checked in these
    cases. When polynomial fits are not satisfactory, splines may be a good
    alternative.
    
    References
    ----------
    .. [1] Wikipedia, "Curve fitting",
           https://en.wikipedia.org/wiki/Curve_fitting
    .. [2] Wikipedia, "Polynomial interpolation",
           https://en.wikipedia.org/wiki/Polynomial_interpolation
    
    Examples
    --------
    >>> import warnings
    >>> x = np.array([0.0, 1.0, 2.0, 3.0,  4.0,  5.0])
    >>> y = np.array([0.0, 0.8, 0.9, 0.1, -0.8, -1.0])
    >>> z = np.polyfit(x, y, 3)
    >>> z
    array([ 0.08703704, -0.81349206,  1.69312169, -0.03968254]) # may vary
    
    It is convenient to use `poly1d` objects for dealing with polynomials:
    
    >>> p = np.poly1d(z)
    >>> p(0.5)
    0.6143849206349179 # may vary
    >>> p(3.5)
    -0.34732142857143039 # may vary
    >>> p(10)
    22.579365079365115 # may vary
    
    High-order polynomials may oscillate wildly:
    
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter('ignore', np.RankWarning)
    ...     p30 = np.poly1d(np.polyfit(x, y, 30))
    ...
    >>> p30(4)
    -0.80000000000000204 # may vary
    >>> p30(5)
    -0.99999999999999445 # may vary
    >>> p30(4.5)
    -0.10547061179440398 # may vary
    
    Illustration:
    
    >>> import matplotlib.pyplot as plt
    >>> xp = np.linspace(-2, 6, 100)
    >>> _ = plt.plot(x, y, '.', xp, p(xp), '-', xp, p30(xp), '--')
    >>> plt.ylim(-2,2)
    (-2, 2)
    >>> plt.show()
    """
def setdiff1d(ar1, ar2, assume_unique = False):
    """
    
        Set difference of 1D arrays with unique elements.
    
        The output is always a masked array. See `numpy.setdiff1d` for more
        details.
    
        See Also
        --------
        numpy.setdiff1d : Equivalent function for ndarrays.
    
        Examples
        --------
        >>> x = np.ma.array([1, 2, 3, 4], mask=[0, 1, 0, 1])
        >>> np.ma.setdiff1d(x, [1, 2])
        masked_array(data=[3, --],
                     mask=[False,  True],
               fill_value=999999)
    
        
    """
def setxor1d(ar1, ar2, assume_unique = False):
    """
    
        Set exclusive-or of 1-D arrays with unique elements.
    
        The output is always a masked array. See `numpy.setxor1d` for more details.
    
        See Also
        --------
        numpy.setxor1d : Equivalent function for ndarrays.
    
        
    """
def union1d(ar1, ar2):
    """
    
        Union of two arrays.
    
        The output is always a masked array. See `numpy.union1d` for more details.
    
        See Also
        --------
        numpy.union1d : Equivalent function for ndarrays.
    
        
    """
def unique(ar1, return_index = False, return_inverse = False):
    """
    
        Finds the unique elements of an array.
    
        Masked values are considered the same element (masked). The output array
        is always a masked array. See `numpy.unique` for more details.
    
        See Also
        --------
        numpy.unique : Equivalent function for ndarrays.
    
        Examples
        --------
        >>> import numpy.ma as ma
        >>> a = [1, 2, 1000, 2, 3]
        >>> mask = [0, 0, 1, 0, 0]
        >>> masked_a = ma.masked_array(a, mask)
        >>> masked_a
        masked_array(data=[1, 2, --, 2, 3],
                    mask=[False, False,  True, False, False],
            fill_value=999999)
        >>> ma.unique(masked_a)
        masked_array(data=[1, 2, 3, --],
                    mask=[False, False, False,  True],
            fill_value=999999)
        >>> ma.unique(masked_a, return_index=True)
        (masked_array(data=[1, 2, 3, --],
                    mask=[False, False, False,  True],
            fill_value=999999), array([0, 1, 4, 2]))
        >>> ma.unique(masked_a, return_inverse=True)
        (masked_array(data=[1, 2, 3, --],
                    mask=[False, False, False,  True],
            fill_value=999999), array([0, 1, 3, 1, 2]))
        >>> ma.unique(masked_a, return_index=True, return_inverse=True)
        (masked_array(data=[1, 2, 3, --],
                    mask=[False, False, False,  True],
            fill_value=999999), array([0, 1, 4, 2]), array([0, 1, 3, 1, 2]))
        
    """
def vander(x, n = None):
    """
    Generate a Vandermonde matrix.
    
    The columns of the output matrix are powers of the input vector. The
    order of the powers is determined by the `increasing` boolean argument.
    Specifically, when `increasing` is False, the `i`-th output column is
    the input vector raised element-wise to the power of ``N - i - 1``. Such
    a matrix with a geometric progression in each row is named for Alexandre-
    Theophile Vandermonde.
    
    Parameters
    ----------
    x : array_like
        1-D input array.
    N : int, optional
        Number of columns in the output.  If `N` is not specified, a square
        array is returned (``N = len(x)``).
    increasing : bool, optional
        Order of the powers of the columns.  If True, the powers increase
        from left to right, if False (the default) they are reversed.
    
        .. versionadded:: 1.9.0
    
    Returns
    -------
    out : ndarray
        Vandermonde matrix.  If `increasing` is False, the first column is
        ``x^(N-1)``, the second ``x^(N-2)`` and so forth. If `increasing` is
        True, the columns are ``x^0, x^1, ..., x^(N-1)``.
    
    See Also
    --------
    polynomial.polynomial.polyvander
    
    Examples
    --------
    >>> x = np.array([1, 2, 3, 5])
    >>> N = 3
    >>> np.vander(x, N)
    array([[ 1,  1,  1],
           [ 4,  2,  1],
           [ 9,  3,  1],
           [25,  5,  1]])
    
    >>> np.column_stack([x**(N-1-i) for i in range(N)])
    array([[ 1,  1,  1],
           [ 4,  2,  1],
           [ 9,  3,  1],
           [25,  5,  1]])
    
    >>> x = np.array([1, 2, 3, 5])
    >>> np.vander(x)
    array([[  1,   1,   1,   1],
           [  8,   4,   2,   1],
           [ 27,   9,   3,   1],
           [125,  25,   5,   1]])
    >>> np.vander(x, increasing=True)
    array([[  1,   1,   1,   1],
           [  1,   2,   4,   8],
           [  1,   3,   9,  27],
           [  1,   5,  25, 125]])
    
    The determinant of a square Vandermonde matrix is the product
    of the differences between the values of the input vector:
    
    >>> np.linalg.det(np.vander(x))
    48.000000000000043 # may vary
    >>> (5-3)*(5-2)*(5-1)*(3-2)*(3-1)*(2-1)
    48
    
    Notes
    -----
    Masked values in the input array result in rows of zeros.
    """
add: numpy.ma.core._MaskedBinaryOperation  # value = <numpy.ma.core._MaskedBinaryOperation object>
atleast_1d: _fromnxfunction_allargs  # value = <numpy.ma.extras._fromnxfunction_allargs object>
atleast_2d: _fromnxfunction_allargs  # value = <numpy.ma.extras._fromnxfunction_allargs object>
atleast_3d: _fromnxfunction_allargs  # value = <numpy.ma.extras._fromnxfunction_allargs object>
column_stack: _fromnxfunction_seq  # value = <numpy.ma.extras._fromnxfunction_seq object>
count: numpy.ma.core._frommethod  # value = <numpy.ma.core._frommethod object>
diagflat: _fromnxfunction_single  # value = <numpy.ma.extras._fromnxfunction_single object>
dstack: _fromnxfunction_seq  # value = <numpy.ma.extras._fromnxfunction_seq object>
hsplit: _fromnxfunction_single  # value = <numpy.ma.extras._fromnxfunction_single object>
hstack: _fromnxfunction_seq  # value = <numpy.ma.extras._fromnxfunction_seq object>
masked: numpy.ma.core.MaskedConstant  # value = masked
mr_: mr_class  # value = <numpy.ma.extras.mr_class object>
nomask: numpy.bool_  # value = False
ones: numpy.ma.core._convert2ma  # value = <numpy.ma.core._convert2ma object>
row_stack: _fromnxfunction_seq  # value = <numpy.ma.extras._fromnxfunction_seq object>
stack: _fromnxfunction_seq  # value = <numpy.ma.extras._fromnxfunction_seq object>
vstack: _fromnxfunction_seq  # value = <numpy.ma.extras._fromnxfunction_seq object>
zeros: numpy.ma.core._convert2ma  # value = <numpy.ma.core._convert2ma object>
