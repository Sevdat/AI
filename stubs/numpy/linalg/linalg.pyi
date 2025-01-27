"""
Lite version of scipy.linalg.

Notes
-----
This module is a lite version of the linalg.py module in SciPy which
contains high-level Python interface to the LAPACK library.  The lite
version only accesses the following LAPACK functions: dgesv, zgesv,
dgeev, zgeev, dgesdd, zgesdd, dgelsd, zgelsd, dsyevd, zheevd, dgetrf,
zgetrf, dpotrf, zpotrf, dgeqrf, zgeqrf, zungqr, dorgqr.
"""
from __future__ import annotations
import functools as functools
import numpy
from numpy._utils import set_module
from numpy import all
from numpy import amax
from numpy import amin
from numpy import argsort
from numpy import array
from numpy import asanyarray
from numpy import asarray
from numpy import atleast_2d
from numpy import complex128 as cdouble
from numpy import complex64 as csingle
from numpy import complexfloating
from numpy.core._multiarray_umath import normalize_axis_index
from numpy.core import overrides
from numpy import count_nonzero
from numpy import dot
from numpy import empty
from numpy import empty_like
from numpy import errstate
from numpy import eye
from numpy import finfo
from numpy import float32 as single
from numpy import float64 as double
from numpy import geterrobj
from numpy import inexact
from numpy import int64 as intp
from numpy import intc as fortran_int
from numpy import intc
from numpy.linalg import LinAlgError
from numpy.linalg import _umath_linalg
from numpy.linalg import cholesky
from numpy.linalg import cond
from numpy.linalg import det
from numpy.linalg import eig
from numpy.linalg import eigh
from numpy.linalg import eigvals
from numpy.linalg import eigvalsh
from numpy.linalg import inv
from numpy.linalg import lstsq
from numpy.linalg import matrix_power
from numpy.linalg import matrix_rank
from numpy.linalg import multi_dot
from numpy.linalg import norm
from numpy.linalg import pinv
from numpy.linalg import qr
from numpy.linalg import slogdet
from numpy.linalg import solve
from numpy.linalg import svd
from numpy.linalg import tensorinv
from numpy.linalg import tensorsolve
from numpy import moveaxis
from numpy import ndarray as NDArray
from numpy import object_
from numpy import prod
from numpy import sort
from numpy import sum
from numpy import swapaxes
from numpy import triu
from numpy import zeros
import operator as operator
import typing
from typing import NamedTuple
import warnings as warnings
__all__: list = ['matrix_power', 'solve', 'tensorsolve', 'tensorinv', 'inv', 'cholesky', 'eigvals', 'eigvalsh', 'pinv', 'slogdet', 'det', 'svd', 'eig', 'eigh', 'lstsq', 'norm', 'qr', 'cond', 'matrix_rank', 'LinAlgError', 'multi_dot']
class EigResult(tuple):
    """
    EigResult(eigenvalues, eigenvectors)
    """
    __match_args__: typing.ClassVar[tuple] = ('eigenvalues', 'eigenvectors')
    __orig_bases__: typing.ClassVar[tuple] = (typing.NamedTuple)
    __slots__: typing.ClassVar[tuple] = tuple()
    _field_defaults: typing.ClassVar[dict] = {}
    _fields: typing.ClassVar[tuple] = ('eigenvalues', 'eigenvectors')
    @staticmethod
    def __new__(_cls, eigenvalues: numpy.ndarray[typing.Any, numpy.dtype[typing.Any]], eigenvectors: numpy.ndarray[typing.Any, numpy.dtype[typing.Any]]):
        """
        Create new instance of EigResult(eigenvalues, eigenvectors)
        """
    @classmethod
    def _make(cls, iterable):
        """
        Make a new EigResult object from a sequence or iterable
        """
    def __getnewargs__(self):
        """
        Return self as a plain tuple.  Used by copy and pickle.
        """
    def __repr__(self):
        """
        Return a nicely formatted representation string
        """
    def _asdict(self):
        """
        Return a new dict which maps field names to their values.
        """
    def _replace(self, **kwds):
        """
        Return a new EigResult object replacing specified fields with new values
        """
class EighResult(tuple):
    """
    EighResult(eigenvalues, eigenvectors)
    """
    __match_args__: typing.ClassVar[tuple] = ('eigenvalues', 'eigenvectors')
    __orig_bases__: typing.ClassVar[tuple] = (typing.NamedTuple)
    __slots__: typing.ClassVar[tuple] = tuple()
    _field_defaults: typing.ClassVar[dict] = {}
    _fields: typing.ClassVar[tuple] = ('eigenvalues', 'eigenvectors')
    @staticmethod
    def __new__(_cls, eigenvalues: numpy.ndarray[typing.Any, numpy.dtype[typing.Any]], eigenvectors: numpy.ndarray[typing.Any, numpy.dtype[typing.Any]]):
        """
        Create new instance of EighResult(eigenvalues, eigenvectors)
        """
    @classmethod
    def _make(cls, iterable):
        """
        Make a new EighResult object from a sequence or iterable
        """
    def __getnewargs__(self):
        """
        Return self as a plain tuple.  Used by copy and pickle.
        """
    def __repr__(self):
        """
        Return a nicely formatted representation string
        """
    def _asdict(self):
        """
        Return a new dict which maps field names to their values.
        """
    def _replace(self, **kwds):
        """
        Return a new EighResult object replacing specified fields with new values
        """
class QRResult(tuple):
    """
    QRResult(Q, R)
    """
    __match_args__: typing.ClassVar[tuple] = ('Q', 'R')
    __orig_bases__: typing.ClassVar[tuple] = (typing.NamedTuple)
    __slots__: typing.ClassVar[tuple] = tuple()
    _field_defaults: typing.ClassVar[dict] = {}
    _fields: typing.ClassVar[tuple] = ('Q', 'R')
    @staticmethod
    def __new__(_cls, Q: numpy.ndarray[typing.Any, numpy.dtype[typing.Any]], R: numpy.ndarray[typing.Any, numpy.dtype[typing.Any]]):
        """
        Create new instance of QRResult(Q, R)
        """
    @classmethod
    def _make(cls, iterable):
        """
        Make a new QRResult object from a sequence or iterable
        """
    def __getnewargs__(self):
        """
        Return self as a plain tuple.  Used by copy and pickle.
        """
    def __repr__(self):
        """
        Return a nicely formatted representation string
        """
    def _asdict(self):
        """
        Return a new dict which maps field names to their values.
        """
    def _replace(self, **kwds):
        """
        Return a new QRResult object replacing specified fields with new values
        """
class SVDResult(tuple):
    """
    SVDResult(U, S, Vh)
    """
    __match_args__: typing.ClassVar[tuple] = ('U', 'S', 'Vh')
    __orig_bases__: typing.ClassVar[tuple] = (typing.NamedTuple)
    __slots__: typing.ClassVar[tuple] = tuple()
    _field_defaults: typing.ClassVar[dict] = {}
    _fields: typing.ClassVar[tuple] = ('U', 'S', 'Vh')
    @staticmethod
    def __new__(_cls, U: numpy.ndarray[typing.Any, numpy.dtype[typing.Any]], S: numpy.ndarray[typing.Any, numpy.dtype[typing.Any]], Vh: numpy.ndarray[typing.Any, numpy.dtype[typing.Any]]):
        """
        Create new instance of SVDResult(U, S, Vh)
        """
    @classmethod
    def _make(cls, iterable):
        """
        Make a new SVDResult object from a sequence or iterable
        """
    def __getnewargs__(self):
        """
        Return self as a plain tuple.  Used by copy and pickle.
        """
    def __repr__(self):
        """
        Return a nicely formatted representation string
        """
    def _asdict(self):
        """
        Return a new dict which maps field names to their values.
        """
    def _replace(self, **kwds):
        """
        Return a new SVDResult object replacing specified fields with new values
        """
class SlogdetResult(tuple):
    """
    SlogdetResult(sign, logabsdet)
    """
    __match_args__: typing.ClassVar[tuple] = ('sign', 'logabsdet')
    __orig_bases__: typing.ClassVar[tuple] = (typing.NamedTuple)
    __slots__: typing.ClassVar[tuple] = tuple()
    _field_defaults: typing.ClassVar[dict] = {}
    _fields: typing.ClassVar[tuple] = ('sign', 'logabsdet')
    @staticmethod
    def __new__(_cls, sign: numpy.ndarray[typing.Any, numpy.dtype[typing.Any]], logabsdet: numpy.ndarray[typing.Any, numpy.dtype[typing.Any]]):
        """
        Create new instance of SlogdetResult(sign, logabsdet)
        """
    @classmethod
    def _make(cls, iterable):
        """
        Make a new SlogdetResult object from a sequence or iterable
        """
    def __getnewargs__(self):
        """
        Return self as a plain tuple.  Used by copy and pickle.
        """
    def __repr__(self):
        """
        Return a nicely formatted representation string
        """
    def _asdict(self):
        """
        Return a new dict which maps field names to their values.
        """
    def _replace(self, **kwds):
        """
        Return a new SlogdetResult object replacing specified fields with new values
        """
def _assert_2d(*arrays):
    ...
def _assert_finite(*arrays):
    ...
def _assert_stacked_2d(*arrays):
    ...
def _assert_stacked_square(*arrays):
    ...
def _commonType(*arrays):
    ...
def _complexType(t, default = numpy.complex128):
    ...
def _cond_dispatcher(x, p = None):
    ...
def _convertarray(a):
    ...
def _eigvalsh_dispatcher(a, UPLO = None):
    ...
def _is_empty_2d(arr):
    ...
def _lstsq_dispatcher(a, b, rcond = None):
    ...
def _makearray(a):
    ...
def _matrix_power_dispatcher(a, n):
    ...
def _matrix_rank_dispatcher(A, tol = None, hermitian = None):
    ...
def _multi_dot(arrays, order, i, j, out = None):
    """
    Actually do the multiplication with the given order.
    """
def _multi_dot_matrix_chain_order(arrays, return_costs = False):
    """
    
        Return a np.array that encodes the optimal order of mutiplications.
    
        The optimal order array is then used by `_multi_dot()` to do the
        multiplication.
    
        Also return the cost matrix if `return_costs` is `True`
    
        The implementation CLOSELY follows Cormen, "Introduction to Algorithms",
        Chapter 15.2, p. 370-378.  Note that Cormen uses 1-based indices.
    
            cost[i, j] = min([
                cost[prefix] + cost[suffix] + cost_mult(prefix, suffix)
                for k in range(i, j)])
    
        
    """
def _multi_dot_three(A, B, C, out = None):
    """
    
        Find the best order for three arrays and do the multiplication.
    
        For three arguments `_multi_dot_three` is approximately 15 times faster
        than `_multi_dot_matrix_chain_order`
    
        
    """
def _multi_svd_norm(x, row_axis, col_axis, op):
    """
    Compute a function of the singular values of the 2-D matrices in `x`.
    
        This is a private utility function used by `numpy.linalg.norm()`.
    
        Parameters
        ----------
        x : ndarray
        row_axis, col_axis : int
            The axes of `x` that hold the 2-D matrices.
        op : callable
            This should be either numpy.amin or `numpy.amax` or `numpy.sum`.
    
        Returns
        -------
        result : float or ndarray
            If `x` is 2-D, the return values is a float.
            Otherwise, it is an array with ``x.ndim - 2`` dimensions.
            The return values are either the minimum or maximum or sum of the
            singular values of the matrices, depending on whether `op`
            is `numpy.amin` or `numpy.amax` or `numpy.sum`.
    
        
    """
def _multidot_dispatcher(arrays, *, out = None):
    ...
def _norm_dispatcher(x, ord = None, axis = None, keepdims = None):
    ...
def _pinv_dispatcher(a, rcond = None, hermitian = None):
    ...
def _qr_dispatcher(a, mode = None):
    ...
def _raise_linalgerror_eigenvalues_nonconvergence(err, flag):
    ...
def _raise_linalgerror_lstsq(err, flag):
    ...
def _raise_linalgerror_nonposdef(err, flag):
    ...
def _raise_linalgerror_qr(err, flag):
    ...
def _raise_linalgerror_singular(err, flag):
    ...
def _raise_linalgerror_svd_nonconvergence(err, flag):
    ...
def _realType(t, default = numpy.float64):
    ...
def _solve_dispatcher(a, b):
    ...
def _svd_dispatcher(a, full_matrices = None, compute_uv = None, hermitian = None):
    ...
def _tensorinv_dispatcher(a, ind = None):
    ...
def _tensorsolve_dispatcher(a, b, axes = None):
    ...
def _to_native_byte_order(*arrays):
    ...
def _unary_dispatcher(a):
    ...
def get_linalg_error_extobj(callback):
    ...
def isComplexType(t):
    ...
def transpose(a):
    """
    
        Transpose each matrix in a stack of matrices.
    
        Unlike np.transpose, this only swaps the last two axes, rather than all of
        them
    
        Parameters
        ----------
        a : (...,M,N) array_like
    
        Returns
        -------
        aT : (...,N,M) ndarray
        
    """
Inf: float  # value = inf
_complex_types_map: dict = {numpy.float32: numpy.complex64, numpy.float64: numpy.complex128, numpy.complex64: numpy.complex64, numpy.complex128: numpy.complex128}
_linalg_error_extobj: list = [8192, 1536, None]
_real_types_map: dict = {numpy.float32: numpy.float32, numpy.float64: numpy.float64, numpy.complex64: numpy.float32, numpy.complex128: numpy.float64}
abs: numpy.ufunc  # value = <ufunc 'absolute'>
add: numpy.ufunc  # value = <ufunc 'add'>
array_function_dispatch: functools.partial  # value = functools.partial(<function array_function_dispatch at 0x00000230DBD7F640>, module='numpy.linalg')
divide: numpy.ufunc  # value = <ufunc 'divide'>
isfinite: numpy.ufunc  # value = <ufunc 'isfinite'>
isnan: numpy.ufunc  # value = <ufunc 'isnan'>
matmul: numpy.ufunc  # value = <ufunc 'matmul'>
multiply: numpy.ufunc  # value = <ufunc 'multiply'>
newaxis = None
reciprocal: numpy.ufunc  # value = <ufunc 'reciprocal'>
sign: numpy.ufunc  # value = <ufunc 'sign'>
sqrt: numpy.ufunc  # value = <ufunc 'sqrt'>
