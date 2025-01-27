"""

Utility function to facilitate testing.

"""
from __future__ import annotations
from _io import StringIO
import contextlib as contextlib
from functools import partial
from functools import wraps
import gc as gc
import numpy as numpy
import numpy as np
from numpy import arange
from numpy import array
from numpy import array_repr
from numpy import empty
from numpy import float32
from numpy import int64 as intp
from numpy import ndarray
import operator as operator
import os as os
import platform as platform
import pprint as pprint
import re as re
import shutil as shutil
import sys as sys
import sysconfig as sysconfig
from tempfile import mkdtemp
from tempfile import mkstemp
import typing
import unittest as unittest
from unittest.case import SkipTest
import warnings as warnings
from warnings import WarningMessage
__all__: list = ['assert_equal', 'assert_almost_equal', 'assert_approx_equal', 'assert_array_equal', 'assert_array_less', 'assert_string_equal', 'assert_array_almost_equal', 'assert_raises', 'build_err_msg', 'decorate_methods', 'jiffies', 'memusage', 'print_assert_equal', 'rundocs', 'runstring', 'verbose', 'measure', 'assert_', 'assert_array_almost_equal_nulp', 'assert_raises_regex', 'assert_array_max_ulp', 'assert_warns', 'assert_no_warnings', 'assert_allclose', 'IgnoreException', 'clear_and_catch_warnings', 'SkipTest', 'KnownFailureException', 'temppath', 'tempdir', 'IS_PYPY', 'HAS_REFCOUNT', 'IS_WASM', 'suppress_warnings', 'assert_array_compare', 'assert_no_gc_cycles', 'break_cycles', 'HAS_LAPACK64', 'IS_PYSTON', '_OLD_PROMOTION', 'IS_MUSL', '_SUPPORTS_SVE']
class IgnoreException(Exception):
    """
    Ignoring this exception due to disabled feature
    """
class KnownFailureException(Exception):
    """
    Raise this exception to mark a test as a known failing test.
    """
class _Dummy(unittest.case.TestCase):
    def nop(self):
        ...
class clear_and_catch_warnings(warnings.catch_warnings):
    """
     Context manager that resets warning registry for catching warnings
    
        Warnings can be slippery, because, whenever a warning is triggered, Python
        adds a ``__warningregistry__`` member to the *calling* module.  This makes
        it impossible to retrigger the warning in this module, whatever you put in
        the warnings filters.  This context manager accepts a sequence of `modules`
        as a keyword argument to its constructor and:
    
        * stores and removes any ``__warningregistry__`` entries in given `modules`
          on entry;
        * resets ``__warningregistry__`` to its previous state on exit.
    
        This makes it possible to trigger any warning afresh inside the context
        manager without disturbing the state of warnings outside.
    
        For compatibility with Python 3.0, please consider all arguments to be
        keyword-only.
    
        Parameters
        ----------
        record : bool, optional
            Specifies whether warnings should be captured by a custom
            implementation of ``warnings.showwarning()`` and be appended to a list
            returned by the context manager. Otherwise None is returned by the
            context manager. The objects appended to the list are arguments whose
            attributes mirror the arguments to ``showwarning()``.
        modules : sequence, optional
            Sequence of modules for which to reset warnings registry on entry and
            restore on exit. To work correctly, all 'ignore' filters should
            filter by one of these modules.
    
        Examples
        --------
        >>> import warnings
        >>> with np.testing.clear_and_catch_warnings(
        ...         modules=[np.core.fromnumeric]):
        ...     warnings.simplefilter('always')
        ...     warnings.filterwarnings('ignore', module='np.core.fromnumeric')
        ...     # do something that raises a warning but ignore those in
        ...     # np.core.fromnumeric
        
    """
    class_modules: typing.ClassVar[tuple] = tuple()
    def __enter__(self):
        ...
    def __exit__(self, *exc_info):
        ...
    def __init__(self, record = False, modules = tuple()):
        ...
class suppress_warnings:
    """
    
        Context manager and decorator doing much the same as
        ``warnings.catch_warnings``.
    
        However, it also provides a filter mechanism to work around
        https://bugs.python.org/issue4180.
    
        This bug causes Python before 3.4 to not reliably show warnings again
        after they have been ignored once (even within catch_warnings). It
        means that no "ignore" filter can be used easily, since following
        tests might need to see the warning. Additionally it allows easier
        specificity for testing warnings and can be nested.
    
        Parameters
        ----------
        forwarding_rule : str, optional
            One of "always", "once", "module", or "location". Analogous to
            the usual warnings module filter mode, it is useful to reduce
            noise mostly on the outmost level. Unsuppressed and unrecorded
            warnings will be forwarded based on this rule. Defaults to "always".
            "location" is equivalent to the warnings "default", match by exact
            location the warning warning originated from.
    
        Notes
        -----
        Filters added inside the context manager will be discarded again
        when leaving it. Upon entering all filters defined outside a
        context will be applied automatically.
    
        When a recording filter is added, matching warnings are stored in the
        ``log`` attribute as well as in the list returned by ``record``.
    
        If filters are added and the ``module`` keyword is given, the
        warning registry of this module will additionally be cleared when
        applying it, entering the context, or exiting it. This could cause
        warnings to appear a second time after leaving the context if they
        were configured to be printed once (default) and were already
        printed before the context was entered.
    
        Nesting this context manager will work as expected when the
        forwarding rule is "always" (default). Unfiltered and unrecorded
        warnings will be passed out and be matched by the outer level.
        On the outmost level they will be printed (or caught by another
        warnings context). The forwarding rule argument can modify this
        behaviour.
    
        Like ``catch_warnings`` this context manager is not threadsafe.
    
        Examples
        --------
    
        With a context manager::
    
            with np.testing.suppress_warnings() as sup:
                sup.filter(DeprecationWarning, "Some text")
                sup.filter(module=np.ma.core)
                log = sup.record(FutureWarning, "Does this occur?")
                command_giving_warnings()
                # The FutureWarning was given once, the filtered warnings were
                # ignored. All other warnings abide outside settings (may be
                # printed/error)
                assert_(len(log) == 1)
                assert_(len(sup.log) == 1)  # also stored in log attribute
    
        Or as a decorator::
    
            sup = np.testing.suppress_warnings()
            sup.filter(module=np.ma.core)  # module must match exactly
            @sup
            def some_function():
                # do something which causes a warning in np.ma.core
                pass
        
    """
    def __call__(self, func):
        """
        
                Function decorator to apply certain suppressions to a whole
                function.
                
        """
    def __enter__(self):
        ...
    def __exit__(self, *exc_info):
        ...
    def __init__(self, forwarding_rule = 'always'):
        ...
    def _clear_registries(self):
        ...
    def _filter(self, category = Warning, message = '', module = None, record = False):
        ...
    def _showwarning(self, message, category, filename, lineno, *args, use_warnmsg = None, **kwargs):
        ...
    def filter(self, category = Warning, message = '', module = None):
        """
        
                Add a new suppressing filter or apply it if the state is entered.
        
                Parameters
                ----------
                category : class, optional
                    Warning class to filter
                message : string, optional
                    Regular expression matching the warning message.
                module : module, optional
                    Module to filter for. Note that the module (and its file)
                    must match exactly and cannot be a submodule. This may make
                    it unreliable for external modules.
        
                Notes
                -----
                When added within a context, filters are only added inside
                the context and will be forgotten when the context is exited.
                
        """
    def record(self, category = Warning, message = '', module = None):
        """
        
                Append a new recording filter or apply it if the state is entered.
        
                All warnings matching will be appended to the ``log`` attribute.
        
                Parameters
                ----------
                category : class, optional
                    Warning class to filter
                message : string, optional
                    Regular expression matching the warning message.
                module : module, optional
                    Module to filter for. Note that the module (and its file)
                    must match exactly and cannot be a submodule. This may make
                    it unreliable for external modules.
        
                Returns
                -------
                log : list
                    A list which will be filled with all matched warnings.
        
                Notes
                -----
                When added within a context, filters are only added inside
                the context and will be forgotten when the context is exited.
                
        """
def GetPerformanceAttributes(object, counter, instance = None, inum = -1, format = None, machine = None):
    ...
def _assert_no_gc_cycles_context(*args, **kwds):
    ...
def _assert_no_warnings_context(*args, **kwds):
    ...
def _assert_valid_refcount(op):
    """
    
        Check that ufuncs don't mishandle refcount of object `1`.
        Used in a few regression tests.
        
    """
def _assert_warns_context(*args, **kwds):
    ...
def _gen_alignment_data(dtype = numpy.float32, type = 'binary', max_size = 24):
    """
    
        generator producing data with different alignment and offsets
        to test simd vectorization
    
        Parameters
        ----------
        dtype : dtype
            data type to produce
        type : string
            'unary': create data for unary operations, creates one input
                     and output array
            'binary': create data for unary operations, creates two input
                     and output array
        max_size : integer
            maximum size of data to produce
    
        Returns
        -------
        if type is 'unary' yields one output, one input array and a message
        containing information on the data
        if type is 'binary' yields one output array, two input array and a message
        containing information on the data
    
        
    """
def _get_glibc_version():
    ...
def _get_mem_available():
    """
    Return available memory in bytes, or None if unknown.
    """
def _integer_repr(x, vdt, comp):
    ...
def _no_tracing(func):
    """
    
        Decorator to temporarily turn off tracing for the duration of a test.
        Needed in tests that check refcounting, otherwise the tracing itself
        influences the refcounts
        
    """
def _parse_size(size_str):
    """
    Convert memory size strings ('12 GB' etc.) to float
    """
def assert_(val, msg = ''):
    """
    
        Assert that works in release mode.
        Accepts callable msg to allow deferring evaluation until failure.
    
        The Python built-in ``assert`` does not work when executing code in
        optimized mode (the ``-O`` flag) - no byte-code is generated for it.
    
        For documentation on usage, refer to the Python documentation.
    
        
    """
def assert_allclose(actual, desired, rtol = 1e-07, atol = 0, equal_nan = True, err_msg = '', verbose = True):
    """
    
        Raises an AssertionError if two objects are not equal up to desired
        tolerance.
    
        Given two array_like objects, check that their shapes and all elements
        are equal (but see the Notes for the special handling of a scalar). An
        exception is raised if the shapes mismatch or any values conflict. In
        contrast to the standard usage in numpy, NaNs are compared like numbers,
        no assertion is raised if both objects have NaNs in the same positions.
    
        The test is equivalent to ``allclose(actual, desired, rtol, atol)`` (note
        that ``allclose`` has different default values). It compares the difference
        between `actual` and `desired` to ``atol + rtol * abs(desired)``.
    
        .. versionadded:: 1.5.0
    
        Parameters
        ----------
        actual : array_like
            Array obtained.
        desired : array_like
            Array desired.
        rtol : float, optional
            Relative tolerance.
        atol : float, optional
            Absolute tolerance.
        equal_nan : bool, optional.
            If True, NaNs will compare equal.
        err_msg : str, optional
            The error message to be printed in case of failure.
        verbose : bool, optional
            If True, the conflicting values are appended to the error message.
    
        Raises
        ------
        AssertionError
            If actual and desired are not equal up to specified precision.
    
        See Also
        --------
        assert_array_almost_equal_nulp, assert_array_max_ulp
    
        Notes
        -----
        When one of `actual` and `desired` is a scalar and the other is
        array_like, the function checks that each element of the array_like
        object is equal to the scalar.
    
        Examples
        --------
        >>> x = [1e-5, 1e-3, 1e-1]
        >>> y = np.arccos(np.cos(x))
        >>> np.testing.assert_allclose(x, y, rtol=1e-5, atol=0)
    
        
    """
def assert_almost_equal(*args, **kwds):
    """
    
        Raises an AssertionError if two items are not equal up to desired
        precision.
    
        .. note:: It is recommended to use one of `assert_allclose`,
                  `assert_array_almost_equal_nulp` or `assert_array_max_ulp`
                  instead of this function for more consistent floating point
                  comparisons.
    
        The test verifies that the elements of `actual` and `desired` satisfy.
    
            ``abs(desired-actual) < float64(1.5 * 10**(-decimal))``
    
        That is a looser test than originally documented, but agrees with what the
        actual implementation in `assert_array_almost_equal` did up to rounding
        vagaries. An exception is raised at conflicting values. For ndarrays this
        delegates to assert_array_almost_equal
    
        Parameters
        ----------
        actual : array_like
            The object to check.
        desired : array_like
            The expected object.
        decimal : int, optional
            Desired precision, default is 7.
        err_msg : str, optional
            The error message to be printed in case of failure.
        verbose : bool, optional
            If True, the conflicting values are appended to the error message.
    
        Raises
        ------
        AssertionError
          If actual and desired are not equal up to specified precision.
    
        See Also
        --------
        assert_allclose: Compare two array_like objects for equality with desired
                         relative and/or absolute precision.
        assert_array_almost_equal_nulp, assert_array_max_ulp, assert_equal
    
        Examples
        --------
        >>> from numpy.testing import assert_almost_equal
        >>> assert_almost_equal(2.3333333333333, 2.33333334)
        >>> assert_almost_equal(2.3333333333333, 2.33333334, decimal=10)
        Traceback (most recent call last):
            ...
        AssertionError:
        Arrays are not almost equal to 10 decimals
         ACTUAL: 2.3333333333333
         DESIRED: 2.33333334
    
        >>> assert_almost_equal(np.array([1.0,2.3333333333333]),
        ...                     np.array([1.0,2.33333334]), decimal=9)
        Traceback (most recent call last):
            ...
        AssertionError:
        Arrays are not almost equal to 9 decimals
        <BLANKLINE>
        Mismatched elements: 1 / 2 (50%)
        Max absolute difference: 6.66669964e-09
        Max relative difference: 2.85715698e-09
         x: array([1.         , 2.333333333])
         y: array([1.        , 2.33333334])
    
        
    """
def assert_approx_equal(*args, **kwds):
    """
    
        Raises an AssertionError if two items are not equal up to significant
        digits.
    
        .. note:: It is recommended to use one of `assert_allclose`,
                  `assert_array_almost_equal_nulp` or `assert_array_max_ulp`
                  instead of this function for more consistent floating point
                  comparisons.
    
        Given two numbers, check that they are approximately equal.
        Approximately equal is defined as the number of significant digits
        that agree.
    
        Parameters
        ----------
        actual : scalar
            The object to check.
        desired : scalar
            The expected object.
        significant : int, optional
            Desired precision, default is 7.
        err_msg : str, optional
            The error message to be printed in case of failure.
        verbose : bool, optional
            If True, the conflicting values are appended to the error message.
    
        Raises
        ------
        AssertionError
          If actual and desired are not equal up to specified precision.
    
        See Also
        --------
        assert_allclose: Compare two array_like objects for equality with desired
                         relative and/or absolute precision.
        assert_array_almost_equal_nulp, assert_array_max_ulp, assert_equal
    
        Examples
        --------
        >>> np.testing.assert_approx_equal(0.12345677777777e-20, 0.1234567e-20)
        >>> np.testing.assert_approx_equal(0.12345670e-20, 0.12345671e-20,
        ...                                significant=8)
        >>> np.testing.assert_approx_equal(0.12345670e-20, 0.12345672e-20,
        ...                                significant=8)
        Traceback (most recent call last):
            ...
        AssertionError:
        Items are not equal to 8 significant digits:
         ACTUAL: 1.234567e-21
         DESIRED: 1.2345672e-21
    
        the evaluated condition that raises the exception is
    
        >>> abs(0.12345670e-20/1e-21 - 0.12345672e-20/1e-21) >= 10**-(8-1)
        True
    
        
    """
def assert_array_almost_equal(*args, **kwds):
    """
    
        Raises an AssertionError if two objects are not equal up to desired
        precision.
    
        .. note:: It is recommended to use one of `assert_allclose`,
                  `assert_array_almost_equal_nulp` or `assert_array_max_ulp`
                  instead of this function for more consistent floating point
                  comparisons.
    
        The test verifies identical shapes and that the elements of ``actual`` and
        ``desired`` satisfy.
    
            ``abs(desired-actual) < 1.5 * 10**(-decimal)``
    
        That is a looser test than originally documented, but agrees with what the
        actual implementation did up to rounding vagaries. An exception is raised
        at shape mismatch or conflicting values. In contrast to the standard usage
        in numpy, NaNs are compared like numbers, no assertion is raised if both
        objects have NaNs in the same positions.
    
        Parameters
        ----------
        x : array_like
            The actual object to check.
        y : array_like
            The desired, expected object.
        decimal : int, optional
            Desired precision, default is 6.
        err_msg : str, optional
          The error message to be printed in case of failure.
        verbose : bool, optional
            If True, the conflicting values are appended to the error message.
    
        Raises
        ------
        AssertionError
            If actual and desired are not equal up to specified precision.
    
        See Also
        --------
        assert_allclose: Compare two array_like objects for equality with desired
                         relative and/or absolute precision.
        assert_array_almost_equal_nulp, assert_array_max_ulp, assert_equal
    
        Examples
        --------
        the first assert does not raise an exception
    
        >>> np.testing.assert_array_almost_equal([1.0,2.333,np.nan],
        ...                                      [1.0,2.333,np.nan])
    
        >>> np.testing.assert_array_almost_equal([1.0,2.33333,np.nan],
        ...                                      [1.0,2.33339,np.nan], decimal=5)
        Traceback (most recent call last):
            ...
        AssertionError:
        Arrays are not almost equal to 5 decimals
        <BLANKLINE>
        Mismatched elements: 1 / 3 (33.3%)
        Max absolute difference: 6.e-05
        Max relative difference: 2.57136612e-05
         x: array([1.     , 2.33333,     nan])
         y: array([1.     , 2.33339,     nan])
    
        >>> np.testing.assert_array_almost_equal([1.0,2.33333,np.nan],
        ...                                      [1.0,2.33333, 5], decimal=5)
        Traceback (most recent call last):
            ...
        AssertionError:
        Arrays are not almost equal to 5 decimals
        <BLANKLINE>
        x and y nan location mismatch:
         x: array([1.     , 2.33333,     nan])
         y: array([1.     , 2.33333, 5.     ])
    
        
    """
def assert_array_almost_equal_nulp(x, y, nulp = 1):
    """
    
        Compare two arrays relatively to their spacing.
    
        This is a relatively robust method to compare two arrays whose amplitude
        is variable.
    
        Parameters
        ----------
        x, y : array_like
            Input arrays.
        nulp : int, optional
            The maximum number of unit in the last place for tolerance (see Notes).
            Default is 1.
    
        Returns
        -------
        None
    
        Raises
        ------
        AssertionError
            If the spacing between `x` and `y` for one or more elements is larger
            than `nulp`.
    
        See Also
        --------
        assert_array_max_ulp : Check that all items of arrays differ in at most
            N Units in the Last Place.
        spacing : Return the distance between x and the nearest adjacent number.
    
        Notes
        -----
        An assertion is raised if the following condition is not met::
    
            abs(x - y) <= nulp * spacing(maximum(abs(x), abs(y)))
    
        Examples
        --------
        >>> x = np.array([1., 1e-10, 1e-20])
        >>> eps = np.finfo(x.dtype).eps
        >>> np.testing.assert_array_almost_equal_nulp(x, x*eps/2 + x)
    
        >>> np.testing.assert_array_almost_equal_nulp(x, x*eps + x)
        Traceback (most recent call last):
          ...
        AssertionError: X and Y are not equal to 1 ULP (max is 2)
    
        
    """
def assert_array_compare(*args, **kwds):
    ...
def assert_array_equal(x, y, err_msg = '', verbose = True, *, strict = False):
    """
    
        Raises an AssertionError if two array_like objects are not equal.
    
        Given two array_like objects, check that the shape is equal and all
        elements of these objects are equal (but see the Notes for the special
        handling of a scalar). An exception is raised at shape mismatch or
        conflicting values. In contrast to the standard usage in numpy, NaNs
        are compared like numbers, no assertion is raised if both objects have
        NaNs in the same positions.
    
        The usual caution for verifying equality with floating point numbers is
        advised.
    
        Parameters
        ----------
        x : array_like
            The actual object to check.
        y : array_like
            The desired, expected object.
        err_msg : str, optional
            The error message to be printed in case of failure.
        verbose : bool, optional
            If True, the conflicting values are appended to the error message.
        strict : bool, optional
            If True, raise an AssertionError when either the shape or the data
            type of the array_like objects does not match. The special
            handling for scalars mentioned in the Notes section is disabled.
    
            .. versionadded:: 1.24.0
    
        Raises
        ------
        AssertionError
            If actual and desired objects are not equal.
    
        See Also
        --------
        assert_allclose: Compare two array_like objects for equality with desired
                         relative and/or absolute precision.
        assert_array_almost_equal_nulp, assert_array_max_ulp, assert_equal
    
        Notes
        -----
        When one of `x` and `y` is a scalar and the other is array_like, the
        function checks that each element of the array_like object is equal to
        the scalar. This behaviour can be disabled with the `strict` parameter.
    
        Examples
        --------
        The first assert does not raise an exception:
    
        >>> np.testing.assert_array_equal([1.0,2.33333,np.nan],
        ...                               [np.exp(0),2.33333, np.nan])
    
        Assert fails with numerical imprecision with floats:
    
        >>> np.testing.assert_array_equal([1.0,np.pi,np.nan],
        ...                               [1, np.sqrt(np.pi)**2, np.nan])
        Traceback (most recent call last):
            ...
        AssertionError:
        Arrays are not equal
        <BLANKLINE>
        Mismatched elements: 1 / 3 (33.3%)
        Max absolute difference: 4.4408921e-16
        Max relative difference: 1.41357986e-16
         x: array([1.      , 3.141593,      nan])
         y: array([1.      , 3.141593,      nan])
    
        Use `assert_allclose` or one of the nulp (number of floating point values)
        functions for these cases instead:
    
        >>> np.testing.assert_allclose([1.0,np.pi,np.nan],
        ...                            [1, np.sqrt(np.pi)**2, np.nan],
        ...                            rtol=1e-10, atol=0)
    
        As mentioned in the Notes section, `assert_array_equal` has special
        handling for scalars. Here the test checks that each value in `x` is 3:
    
        >>> x = np.full((2, 5), fill_value=3)
        >>> np.testing.assert_array_equal(x, 3)
    
        Use `strict` to raise an AssertionError when comparing a scalar with an
        array:
    
        >>> np.testing.assert_array_equal(x, 3, strict=True)
        Traceback (most recent call last):
            ...
        AssertionError:
        Arrays are not equal
        <BLANKLINE>
        (shapes (2, 5), () mismatch)
         x: array([[3, 3, 3, 3, 3],
               [3, 3, 3, 3, 3]])
         y: array(3)
    
        The `strict` parameter also ensures that the array data types match:
    
        >>> x = np.array([2, 2, 2])
        >>> y = np.array([2., 2., 2.], dtype=np.float32)
        >>> np.testing.assert_array_equal(x, y, strict=True)
        Traceback (most recent call last):
            ...
        AssertionError:
        Arrays are not equal
        <BLANKLINE>
        (dtypes int64, float32 mismatch)
         x: array([2, 2, 2])
         y: array([2., 2., 2.], dtype=float32)
        
    """
def assert_array_less(x, y, err_msg = '', verbose = True):
    """
    
        Raises an AssertionError if two array_like objects are not ordered by less
        than.
    
        Given two array_like objects, check that the shape is equal and all
        elements of the first object are strictly smaller than those of the
        second object. An exception is raised at shape mismatch or incorrectly
        ordered values. Shape mismatch does not raise if an object has zero
        dimension. In contrast to the standard usage in numpy, NaNs are
        compared, no assertion is raised if both objects have NaNs in the same
        positions.
    
        Parameters
        ----------
        x : array_like
          The smaller object to check.
        y : array_like
          The larger object to compare.
        err_msg : string
          The error message to be printed in case of failure.
        verbose : bool
            If True, the conflicting values are appended to the error message.
    
        Raises
        ------
        AssertionError
          If x is not strictly smaller than y, element-wise.
    
        See Also
        --------
        assert_array_equal: tests objects for equality
        assert_array_almost_equal: test objects for equality up to precision
    
        Examples
        --------
        >>> np.testing.assert_array_less([1.0, 1.0, np.nan], [1.1, 2.0, np.nan])
        >>> np.testing.assert_array_less([1.0, 1.0, np.nan], [1, 2.0, np.nan])
        Traceback (most recent call last):
            ...
        AssertionError:
        Arrays are not less-ordered
        <BLANKLINE>
        Mismatched elements: 1 / 3 (33.3%)
        Max absolute difference: 1.
        Max relative difference: 0.5
         x: array([ 1.,  1., nan])
         y: array([ 1.,  2., nan])
    
        >>> np.testing.assert_array_less([1.0, 4.0], 3)
        Traceback (most recent call last):
            ...
        AssertionError:
        Arrays are not less-ordered
        <BLANKLINE>
        Mismatched elements: 1 / 2 (50%)
        Max absolute difference: 2.
        Max relative difference: 0.66666667
         x: array([1., 4.])
         y: array(3)
    
        >>> np.testing.assert_array_less([1.0, 2.0, 3.0], [4])
        Traceback (most recent call last):
            ...
        AssertionError:
        Arrays are not less-ordered
        <BLANKLINE>
        (shapes (3,), (1,) mismatch)
         x: array([1., 2., 3.])
         y: array([4])
    
        
    """
def assert_array_max_ulp(a, b, maxulp = 1, dtype = None):
    """
    
        Check that all items of arrays differ in at most N Units in the Last Place.
    
        Parameters
        ----------
        a, b : array_like
            Input arrays to be compared.
        maxulp : int, optional
            The maximum number of units in the last place that elements of `a` and
            `b` can differ. Default is 1.
        dtype : dtype, optional
            Data-type to convert `a` and `b` to if given. Default is None.
    
        Returns
        -------
        ret : ndarray
            Array containing number of representable floating point numbers between
            items in `a` and `b`.
    
        Raises
        ------
        AssertionError
            If one or more elements differ by more than `maxulp`.
    
        Notes
        -----
        For computing the ULP difference, this API does not differentiate between
        various representations of NAN (ULP difference between 0x7fc00000 and 0xffc00000
        is zero).
    
        See Also
        --------
        assert_array_almost_equal_nulp : Compare two arrays relatively to their
            spacing.
    
        Examples
        --------
        >>> a = np.linspace(0., 1., 100)
        >>> res = np.testing.assert_array_max_ulp(a, np.arcsin(np.sin(a)))
    
        
    """
def assert_equal(actual, desired, err_msg = '', verbose = True):
    """
    
        Raises an AssertionError if two objects are not equal.
    
        Given two objects (scalars, lists, tuples, dictionaries or numpy arrays),
        check that all elements of these objects are equal. An exception is raised
        at the first conflicting values.
    
        When one of `actual` and `desired` is a scalar and the other is array_like,
        the function checks that each element of the array_like object is equal to
        the scalar.
    
        This function handles NaN comparisons as if NaN was a "normal" number.
        That is, AssertionError is not raised if both objects have NaNs in the same
        positions.  This is in contrast to the IEEE standard on NaNs, which says
        that NaN compared to anything must return False.
    
        Parameters
        ----------
        actual : array_like
            The object to check.
        desired : array_like
            The expected object.
        err_msg : str, optional
            The error message to be printed in case of failure.
        verbose : bool, optional
            If True, the conflicting values are appended to the error message.
    
        Raises
        ------
        AssertionError
            If actual and desired are not equal.
    
        Examples
        --------
        >>> np.testing.assert_equal([4,5], [4,6])
        Traceback (most recent call last):
            ...
        AssertionError:
        Items are not equal:
        item=1
         ACTUAL: 5
         DESIRED: 6
    
        The following comparison does not raise an exception.  There are NaNs
        in the inputs, but they are in the same positions.
    
        >>> np.testing.assert_equal(np.array([1.0, 2.0, np.nan]), [1, 2, np.nan])
    
        
    """
def assert_no_gc_cycles(*args, **kwargs):
    """
    
        Fail if the given callable produces any reference cycles.
    
        If called with all arguments omitted, may be used as a context manager:
    
            with assert_no_gc_cycles():
                do_something()
    
        .. versionadded:: 1.15.0
    
        Parameters
        ----------
        func : callable
            The callable to test.
        \\*args : Arguments
            Arguments passed to `func`.
        \\*\\*kwargs : Kwargs
            Keyword arguments passed to `func`.
    
        Returns
        -------
        Nothing. The result is deliberately discarded to ensure that all cycles
        are found.
    
        
    """
def assert_no_warnings(*args, **kwargs):
    """
    
        Fail if the given callable produces any warnings.
    
        If called with all arguments omitted, may be used as a context manager:
    
            with assert_no_warnings():
                do_something()
    
        The ability to be used as a context manager is new in NumPy v1.11.0.
    
        .. versionadded:: 1.7.0
    
        Parameters
        ----------
        func : callable
            The callable to test.
        \\*args : Arguments
            Arguments passed to `func`.
        \\*\\*kwargs : Kwargs
            Keyword arguments passed to `func`.
    
        Returns
        -------
        The value returned by `func`.
    
        
    """
def assert_raises(*args, **kwargs):
    """
    
        assert_raises(exception_class, callable, *args, **kwargs)
        assert_raises(exception_class)
    
        Fail unless an exception of class exception_class is thrown
        by callable when invoked with arguments args and keyword
        arguments kwargs. If a different type of exception is
        thrown, it will not be caught, and the test case will be
        deemed to have suffered an error, exactly as for an
        unexpected exception.
    
        Alternatively, `assert_raises` can be used as a context manager:
    
        >>> from numpy.testing import assert_raises
        >>> with assert_raises(ZeroDivisionError):
        ...     1 / 0
    
        is equivalent to
    
        >>> def div(x, y):
        ...     return x / y
        >>> assert_raises(ZeroDivisionError, div, 1, 0)
    
        
    """
def assert_raises_regex(exception_class, expected_regexp, *args, **kwargs):
    """
    
        assert_raises_regex(exception_class, expected_regexp, callable, *args,
                            **kwargs)
        assert_raises_regex(exception_class, expected_regexp)
    
        Fail unless an exception of class exception_class and with message that
        matches expected_regexp is thrown by callable when invoked with arguments
        args and keyword arguments kwargs.
    
        Alternatively, can be used as a context manager like `assert_raises`.
    
        Notes
        -----
        .. versionadded:: 1.9.0
    
        
    """
def assert_string_equal(actual, desired):
    """
    
        Test if two strings are equal.
    
        If the given strings are equal, `assert_string_equal` does nothing.
        If they are not equal, an AssertionError is raised, and the diff
        between the strings is shown.
    
        Parameters
        ----------
        actual : str
            The string to test for equality against the expected string.
        desired : str
            The expected string.
    
        Examples
        --------
        >>> np.testing.assert_string_equal('abc', 'abc')
        >>> np.testing.assert_string_equal('abc', 'abcd')
        Traceback (most recent call last):
          File "<stdin>", line 1, in <module>
        ...
        AssertionError: Differences in strings:
        - abc+ abcd?    +
    
        
    """
def assert_warns(warning_class, *args, **kwargs):
    """
    
        Fail unless the given callable throws the specified warning.
    
        A warning of class warning_class should be thrown by the callable when
        invoked with arguments args and keyword arguments kwargs.
        If a different type of warning is thrown, it will not be caught.
    
        If called with all arguments other than the warning class omitted, may be
        used as a context manager:
    
            with assert_warns(SomeWarning):
                do_something()
    
        The ability to be used as a context manager is new in NumPy v1.11.0.
    
        .. versionadded:: 1.4.0
    
        Parameters
        ----------
        warning_class : class
            The class defining the warning that `func` is expected to throw.
        func : callable, optional
            Callable to test
        *args : Arguments
            Arguments for `func`.
        **kwargs : Kwargs
            Keyword arguments for `func`.
    
        Returns
        -------
        The value returned by `func`.
    
        Examples
        --------
        >>> import warnings
        >>> def deprecated_func(num):
        ...     warnings.warn("Please upgrade", DeprecationWarning)
        ...     return num*num
        >>> with np.testing.assert_warns(DeprecationWarning):
        ...     assert deprecated_func(4) == 16
        >>> # or passing a func
        >>> ret = np.testing.assert_warns(DeprecationWarning, deprecated_func, 4)
        >>> assert ret == 16
        
    """
def break_cycles():
    """
    
        Break reference cycles by calling gc.collect
        Objects can call other objects' methods (for instance, another object's
         __del__) inside their own __del__. On PyPy, the interpreter only runs
        between calls to gc.collect, so multiple calls are needed to completely
        release all cycles.
        
    """
def build_err_msg(arrays, err_msg, header = 'Items are not equal:', verbose = True, names = ('ACTUAL', 'DESIRED'), precision = 8):
    ...
def check_free_memory(free_bytes):
    """
    
        Check whether `free_bytes` amount of memory is currently free.
        Returns: None if enough memory available, otherwise error message
        
    """
def check_support_sve():
    """
    
        gh-22982
        
    """
def decorate_methods(cls, decorator, testmatch = None):
    """
    
        Apply a decorator to all methods in a class matching a regular expression.
    
        The given decorator is applied to all public methods of `cls` that are
        matched by the regular expression `testmatch`
        (``testmatch.search(methodname)``). Methods that are private, i.e. start
        with an underscore, are ignored.
    
        Parameters
        ----------
        cls : class
            Class whose methods to decorate.
        decorator : function
            Decorator to apply to methods
        testmatch : compiled regexp or str, optional
            The regular expression. Default value is None, in which case the
            nose default (``re.compile(r'(?:^|[\\b_\\.%s-])[Tt]est' % os.sep)``)
            is used.
            If `testmatch` is a string, it is compiled to a regular expression
            first.
    
        
    """
def integer_repr(x):
    """
    Return the signed-magnitude interpretation of the binary representation
        of x.
    """
def jiffies(_load_time = list()):
    """
    
            Return number of jiffies elapsed.
    
            Return number of jiffies (1/100ths of a second) that this
            process has been scheduled in user mode. See man 5 proc.
    
            
    """
def measure(code_str, times = 1, label = None):
    """
    
        Return elapsed time for executing code in the namespace of the caller.
    
        The supplied code string is compiled with the Python builtin ``compile``.
        The precision of the timing is 10 milli-seconds. If the code will execute
        fast on this timescale, it can be executed many times to get reasonable
        timing accuracy.
    
        Parameters
        ----------
        code_str : str
            The code to be timed.
        times : int, optional
            The number of times the code is executed. Default is 1. The code is
            only compiled once.
        label : str, optional
            A label to identify `code_str` with. This is passed into ``compile``
            as the second argument (for run-time error messages).
    
        Returns
        -------
        elapsed : float
            Total elapsed time in seconds for executing `code_str` `times` times.
    
        Examples
        --------
        >>> times = 10
        >>> etime = np.testing.measure('for i in range(1000): np.sqrt(i**2)', times=times)
        >>> print("Time for a single execution : ", etime / times, "s")  # doctest: +SKIP
        Time for a single execution :  0.005 s
    
        
    """
def memusage(processName = 'python', instance = 0):
    ...
def nulp_diff(x, y, dtype = None):
    """
    For each item in x and y, return the number of representable floating
        points between them.
    
        Parameters
        ----------
        x : array_like
            first input array
        y : array_like
            second input array
        dtype : dtype, optional
            Data-type to convert `x` and `y` to if given. Default is None.
    
        Returns
        -------
        nulp : array_like
            number of representable floating point numbers between each item in x
            and y.
    
        Notes
        -----
        For computing the ULP difference, this API does not differentiate between
        various representations of NAN (ULP difference between 0x7fc00000 and 0xffc00000
        is zero).
    
        Examples
        --------
        # By definition, epsilon is the smallest number such as 1 + eps != 1, so
        # there should be exactly one ULP between 1 and 1 + eps
        >>> nulp_diff(1, 1 + np.finfo(x.dtype).eps)
        1.0
        
    """
def print_assert_equal(test_string, actual, desired):
    """
    
        Test if two objects are equal, and print an error message if test fails.
    
        The test is performed with ``actual == desired``.
    
        Parameters
        ----------
        test_string : str
            The message supplied to AssertionError.
        actual : object
            The object to test for equality against `desired`.
        desired : object
            The expected result.
    
        Examples
        --------
        >>> np.testing.print_assert_equal('Test XYZ of func xyz', [0, 1], [0, 1])
        >>> np.testing.print_assert_equal('Test XYZ of func xyz', [0, 1], [0, 2])
        Traceback (most recent call last):
        ...
        AssertionError: Test XYZ of func xyz failed
        ACTUAL:
        [0, 1]
        DESIRED:
        [0, 2]
    
        
    """
def requires_memory(free_bytes):
    """
    Decorator to skip a test if not enough memory is available
    """
def rundocs(filename = None, raise_on_error = True):
    """
    
        Run doctests found in the given file.
    
        By default `rundocs` raises an AssertionError on failure.
    
        Parameters
        ----------
        filename : str
            The path to the file for which the doctests are run.
        raise_on_error : bool
            Whether to raise an AssertionError when a doctest fails. Default is
            True.
    
        Notes
        -----
        The doctests can be run by the user/developer by adding the ``doctests``
        argument to the ``test()`` call. For example, to run all tests (including
        doctests) for `numpy.lib`:
    
        >>> np.lib.test(doctests=True)  # doctest: +SKIP
        
    """
def runstring(astr, dict):
    ...
def tempdir(*args, **kwds):
    """
    Context manager to provide a temporary test folder.
    
        All arguments are passed as this to the underlying tempfile.mkdtemp
        function.
    
        
    """
def temppath(*args, **kwds):
    """
    Context manager for temporary files.
    
        Context manager that returns the path to a closed temporary file. Its
        parameters are the same as for tempfile.mkstemp and are passed directly
        to that function. The underlying file is removed when the context is
        exited, so it should be closed at that time.
    
        Windows does not allow a temporary file to be opened if it is already
        open, so the underlying file must be closed after opening before it
        can be opened again.
    
        
    """
HAS_LAPACK64: bool = True
HAS_REFCOUNT: bool = True
IS_MUSL: bool = False
IS_PYPY: bool = False
IS_PYSTON: bool = False
IS_WASM: bool = False
_SUPPORTS_SVE: bool = False
_d: _Dummy  # value = <numpy.testing._private.utils._Dummy testMethod=nop>
_glibcver: str = '0.0'
_v: str = ''
isfinite: numpy.ufunc  # value = <ufunc 'isfinite'>
isinf: numpy.ufunc  # value = <ufunc 'isinf'>
isnan: numpy.ufunc  # value = <ufunc 'isnan'>
isnat: numpy.ufunc  # value = <ufunc 'isnat'>
verbose: int = 0
KnownFailureTest = KnownFailureException
