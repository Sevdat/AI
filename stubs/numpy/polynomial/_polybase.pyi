"""

Abstract base class for the various polynomial Classes.

The ABCPolyBase class provides the methods needed to implement the common API
for the various polynomial classes. It operates as a mixin, but uses the
abc module from the stdlib, hence it is only available for Python >= 2.6.

"""
from __future__ import annotations
import abc as abc
import numbers as numbers
import numpy as np
from numpy.polynomial import polyutils as pu
import os as os
import typing
__all__: list = ['ABCPolyBase']
class ABCPolyBase(abc.ABC):
    """
    An abstract base class for immutable series classes.
    
        ABCPolyBase provides the standard Python numerical methods
        '+', '-', '*', '//', '%', 'divmod', '**', and '()' along with the
        methods listed below.
    
        .. versionadded:: 1.9.0
    
        Parameters
        ----------
        coef : array_like
            Series coefficients in order of increasing degree, i.e.,
            ``(1, 2, 3)`` gives ``1*P_0(x) + 2*P_1(x) + 3*P_2(x)``, where
            ``P_i`` is the basis polynomials of degree ``i``.
        domain : (2,) array_like, optional
            Domain to use. The interval ``[domain[0], domain[1]]`` is mapped
            to the interval ``[window[0], window[1]]`` by shifting and scaling.
            The default value is the derived class domain.
        window : (2,) array_like, optional
            Window, see domain for its use. The default value is the
            derived class window.
        symbol : str, optional
            Symbol used to represent the independent variable in string 
            representations of the polynomial expression, e.g. for printing.
            The symbol must be a valid Python identifier. Default value is 'x'.
    
            .. versionadded:: 1.24
    
        Attributes
        ----------
        coef : (N,) ndarray
            Series coefficients in order of increasing degree.
        domain : (2,) ndarray
            Domain that is mapped to window.
        window : (2,) ndarray
            Window that domain is mapped to.
        symbol : str
            Symbol representing the independent variable.
    
        Class Attributes
        ----------------
        maxpower : int
            Maximum power allowed, i.e., the largest number ``n`` such that
            ``p(x)**n`` is allowed. This is to limit runaway polynomial size.
        domain : (2,) ndarray
            Default domain of the class.
        window : (2,) ndarray
            Default window of the class.
    
        
    """
    __abstractmethods__: typing.ClassVar[frozenset]  # value = frozenset({'domain', '_der', '_line', '_val', '_fromroots', '_mul', '_pow', 'basis_name', '_int', 'window', '_roots', '_add', '_div', '_fit', '_sub'})
    __array_ufunc__ = None
    __hash__: typing.ClassVar[None] = None
    _abc_impl: typing.ClassVar[_abc._abc_data]  # value = <_abc._abc_data object>
    _subscript_mapping: typing.ClassVar[dict] = {48: '₀', 49: '₁', 50: '₂', 51: '₃', 52: '₄', 53: '₅', 54: '₆', 55: '₇', 56: '₈', 57: '₉'}
    _superscript_mapping: typing.ClassVar[dict] = {48: '⁰', 49: '¹', 50: '²', 51: '³', 52: '⁴', 53: '⁵', 54: '⁶', 55: '⁷', 56: '⁸', 57: '⁹'}
    _use_unicode: typing.ClassVar[bool] = False
    maxpower: typing.ClassVar[int] = 100
    @staticmethod
    def _add(c1, c2):
        ...
    @staticmethod
    def _der(c, m, scl):
        ...
    @staticmethod
    def _div(c1, c2):
        ...
    @staticmethod
    def _fit(x, y, deg, rcond, full):
        ...
    @staticmethod
    def _fromroots(r):
        ...
    @staticmethod
    def _int(c, m, k, lbnd, scl):
        ...
    @staticmethod
    def _line(off, scl):
        ...
    @staticmethod
    def _mul(c1, c2):
        ...
    @staticmethod
    def _pow(c, pow, maxpower = None):
        ...
    @staticmethod
    def _repr_latex_scalar(x, parens = False):
        ...
    @staticmethod
    def _roots(c):
        ...
    @staticmethod
    def _sub(c1, c2):
        ...
    @staticmethod
    def _val(x, c):
        ...
    @classmethod
    def _repr_latex_term(cls, i, arg_str, needs_parens):
        ...
    @classmethod
    def _str_term_ascii(cls, i, arg_str):
        """
        
                String representation of a single polynomial term using ** and _ to
                represent superscripts and subscripts, respectively.
                
        """
    @classmethod
    def _str_term_unicode(cls, i, arg_str):
        """
        
                String representation of single polynomial term using unicode
                characters for superscripts and subscripts.
                
        """
    @classmethod
    def basis(cls, deg, domain = None, window = None, symbol = 'x'):
        """
        Series basis polynomial of degree `deg`.
        
                Returns the series representing the basis polynomial of degree `deg`.
        
                .. versionadded:: 1.7.0
        
                Parameters
                ----------
                deg : int
                    Degree of the basis polynomial for the series. Must be >= 0.
                domain : {None, array_like}, optional
                    If given, the array must be of the form ``[beg, end]``, where
                    ``beg`` and ``end`` are the endpoints of the domain. If None is
                    given then the class domain is used. The default is None.
                window : {None, array_like}, optional
                    If given, the resulting array must be if the form
                    ``[beg, end]``, where ``beg`` and ``end`` are the endpoints of
                    the window. If None is given then the class window is used. The
                    default is None.
                symbol : str, optional
                    Symbol representing the independent variable. Default is 'x'.
        
                Returns
                -------
                new_series : series
                    A series with the coefficient of the `deg` term set to one and
                    all others zero.
        
                
        """
    @classmethod
    def cast(cls, series, domain = None, window = None):
        """
        Convert series to series of this class.
        
                The `series` is expected to be an instance of some polynomial
                series of one of the types supported by by the numpy.polynomial
                module, but could be some other class that supports the convert
                method.
        
                .. versionadded:: 1.7.0
        
                Parameters
                ----------
                series : series
                    The series instance to be converted.
                domain : {None, array_like}, optional
                    If given, the array must be of the form ``[beg, end]``, where
                    ``beg`` and ``end`` are the endpoints of the domain. If None is
                    given then the class domain is used. The default is None.
                window : {None, array_like}, optional
                    If given, the resulting array must be if the form
                    ``[beg, end]``, where ``beg`` and ``end`` are the endpoints of
                    the window. If None is given then the class window is used. The
                    default is None.
        
                Returns
                -------
                new_series : series
                    A series of the same kind as the calling class and equal to
                    `series` when evaluated.
        
                See Also
                --------
                convert : similar instance method
        
                
        """
    @classmethod
    def fit(cls, x, y, deg, domain = None, rcond = None, full = False, w = None, window = None, symbol = 'x'):
        """
        Least squares fit to data.
        
                Return a series instance that is the least squares fit to the data
                `y` sampled at `x`. The domain of the returned instance can be
                specified and this will often result in a superior fit with less
                chance of ill conditioning.
        
                Parameters
                ----------
                x : array_like, shape (M,)
                    x-coordinates of the M sample points ``(x[i], y[i])``.
                y : array_like, shape (M,)
                    y-coordinates of the M sample points ``(x[i], y[i])``.
                deg : int or 1-D array_like
                    Degree(s) of the fitting polynomials. If `deg` is a single integer
                    all terms up to and including the `deg`'th term are included in the
                    fit. For NumPy versions >= 1.11.0 a list of integers specifying the
                    degrees of the terms to include may be used instead.
                domain : {None, [beg, end], []}, optional
                    Domain to use for the returned series. If ``None``,
                    then a minimal domain that covers the points `x` is chosen.  If
                    ``[]`` the class domain is used. The default value was the
                    class domain in NumPy 1.4 and ``None`` in later versions.
                    The ``[]`` option was added in numpy 1.5.0.
                rcond : float, optional
                    Relative condition number of the fit. Singular values smaller
                    than this relative to the largest singular value will be
                    ignored. The default value is len(x)*eps, where eps is the
                    relative precision of the float type, about 2e-16 in most
                    cases.
                full : bool, optional
                    Switch determining nature of return value. When it is False
                    (the default) just the coefficients are returned, when True
                    diagnostic information from the singular value decomposition is
                    also returned.
                w : array_like, shape (M,), optional
                    Weights. If not None, the weight ``w[i]`` applies to the unsquared
                    residual ``y[i] - y_hat[i]`` at ``x[i]``. Ideally the weights are
                    chosen so that the errors of the products ``w[i]*y[i]`` all have
                    the same variance.  When using inverse-variance weighting, use
                    ``w[i] = 1/sigma(y[i])``.  The default value is None.
        
                    .. versionadded:: 1.5.0
                window : {[beg, end]}, optional
                    Window to use for the returned series. The default
                    value is the default class domain
        
                    .. versionadded:: 1.6.0
                symbol : str, optional
                    Symbol representing the independent variable. Default is 'x'.
        
                Returns
                -------
                new_series : series
                    A series that represents the least squares fit to the data and
                    has the domain and window specified in the call. If the
                    coefficients for the unscaled and unshifted basis polynomials are
                    of interest, do ``new_series.convert().coef``.
        
                [resid, rank, sv, rcond] : list
                    These values are only returned if ``full == True``
        
                    - resid -- sum of squared residuals of the least squares fit
                    - rank -- the numerical rank of the scaled Vandermonde matrix
                    - sv -- singular values of the scaled Vandermonde matrix
                    - rcond -- value of `rcond`.
        
                    For more details, see `linalg.lstsq`.
        
                
        """
    @classmethod
    def fromroots(cls, roots, domain = list(), window = None, symbol = 'x'):
        """
        Return series instance that has the specified roots.
        
                Returns a series representing the product
                ``(x - r[0])*(x - r[1])*...*(x - r[n-1])``, where ``r`` is a
                list of roots.
        
                Parameters
                ----------
                roots : array_like
                    List of roots.
                domain : {[], None, array_like}, optional
                    Domain for the resulting series. If None the domain is the
                    interval from the smallest root to the largest. If [] the
                    domain is the class domain. The default is [].
                window : {None, array_like}, optional
                    Window for the returned series. If None the class window is
                    used. The default is None.
                symbol : str, optional
                    Symbol representing the independent variable. Default is 'x'.
        
                Returns
                -------
                new_series : series
                    Series with the specified roots.
        
                
        """
    @classmethod
    def identity(cls, domain = None, window = None, symbol = 'x'):
        """
        Identity function.
        
                If ``p`` is the returned series, then ``p(x) == x`` for all
                values of x.
        
                Parameters
                ----------
                domain : {None, array_like}, optional
                    If given, the array must be of the form ``[beg, end]``, where
                    ``beg`` and ``end`` are the endpoints of the domain. If None is
                    given then the class domain is used. The default is None.
                window : {None, array_like}, optional
                    If given, the resulting array must be if the form
                    ``[beg, end]``, where ``beg`` and ``end`` are the endpoints of
                    the window. If None is given then the class window is used. The
                    default is None.
                symbol : str, optional
                    Symbol representing the independent variable. Default is 'x'.
        
                Returns
                -------
                new_series : series
                     Series of representing the identity.
        
                
        """
    def __add__(self, other):
        ...
    def __call__(self, arg):
        ...
    def __divmod__(self, other):
        ...
    def __eq__(self, other):
        ...
    def __floordiv__(self, other):
        ...
    def __format__(self, fmt_str):
        ...
    def __getstate__(self):
        ...
    def __init__(self, coef, domain = None, window = None, symbol = 'x'):
        ...
    def __iter__(self):
        ...
    def __len__(self):
        ...
    def __mod__(self, other):
        ...
    def __mul__(self, other):
        ...
    def __ne__(self, other):
        ...
    def __neg__(self):
        ...
    def __pos__(self):
        ...
    def __pow__(self, other):
        ...
    def __radd__(self, other):
        ...
    def __rdiv__(self, other):
        ...
    def __rdivmod__(self, other):
        ...
    def __repr__(self):
        ...
    def __rfloordiv__(self, other):
        ...
    def __rmod__(self, other):
        ...
    def __rmul__(self, other):
        ...
    def __rsub__(self, other):
        ...
    def __rtruediv__(self, other):
        ...
    def __setstate__(self, dict):
        ...
    def __str__(self):
        ...
    def __sub__(self, other):
        ...
    def __truediv__(self, other):
        ...
    def _generate_string(self, term_method):
        """
        
                Generate the full string representation of the polynomial, using
                ``term_method`` to generate each polynomial term.
                
        """
    def _get_coefficients(self, other):
        """
        Interpret other as polynomial coefficients.
        
                The `other` argument is checked to see if it is of the same
                class as self with identical domain and window. If so,
                return its coefficients, otherwise return `other`.
        
                .. versionadded:: 1.9.0
        
                Parameters
                ----------
                other : anything
                    Object to be checked.
        
                Returns
                -------
                coef
                    The coefficients of`other` if it is a compatible instance,
                    of ABCPolyBase, otherwise `other`.
        
                Raises
                ------
                TypeError
                    When `other` is an incompatible instance of ABCPolyBase.
        
                
        """
    def _repr_latex_(self):
        ...
    def convert(self, domain = None, kind = None, window = None):
        """
        Convert series to a different kind and/or domain and/or window.
        
                Parameters
                ----------
                domain : array_like, optional
                    The domain of the converted series. If the value is None,
                    the default domain of `kind` is used.
                kind : class, optional
                    The polynomial series type class to which the current instance
                    should be converted. If kind is None, then the class of the
                    current instance is used.
                window : array_like, optional
                    The window of the converted series. If the value is None,
                    the default window of `kind` is used.
        
                Returns
                -------
                new_series : series
                    The returned class can be of different type than the current
                    instance and/or have a different domain and/or different
                    window.
        
                Notes
                -----
                Conversion between domains and class types can result in
                numerically ill defined series.
        
                
        """
    def copy(self):
        """
        Return a copy.
        
                Returns
                -------
                new_series : series
                    Copy of self.
        
                
        """
    def cutdeg(self, deg):
        """
        Truncate series to the given degree.
        
                Reduce the degree of the series to `deg` by discarding the
                high order terms. If `deg` is greater than the current degree a
                copy of the current series is returned. This can be useful in least
                squares where the coefficients of the high degree terms may be very
                small.
        
                .. versionadded:: 1.5.0
        
                Parameters
                ----------
                deg : non-negative int
                    The series is reduced to degree `deg` by discarding the high
                    order terms. The value of `deg` must be a non-negative integer.
        
                Returns
                -------
                new_series : series
                    New instance of series with reduced degree.
        
                
        """
    def degree(self):
        """
        The degree of the series.
        
                .. versionadded:: 1.5.0
        
                Returns
                -------
                degree : int
                    Degree of the series, one less than the number of coefficients.
        
                Examples
                --------
        
                Create a polynomial object for ``1 + 7*x + 4*x**2``:
        
                >>> poly = np.polynomial.Polynomial([1, 7, 4])
                >>> print(poly)
                1.0 + 7.0·x + 4.0·x²
                >>> poly.degree()
                2
        
                Note that this method does not check for non-zero coefficients.
                You must trim the polynomial to remove any trailing zeroes:
        
                >>> poly = np.polynomial.Polynomial([1, 7, 0])
                >>> print(poly)
                1.0 + 7.0·x + 0.0·x²
                >>> poly.degree()
                2
                >>> poly.trim().degree()
                1
        
                
        """
    def deriv(self, m = 1):
        """
        Differentiate.
        
                Return a series instance of that is the derivative of the current
                series.
        
                Parameters
                ----------
                m : non-negative int
                    Find the derivative of order `m`.
        
                Returns
                -------
                new_series : series
                    A new series representing the derivative. The domain is the same
                    as the domain of the differentiated series.
        
                
        """
    def has_samecoef(self, other):
        """
        Check if coefficients match.
        
                .. versionadded:: 1.6.0
        
                Parameters
                ----------
                other : class instance
                    The other class must have the ``coef`` attribute.
        
                Returns
                -------
                bool : boolean
                    True if the coefficients are the same, False otherwise.
        
                
        """
    def has_samedomain(self, other):
        """
        Check if domains match.
        
                .. versionadded:: 1.6.0
        
                Parameters
                ----------
                other : class instance
                    The other class must have the ``domain`` attribute.
        
                Returns
                -------
                bool : boolean
                    True if the domains are the same, False otherwise.
        
                
        """
    def has_sametype(self, other):
        """
        Check if types match.
        
                .. versionadded:: 1.7.0
        
                Parameters
                ----------
                other : object
                    Class instance.
        
                Returns
                -------
                bool : boolean
                    True if other is same class as self
        
                
        """
    def has_samewindow(self, other):
        """
        Check if windows match.
        
                .. versionadded:: 1.6.0
        
                Parameters
                ----------
                other : class instance
                    The other class must have the ``window`` attribute.
        
                Returns
                -------
                bool : boolean
                    True if the windows are the same, False otherwise.
        
                
        """
    def integ(self, m = 1, k = list(), lbnd = None):
        """
        Integrate.
        
                Return a series instance that is the definite integral of the
                current series.
        
                Parameters
                ----------
                m : non-negative int
                    The number of integrations to perform.
                k : array_like
                    Integration constants. The first constant is applied to the
                    first integration, the second to the second, and so on. The
                    list of values must less than or equal to `m` in length and any
                    missing values are set to zero.
                lbnd : Scalar
                    The lower bound of the definite integral.
        
                Returns
                -------
                new_series : series
                    A new series representing the integral. The domain is the same
                    as the domain of the integrated series.
        
                
        """
    def linspace(self, n = 100, domain = None):
        """
        Return x, y values at equally spaced points in domain.
        
                Returns the x, y values at `n` linearly spaced points across the
                domain.  Here y is the value of the polynomial at the points x. By
                default the domain is the same as that of the series instance.
                This method is intended mostly as a plotting aid.
        
                .. versionadded:: 1.5.0
        
                Parameters
                ----------
                n : int, optional
                    Number of point pairs to return. The default value is 100.
                domain : {None, array_like}, optional
                    If not None, the specified domain is used instead of that of
                    the calling instance. It should be of the form ``[beg,end]``.
                    The default is None which case the class domain is used.
        
                Returns
                -------
                x, y : ndarray
                    x is equal to linspace(self.domain[0], self.domain[1], n) and
                    y is the series evaluated at element of x.
        
                
        """
    def mapparms(self):
        """
        Return the mapping parameters.
        
                The returned values define a linear map ``off + scl*x`` that is
                applied to the input arguments before the series is evaluated. The
                map depends on the ``domain`` and ``window``; if the current
                ``domain`` is equal to the ``window`` the resulting map is the
                identity.  If the coefficients of the series instance are to be
                used by themselves outside this class, then the linear function
                must be substituted for the ``x`` in the standard representation of
                the base polynomials.
        
                Returns
                -------
                off, scl : float or complex
                    The mapping function is defined by ``off + scl*x``.
        
                Notes
                -----
                If the current domain is the interval ``[l1, r1]`` and the window
                is ``[l2, r2]``, then the linear mapping function ``L`` is
                defined by the equations::
        
                    L(l1) = l2
                    L(r1) = r2
        
                
        """
    def roots(self):
        """
        Return the roots of the series polynomial.
        
                Compute the roots for the series. Note that the accuracy of the
                roots decreases the further outside the `domain` they lie.
        
                Returns
                -------
                roots : ndarray
                    Array containing the roots of the series.
        
                
        """
    def trim(self, tol = 0):
        """
        Remove trailing coefficients
        
                Remove trailing coefficients until a coefficient is reached whose
                absolute value greater than `tol` or the beginning of the series is
                reached. If all the coefficients would be removed the series is set
                to ``[0]``. A new series instance is returned with the new
                coefficients.  The current instance remains unchanged.
        
                Parameters
                ----------
                tol : non-negative number.
                    All trailing coefficients less than `tol` will be removed.
        
                Returns
                -------
                new_series : series
                    New instance of series with trimmed coefficients.
        
                
        """
    def truncate(self, size):
        """
        Truncate series to length `size`.
        
                Reduce the series to length `size` by discarding the high
                degree terms. The value of `size` must be a positive integer. This
                can be useful in least squares where the coefficients of the
                high degree terms may be very small.
        
                Parameters
                ----------
                size : positive int
                    The series is reduced to length `size` by discarding the high
                    degree terms. The value of `size` must be a positive integer.
        
                Returns
                -------
                new_series : series
                    New instance of series with truncated coefficients.
        
                
        """
    @property
    def basis_name(self):
        ...
    @property
    def domain(self):
        ...
    @property
    def symbol(self):
        ...
    @property
    def window(self):
        ...
