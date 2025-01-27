"""

==================================================
Laguerre Series (:mod:`numpy.polynomial.laguerre`)
==================================================

This module provides a number of objects (mostly functions) useful for
dealing with Laguerre series, including a `Laguerre` class that
encapsulates the usual arithmetic operations.  (General information
on how this module represents and works with such polynomials is in the
docstring for its "parent" sub-package, `numpy.polynomial`).

Classes
-------
.. autosummary::
   :toctree: generated/

   Laguerre

Constants
---------
.. autosummary::
   :toctree: generated/

   lagdomain
   lagzero
   lagone
   lagx

Arithmetic
----------
.. autosummary::
   :toctree: generated/

   lagadd
   lagsub
   lagmulx
   lagmul
   lagdiv
   lagpow
   lagval
   lagval2d
   lagval3d
   laggrid2d
   laggrid3d

Calculus
--------
.. autosummary::
   :toctree: generated/

   lagder
   lagint

Misc Functions
--------------
.. autosummary::
   :toctree: generated/

   lagfromroots
   lagroots
   lagvander
   lagvander2d
   lagvander3d
   laggauss
   lagweight
   lagcompanion
   lagfit
   lagtrim
   lagline
   lag2poly
   poly2lag

See also
--------
`numpy.polynomial`

"""
from __future__ import annotations
import numpy
import numpy as np
from numpy.core._multiarray_umath import normalize_axis_index
from numpy import linalg as la
import numpy.polynomial._polybase
from numpy.polynomial._polybase import ABCPolyBase
from numpy.polynomial import polyutils as pu
from numpy.polynomial.polyutils import trimcoef as lagtrim
import typing
__all__: list = ['lagzero', 'lagone', 'lagx', 'lagdomain', 'lagline', 'lagadd', 'lagsub', 'lagmulx', 'lagmul', 'lagdiv', 'lagpow', 'lagval', 'lagder', 'lagint', 'lag2poly', 'poly2lag', 'lagfromroots', 'lagvander', 'lagfit', 'lagtrim', 'lagroots', 'Laguerre', 'lagval2d', 'lagval3d', 'laggrid2d', 'laggrid3d', 'lagvander2d', 'lagvander3d', 'lagcompanion', 'laggauss', 'lagweight']
class Laguerre(numpy.polynomial._polybase.ABCPolyBase):
    """
    A Laguerre series class.
    
        The Laguerre class provides the standard Python numerical methods
        '+', '-', '*', '//', '%', 'divmod', '**', and '()' as well as the
        attributes and methods listed in the `ABCPolyBase` documentation.
    
        Parameters
        ----------
        coef : array_like
            Laguerre coefficients in order of increasing degree, i.e,
            ``(1, 2, 3)`` gives ``1*L_0(x) + 2*L_1(X) + 3*L_2(x)``.
        domain : (2,) array_like, optional
            Domain to use. The interval ``[domain[0], domain[1]]`` is mapped
            to the interval ``[window[0], window[1]]`` by shifting and scaling.
            The default value is [0, 1].
        window : (2,) array_like, optional
            Window, see `domain` for its use. The default value is [0, 1].
    
            .. versionadded:: 1.6.0
        symbol : str, optional
            Symbol used to represent the independent variable in string
            representations of the polynomial expression, e.g. for printing.
            The symbol must be a valid Python identifier. Default value is 'x'.
    
            .. versionadded:: 1.24
    
        
    """
    __abstractmethods__: typing.ClassVar[frozenset]  # value = frozenset()
    _abc_impl: typing.ClassVar[_abc._abc_data]  # value = <_abc._abc_data object>
    basis_name: typing.ClassVar[str] = 'L'
    domain: typing.ClassVar[numpy.ndarray]  # value = array([0, 1])
    window: typing.ClassVar[numpy.ndarray]  # value = array([0, 1])
    @staticmethod
    def _add(c1, c2):
        """
        
            Add one Laguerre series to another.
        
            Returns the sum of two Laguerre series `c1` + `c2`.  The arguments
            are sequences of coefficients ordered from lowest order term to
            highest, i.e., [1,2,3] represents the series ``P_0 + 2*P_1 + 3*P_2``.
        
            Parameters
            ----------
            c1, c2 : array_like
                1-D arrays of Laguerre series coefficients ordered from low to
                high.
        
            Returns
            -------
            out : ndarray
                Array representing the Laguerre series of their sum.
        
            See Also
            --------
            lagsub, lagmulx, lagmul, lagdiv, lagpow
        
            Notes
            -----
            Unlike multiplication, division, etc., the sum of two Laguerre series
            is a Laguerre series (without having to "reproject" the result onto
            the basis set) so addition, just like that of "standard" polynomials,
            is simply "component-wise."
        
            Examples
            --------
            >>> from numpy.polynomial.laguerre import lagadd
            >>> lagadd([1, 2, 3], [1, 2, 3, 4])
            array([2.,  4.,  6.,  4.])
        
        
            
        """
    @staticmethod
    def _der(c, m = 1, scl = 1, axis = 0):
        """
        
            Differentiate a Laguerre series.
        
            Returns the Laguerre series coefficients `c` differentiated `m` times
            along `axis`.  At each iteration the result is multiplied by `scl` (the
            scaling factor is for use in a linear change of variable). The argument
            `c` is an array of coefficients from low to high degree along each
            axis, e.g., [1,2,3] represents the series ``1*L_0 + 2*L_1 + 3*L_2``
            while [[1,2],[1,2]] represents ``1*L_0(x)*L_0(y) + 1*L_1(x)*L_0(y) +
            2*L_0(x)*L_1(y) + 2*L_1(x)*L_1(y)`` if axis=0 is ``x`` and axis=1 is
            ``y``.
        
            Parameters
            ----------
            c : array_like
                Array of Laguerre series coefficients. If `c` is multidimensional
                the different axis correspond to different variables with the
                degree in each axis given by the corresponding index.
            m : int, optional
                Number of derivatives taken, must be non-negative. (Default: 1)
            scl : scalar, optional
                Each differentiation is multiplied by `scl`.  The end result is
                multiplication by ``scl**m``.  This is for use in a linear change of
                variable. (Default: 1)
            axis : int, optional
                Axis over which the derivative is taken. (Default: 0).
        
                .. versionadded:: 1.7.0
        
            Returns
            -------
            der : ndarray
                Laguerre series of the derivative.
        
            See Also
            --------
            lagint
        
            Notes
            -----
            In general, the result of differentiating a Laguerre series does not
            resemble the same operation on a power series. Thus the result of this
            function may be "unintuitive," albeit correct; see Examples section
            below.
        
            Examples
            --------
            >>> from numpy.polynomial.laguerre import lagder
            >>> lagder([ 1.,  1.,  1., -3.])
            array([1.,  2.,  3.])
            >>> lagder([ 1.,  0.,  0., -4.,  3.], m=2)
            array([1.,  2.,  3.])
        
            
        """
    @staticmethod
    def _div(c1, c2):
        """
        
            Divide one Laguerre series by another.
        
            Returns the quotient-with-remainder of two Laguerre series
            `c1` / `c2`.  The arguments are sequences of coefficients from lowest
            order "term" to highest, e.g., [1,2,3] represents the series
            ``P_0 + 2*P_1 + 3*P_2``.
        
            Parameters
            ----------
            c1, c2 : array_like
                1-D arrays of Laguerre series coefficients ordered from low to
                high.
        
            Returns
            -------
            [quo, rem] : ndarrays
                Of Laguerre series coefficients representing the quotient and
                remainder.
        
            See Also
            --------
            lagadd, lagsub, lagmulx, lagmul, lagpow
        
            Notes
            -----
            In general, the (polynomial) division of one Laguerre series by another
            results in quotient and remainder terms that are not in the Laguerre
            polynomial basis set.  Thus, to express these results as a Laguerre
            series, it is necessary to "reproject" the results onto the Laguerre
            basis set, which may produce "unintuitive" (but correct) results; see
            Examples section below.
        
            Examples
            --------
            >>> from numpy.polynomial.laguerre import lagdiv
            >>> lagdiv([  8., -13.,  38., -51.,  36.], [0, 1, 2])
            (array([1., 2., 3.]), array([0.]))
            >>> lagdiv([  9., -12.,  38., -51.,  36.], [0, 1, 2])
            (array([1., 2., 3.]), array([1., 1.]))
        
            
        """
    @staticmethod
    def _fit(x, y, deg, rcond = None, full = False, w = None):
        """
        
            Least squares fit of Laguerre series to data.
        
            Return the coefficients of a Laguerre series of degree `deg` that is the
            least squares fit to the data values `y` given at points `x`. If `y` is
            1-D the returned coefficients will also be 1-D. If `y` is 2-D multiple
            fits are done, one for each column of `y`, and the resulting
            coefficients are stored in the corresponding columns of a 2-D return.
            The fitted polynomial(s) are in the form
        
            .. math::  p(x) = c_0 + c_1 * L_1(x) + ... + c_n * L_n(x),
        
            where ``n`` is `deg`.
        
            Parameters
            ----------
            x : array_like, shape (M,)
                x-coordinates of the M sample points ``(x[i], y[i])``.
            y : array_like, shape (M,) or (M, K)
                y-coordinates of the sample points. Several data sets of sample
                points sharing the same x-coordinates can be fitted at once by
                passing in a 2D-array that contains one dataset per column.
            deg : int or 1-D array_like
                Degree(s) of the fitting polynomials. If `deg` is a single integer
                all terms up to and including the `deg`'th term are included in the
                fit. For NumPy versions >= 1.11.0 a list of integers specifying the
                degrees of the terms to include may be used instead.
            rcond : float, optional
                Relative condition number of the fit. Singular values smaller than
                this relative to the largest singular value will be ignored. The
                default value is len(x)*eps, where eps is the relative precision of
                the float type, about 2e-16 in most cases.
            full : bool, optional
                Switch determining nature of return value. When it is False (the
                default) just the coefficients are returned, when True diagnostic
                information from the singular value decomposition is also returned.
            w : array_like, shape (`M`,), optional
                Weights. If not None, the weight ``w[i]`` applies to the unsquared
                residual ``y[i] - y_hat[i]`` at ``x[i]``. Ideally the weights are
                chosen so that the errors of the products ``w[i]*y[i]`` all have the
                same variance.  When using inverse-variance weighting, use
                ``w[i] = 1/sigma(y[i])``.  The default value is None.
        
            Returns
            -------
            coef : ndarray, shape (M,) or (M, K)
                Laguerre coefficients ordered from low to high. If `y` was 2-D,
                the coefficients for the data in column *k*  of `y` are in column
                *k*.
        
            [residuals, rank, singular_values, rcond] : list
                These values are only returned if ``full == True``
        
                - residuals -- sum of squared residuals of the least squares fit
                - rank -- the numerical rank of the scaled Vandermonde matrix
                - singular_values -- singular values of the scaled Vandermonde matrix
                - rcond -- value of `rcond`.
        
                For more details, see `numpy.linalg.lstsq`.
        
            Warns
            -----
            RankWarning
                The rank of the coefficient matrix in the least-squares fit is
                deficient. The warning is only raised if ``full == False``.  The
                warnings can be turned off by
        
                >>> import warnings
                >>> warnings.simplefilter('ignore', np.RankWarning)
        
            See Also
            --------
            numpy.polynomial.polynomial.polyfit
            numpy.polynomial.legendre.legfit
            numpy.polynomial.chebyshev.chebfit
            numpy.polynomial.hermite.hermfit
            numpy.polynomial.hermite_e.hermefit
            lagval : Evaluates a Laguerre series.
            lagvander : pseudo Vandermonde matrix of Laguerre series.
            lagweight : Laguerre weight function.
            numpy.linalg.lstsq : Computes a least-squares fit from the matrix.
            scipy.interpolate.UnivariateSpline : Computes spline fits.
        
            Notes
            -----
            The solution is the coefficients of the Laguerre series ``p`` that
            minimizes the sum of the weighted squared errors
        
            .. math:: E = \\sum_j w_j^2 * |y_j - p(x_j)|^2,
        
            where the :math:`w_j` are the weights. This problem is solved by
            setting up as the (typically) overdetermined matrix equation
        
            .. math:: V(x) * c = w * y,
        
            where ``V`` is the weighted pseudo Vandermonde matrix of `x`, ``c`` are the
            coefficients to be solved for, `w` are the weights, and `y` are the
            observed values.  This equation is then solved using the singular value
            decomposition of ``V``.
        
            If some of the singular values of `V` are so small that they are
            neglected, then a `RankWarning` will be issued. This means that the
            coefficient values may be poorly determined. Using a lower order fit
            will usually get rid of the warning.  The `rcond` parameter can also be
            set to a value smaller than its default, but the resulting fit may be
            spurious and have large contributions from roundoff error.
        
            Fits using Laguerre series are probably most useful when the data can
            be approximated by ``sqrt(w(x)) * p(x)``, where ``w(x)`` is the Laguerre
            weight. In that case the weight ``sqrt(w(x[i]))`` should be used
            together with data values ``y[i]/sqrt(w(x[i]))``. The weight function is
            available as `lagweight`.
        
            References
            ----------
            .. [1] Wikipedia, "Curve fitting",
                   https://en.wikipedia.org/wiki/Curve_fitting
        
            Examples
            --------
            >>> from numpy.polynomial.laguerre import lagfit, lagval
            >>> x = np.linspace(0, 10)
            >>> err = np.random.randn(len(x))/10
            >>> y = lagval(x, [1, 2, 3]) + err
            >>> lagfit(x, y, 2)
            array([ 0.96971004,  2.00193749,  3.00288744]) # may vary
        
            
        """
    @staticmethod
    def _fromroots(roots):
        """
        
            Generate a Laguerre series with given roots.
        
            The function returns the coefficients of the polynomial
        
            .. math:: p(x) = (x - r_0) * (x - r_1) * ... * (x - r_n),
        
            in Laguerre form, where the `r_n` are the roots specified in `roots`.
            If a zero has multiplicity n, then it must appear in `roots` n times.
            For instance, if 2 is a root of multiplicity three and 3 is a root of
            multiplicity 2, then `roots` looks something like [2, 2, 2, 3, 3]. The
            roots can appear in any order.
        
            If the returned coefficients are `c`, then
        
            .. math:: p(x) = c_0 + c_1 * L_1(x) + ... +  c_n * L_n(x)
        
            The coefficient of the last term is not generally 1 for monic
            polynomials in Laguerre form.
        
            Parameters
            ----------
            roots : array_like
                Sequence containing the roots.
        
            Returns
            -------
            out : ndarray
                1-D array of coefficients.  If all roots are real then `out` is a
                real array, if some of the roots are complex, then `out` is complex
                even if all the coefficients in the result are real (see Examples
                below).
        
            See Also
            --------
            numpy.polynomial.polynomial.polyfromroots
            numpy.polynomial.legendre.legfromroots
            numpy.polynomial.chebyshev.chebfromroots
            numpy.polynomial.hermite.hermfromroots
            numpy.polynomial.hermite_e.hermefromroots
        
            Examples
            --------
            >>> from numpy.polynomial.laguerre import lagfromroots, lagval
            >>> coef = lagfromroots((-1, 0, 1))
            >>> lagval((-1, 0, 1), coef)
            array([0.,  0.,  0.])
            >>> coef = lagfromroots((-1j, 1j))
            >>> lagval((-1j, 1j), coef)
            array([0.+0.j, 0.+0.j])
        
            
        """
    @staticmethod
    def _int(c, m = 1, k = list(), lbnd = 0, scl = 1, axis = 0):
        """
        
            Integrate a Laguerre series.
        
            Returns the Laguerre series coefficients `c` integrated `m` times from
            `lbnd` along `axis`. At each iteration the resulting series is
            **multiplied** by `scl` and an integration constant, `k`, is added.
            The scaling factor is for use in a linear change of variable.  ("Buyer
            beware": note that, depending on what one is doing, one may want `scl`
            to be the reciprocal of what one might expect; for more information,
            see the Notes section below.)  The argument `c` is an array of
            coefficients from low to high degree along each axis, e.g., [1,2,3]
            represents the series ``L_0 + 2*L_1 + 3*L_2`` while [[1,2],[1,2]]
            represents ``1*L_0(x)*L_0(y) + 1*L_1(x)*L_0(y) + 2*L_0(x)*L_1(y) +
            2*L_1(x)*L_1(y)`` if axis=0 is ``x`` and axis=1 is ``y``.
        
        
            Parameters
            ----------
            c : array_like
                Array of Laguerre series coefficients. If `c` is multidimensional
                the different axis correspond to different variables with the
                degree in each axis given by the corresponding index.
            m : int, optional
                Order of integration, must be positive. (Default: 1)
            k : {[], list, scalar}, optional
                Integration constant(s).  The value of the first integral at
                ``lbnd`` is the first value in the list, the value of the second
                integral at ``lbnd`` is the second value, etc.  If ``k == []`` (the
                default), all constants are set to zero.  If ``m == 1``, a single
                scalar can be given instead of a list.
            lbnd : scalar, optional
                The lower bound of the integral. (Default: 0)
            scl : scalar, optional
                Following each integration the result is *multiplied* by `scl`
                before the integration constant is added. (Default: 1)
            axis : int, optional
                Axis over which the integral is taken. (Default: 0).
        
                .. versionadded:: 1.7.0
        
            Returns
            -------
            S : ndarray
                Laguerre series coefficients of the integral.
        
            Raises
            ------
            ValueError
                If ``m < 0``, ``len(k) > m``, ``np.ndim(lbnd) != 0``, or
                ``np.ndim(scl) != 0``.
        
            See Also
            --------
            lagder
        
            Notes
            -----
            Note that the result of each integration is *multiplied* by `scl`.
            Why is this important to note?  Say one is making a linear change of
            variable :math:`u = ax + b` in an integral relative to `x`.  Then
            :math:`dx = du/a`, so one will need to set `scl` equal to
            :math:`1/a` - perhaps not what one would have first thought.
        
            Also note that, in general, the result of integrating a C-series needs
            to be "reprojected" onto the C-series basis set.  Thus, typically,
            the result of this function is "unintuitive," albeit correct; see
            Examples section below.
        
            Examples
            --------
            >>> from numpy.polynomial.laguerre import lagint
            >>> lagint([1,2,3])
            array([ 1.,  1.,  1., -3.])
            >>> lagint([1,2,3], m=2)
            array([ 1.,  0.,  0., -4.,  3.])
            >>> lagint([1,2,3], k=1)
            array([ 2.,  1.,  1., -3.])
            >>> lagint([1,2,3], lbnd=-1)
            array([11.5,  1. ,  1. , -3. ])
            >>> lagint([1,2], m=2, k=[1,2], lbnd=-1)
            array([ 11.16666667,  -5.        ,  -3.        ,   2.        ]) # may vary
        
            
        """
    @staticmethod
    def _line(off, scl):
        """
        
            Laguerre series whose graph is a straight line.
        
            Parameters
            ----------
            off, scl : scalars
                The specified line is given by ``off + scl*x``.
        
            Returns
            -------
            y : ndarray
                This module's representation of the Laguerre series for
                ``off + scl*x``.
        
            See Also
            --------
            numpy.polynomial.polynomial.polyline
            numpy.polynomial.chebyshev.chebline
            numpy.polynomial.legendre.legline
            numpy.polynomial.hermite.hermline
            numpy.polynomial.hermite_e.hermeline
        
            Examples
            --------
            >>> from numpy.polynomial.laguerre import lagline, lagval
            >>> lagval(0,lagline(3, 2))
            3.0
            >>> lagval(1,lagline(3, 2))
            5.0
        
            
        """
    @staticmethod
    def _mul(c1, c2):
        """
        
            Multiply one Laguerre series by another.
        
            Returns the product of two Laguerre series `c1` * `c2`.  The arguments
            are sequences of coefficients, from lowest order "term" to highest,
            e.g., [1,2,3] represents the series ``P_0 + 2*P_1 + 3*P_2``.
        
            Parameters
            ----------
            c1, c2 : array_like
                1-D arrays of Laguerre series coefficients ordered from low to
                high.
        
            Returns
            -------
            out : ndarray
                Of Laguerre series coefficients representing their product.
        
            See Also
            --------
            lagadd, lagsub, lagmulx, lagdiv, lagpow
        
            Notes
            -----
            In general, the (polynomial) product of two C-series results in terms
            that are not in the Laguerre polynomial basis set.  Thus, to express
            the product as a Laguerre series, it is necessary to "reproject" the
            product onto said basis set, which may produce "unintuitive" (but
            correct) results; see Examples section below.
        
            Examples
            --------
            >>> from numpy.polynomial.laguerre import lagmul
            >>> lagmul([1, 2, 3], [0, 1, 2])
            array([  8., -13.,  38., -51.,  36.])
        
            
        """
    @staticmethod
    def _pow(c, pow, maxpower = 16):
        """
        Raise a Laguerre series to a power.
        
            Returns the Laguerre series `c` raised to the power `pow`. The
            argument `c` is a sequence of coefficients ordered from low to high.
            i.e., [1,2,3] is the series  ``P_0 + 2*P_1 + 3*P_2.``
        
            Parameters
            ----------
            c : array_like
                1-D array of Laguerre series coefficients ordered from low to
                high.
            pow : integer
                Power to which the series will be raised
            maxpower : integer, optional
                Maximum power allowed. This is mainly to limit growth of the series
                to unmanageable size. Default is 16
        
            Returns
            -------
            coef : ndarray
                Laguerre series of power.
        
            See Also
            --------
            lagadd, lagsub, lagmulx, lagmul, lagdiv
        
            Examples
            --------
            >>> from numpy.polynomial.laguerre import lagpow
            >>> lagpow([1, 2, 3], 2)
            array([ 14., -16.,  56., -72.,  54.])
        
            
        """
    @staticmethod
    def _roots(c):
        """
        
            Compute the roots of a Laguerre series.
        
            Return the roots (a.k.a. "zeros") of the polynomial
        
            .. math:: p(x) = \\sum_i c[i] * L_i(x).
        
            Parameters
            ----------
            c : 1-D array_like
                1-D array of coefficients.
        
            Returns
            -------
            out : ndarray
                Array of the roots of the series. If all the roots are real,
                then `out` is also real, otherwise it is complex.
        
            See Also
            --------
            numpy.polynomial.polynomial.polyroots
            numpy.polynomial.legendre.legroots
            numpy.polynomial.chebyshev.chebroots
            numpy.polynomial.hermite.hermroots
            numpy.polynomial.hermite_e.hermeroots
        
            Notes
            -----
            The root estimates are obtained as the eigenvalues of the companion
            matrix, Roots far from the origin of the complex plane may have large
            errors due to the numerical instability of the series for such
            values. Roots with multiplicity greater than 1 will also show larger
            errors as the value of the series near such points is relatively
            insensitive to errors in the roots. Isolated roots near the origin can
            be improved by a few iterations of Newton's method.
        
            The Laguerre series basis polynomials aren't powers of `x` so the
            results of this function may seem unintuitive.
        
            Examples
            --------
            >>> from numpy.polynomial.laguerre import lagroots, lagfromroots
            >>> coef = lagfromroots([0, 1, 2])
            >>> coef
            array([  2.,  -8.,  12.,  -6.])
            >>> lagroots(coef)
            array([-4.4408921e-16,  1.0000000e+00,  2.0000000e+00])
        
            
        """
    @staticmethod
    def _sub(c1, c2):
        """
        
            Subtract one Laguerre series from another.
        
            Returns the difference of two Laguerre series `c1` - `c2`.  The
            sequences of coefficients are from lowest order term to highest, i.e.,
            [1,2,3] represents the series ``P_0 + 2*P_1 + 3*P_2``.
        
            Parameters
            ----------
            c1, c2 : array_like
                1-D arrays of Laguerre series coefficients ordered from low to
                high.
        
            Returns
            -------
            out : ndarray
                Of Laguerre series coefficients representing their difference.
        
            See Also
            --------
            lagadd, lagmulx, lagmul, lagdiv, lagpow
        
            Notes
            -----
            Unlike multiplication, division, etc., the difference of two Laguerre
            series is a Laguerre series (without having to "reproject" the result
            onto the basis set) so subtraction, just like that of "standard"
            polynomials, is simply "component-wise."
        
            Examples
            --------
            >>> from numpy.polynomial.laguerre import lagsub
            >>> lagsub([1, 2, 3, 4], [1, 2, 3])
            array([0.,  0.,  0.,  4.])
        
            
        """
    @staticmethod
    def _val(x, c, tensor = True):
        """
        
            Evaluate a Laguerre series at points x.
        
            If `c` is of length `n + 1`, this function returns the value:
        
            .. math:: p(x) = c_0 * L_0(x) + c_1 * L_1(x) + ... + c_n * L_n(x)
        
            The parameter `x` is converted to an array only if it is a tuple or a
            list, otherwise it is treated as a scalar. In either case, either `x`
            or its elements must support multiplication and addition both with
            themselves and with the elements of `c`.
        
            If `c` is a 1-D array, then `p(x)` will have the same shape as `x`.  If
            `c` is multidimensional, then the shape of the result depends on the
            value of `tensor`. If `tensor` is true the shape will be c.shape[1:] +
            x.shape. If `tensor` is false the shape will be c.shape[1:]. Note that
            scalars have shape (,).
        
            Trailing zeros in the coefficients will be used in the evaluation, so
            they should be avoided if efficiency is a concern.
        
            Parameters
            ----------
            x : array_like, compatible object
                If `x` is a list or tuple, it is converted to an ndarray, otherwise
                it is left unchanged and treated as a scalar. In either case, `x`
                or its elements must support addition and multiplication with
                themselves and with the elements of `c`.
            c : array_like
                Array of coefficients ordered so that the coefficients for terms of
                degree n are contained in c[n]. If `c` is multidimensional the
                remaining indices enumerate multiple polynomials. In the two
                dimensional case the coefficients may be thought of as stored in
                the columns of `c`.
            tensor : boolean, optional
                If True, the shape of the coefficient array is extended with ones
                on the right, one for each dimension of `x`. Scalars have dimension 0
                for this action. The result is that every column of coefficients in
                `c` is evaluated for every element of `x`. If False, `x` is broadcast
                over the columns of `c` for the evaluation.  This keyword is useful
                when `c` is multidimensional. The default value is True.
        
                .. versionadded:: 1.7.0
        
            Returns
            -------
            values : ndarray, algebra_like
                The shape of the return value is described above.
        
            See Also
            --------
            lagval2d, laggrid2d, lagval3d, laggrid3d
        
            Notes
            -----
            The evaluation uses Clenshaw recursion, aka synthetic division.
        
            Examples
            --------
            >>> from numpy.polynomial.laguerre import lagval
            >>> coef = [1,2,3]
            >>> lagval(1, coef)
            -0.5
            >>> lagval([[1,2],[3,4]], coef)
            array([[-0.5, -4. ],
                   [-4.5, -2. ]])
        
            
        """
def lag2poly(c):
    """
    
        Convert a Laguerre series to a polynomial.
    
        Convert an array representing the coefficients of a Laguerre series,
        ordered from lowest degree to highest, to an array of the coefficients
        of the equivalent polynomial (relative to the "standard" basis) ordered
        from lowest to highest degree.
    
        Parameters
        ----------
        c : array_like
            1-D array containing the Laguerre series coefficients, ordered
            from lowest order term to highest.
    
        Returns
        -------
        pol : ndarray
            1-D array containing the coefficients of the equivalent polynomial
            (relative to the "standard" basis) ordered from lowest order term
            to highest.
    
        See Also
        --------
        poly2lag
    
        Notes
        -----
        The easy way to do conversions between polynomial basis sets
        is to use the convert method of a class instance.
    
        Examples
        --------
        >>> from numpy.polynomial.laguerre import lag2poly
        >>> lag2poly([ 23., -63.,  58., -18.])
        array([0., 1., 2., 3.])
    
        
    """
def lagadd(c1, c2):
    """
    
        Add one Laguerre series to another.
    
        Returns the sum of two Laguerre series `c1` + `c2`.  The arguments
        are sequences of coefficients ordered from lowest order term to
        highest, i.e., [1,2,3] represents the series ``P_0 + 2*P_1 + 3*P_2``.
    
        Parameters
        ----------
        c1, c2 : array_like
            1-D arrays of Laguerre series coefficients ordered from low to
            high.
    
        Returns
        -------
        out : ndarray
            Array representing the Laguerre series of their sum.
    
        See Also
        --------
        lagsub, lagmulx, lagmul, lagdiv, lagpow
    
        Notes
        -----
        Unlike multiplication, division, etc., the sum of two Laguerre series
        is a Laguerre series (without having to "reproject" the result onto
        the basis set) so addition, just like that of "standard" polynomials,
        is simply "component-wise."
    
        Examples
        --------
        >>> from numpy.polynomial.laguerre import lagadd
        >>> lagadd([1, 2, 3], [1, 2, 3, 4])
        array([2.,  4.,  6.,  4.])
    
    
        
    """
def lagcompanion(c):
    """
    
        Return the companion matrix of c.
    
        The usual companion matrix of the Laguerre polynomials is already
        symmetric when `c` is a basis Laguerre polynomial, so no scaling is
        applied.
    
        Parameters
        ----------
        c : array_like
            1-D array of Laguerre series coefficients ordered from low to high
            degree.
    
        Returns
        -------
        mat : ndarray
            Companion matrix of dimensions (deg, deg).
    
        Notes
        -----
    
        .. versionadded:: 1.7.0
    
        
    """
def lagder(c, m = 1, scl = 1, axis = 0):
    """
    
        Differentiate a Laguerre series.
    
        Returns the Laguerre series coefficients `c` differentiated `m` times
        along `axis`.  At each iteration the result is multiplied by `scl` (the
        scaling factor is for use in a linear change of variable). The argument
        `c` is an array of coefficients from low to high degree along each
        axis, e.g., [1,2,3] represents the series ``1*L_0 + 2*L_1 + 3*L_2``
        while [[1,2],[1,2]] represents ``1*L_0(x)*L_0(y) + 1*L_1(x)*L_0(y) +
        2*L_0(x)*L_1(y) + 2*L_1(x)*L_1(y)`` if axis=0 is ``x`` and axis=1 is
        ``y``.
    
        Parameters
        ----------
        c : array_like
            Array of Laguerre series coefficients. If `c` is multidimensional
            the different axis correspond to different variables with the
            degree in each axis given by the corresponding index.
        m : int, optional
            Number of derivatives taken, must be non-negative. (Default: 1)
        scl : scalar, optional
            Each differentiation is multiplied by `scl`.  The end result is
            multiplication by ``scl**m``.  This is for use in a linear change of
            variable. (Default: 1)
        axis : int, optional
            Axis over which the derivative is taken. (Default: 0).
    
            .. versionadded:: 1.7.0
    
        Returns
        -------
        der : ndarray
            Laguerre series of the derivative.
    
        See Also
        --------
        lagint
    
        Notes
        -----
        In general, the result of differentiating a Laguerre series does not
        resemble the same operation on a power series. Thus the result of this
        function may be "unintuitive," albeit correct; see Examples section
        below.
    
        Examples
        --------
        >>> from numpy.polynomial.laguerre import lagder
        >>> lagder([ 1.,  1.,  1., -3.])
        array([1.,  2.,  3.])
        >>> lagder([ 1.,  0.,  0., -4.,  3.], m=2)
        array([1.,  2.,  3.])
    
        
    """
def lagdiv(c1, c2):
    """
    
        Divide one Laguerre series by another.
    
        Returns the quotient-with-remainder of two Laguerre series
        `c1` / `c2`.  The arguments are sequences of coefficients from lowest
        order "term" to highest, e.g., [1,2,3] represents the series
        ``P_0 + 2*P_1 + 3*P_2``.
    
        Parameters
        ----------
        c1, c2 : array_like
            1-D arrays of Laguerre series coefficients ordered from low to
            high.
    
        Returns
        -------
        [quo, rem] : ndarrays
            Of Laguerre series coefficients representing the quotient and
            remainder.
    
        See Also
        --------
        lagadd, lagsub, lagmulx, lagmul, lagpow
    
        Notes
        -----
        In general, the (polynomial) division of one Laguerre series by another
        results in quotient and remainder terms that are not in the Laguerre
        polynomial basis set.  Thus, to express these results as a Laguerre
        series, it is necessary to "reproject" the results onto the Laguerre
        basis set, which may produce "unintuitive" (but correct) results; see
        Examples section below.
    
        Examples
        --------
        >>> from numpy.polynomial.laguerre import lagdiv
        >>> lagdiv([  8., -13.,  38., -51.,  36.], [0, 1, 2])
        (array([1., 2., 3.]), array([0.]))
        >>> lagdiv([  9., -12.,  38., -51.,  36.], [0, 1, 2])
        (array([1., 2., 3.]), array([1., 1.]))
    
        
    """
def lagfit(x, y, deg, rcond = None, full = False, w = None):
    """
    
        Least squares fit of Laguerre series to data.
    
        Return the coefficients of a Laguerre series of degree `deg` that is the
        least squares fit to the data values `y` given at points `x`. If `y` is
        1-D the returned coefficients will also be 1-D. If `y` is 2-D multiple
        fits are done, one for each column of `y`, and the resulting
        coefficients are stored in the corresponding columns of a 2-D return.
        The fitted polynomial(s) are in the form
    
        .. math::  p(x) = c_0 + c_1 * L_1(x) + ... + c_n * L_n(x),
    
        where ``n`` is `deg`.
    
        Parameters
        ----------
        x : array_like, shape (M,)
            x-coordinates of the M sample points ``(x[i], y[i])``.
        y : array_like, shape (M,) or (M, K)
            y-coordinates of the sample points. Several data sets of sample
            points sharing the same x-coordinates can be fitted at once by
            passing in a 2D-array that contains one dataset per column.
        deg : int or 1-D array_like
            Degree(s) of the fitting polynomials. If `deg` is a single integer
            all terms up to and including the `deg`'th term are included in the
            fit. For NumPy versions >= 1.11.0 a list of integers specifying the
            degrees of the terms to include may be used instead.
        rcond : float, optional
            Relative condition number of the fit. Singular values smaller than
            this relative to the largest singular value will be ignored. The
            default value is len(x)*eps, where eps is the relative precision of
            the float type, about 2e-16 in most cases.
        full : bool, optional
            Switch determining nature of return value. When it is False (the
            default) just the coefficients are returned, when True diagnostic
            information from the singular value decomposition is also returned.
        w : array_like, shape (`M`,), optional
            Weights. If not None, the weight ``w[i]`` applies to the unsquared
            residual ``y[i] - y_hat[i]`` at ``x[i]``. Ideally the weights are
            chosen so that the errors of the products ``w[i]*y[i]`` all have the
            same variance.  When using inverse-variance weighting, use
            ``w[i] = 1/sigma(y[i])``.  The default value is None.
    
        Returns
        -------
        coef : ndarray, shape (M,) or (M, K)
            Laguerre coefficients ordered from low to high. If `y` was 2-D,
            the coefficients for the data in column *k*  of `y` are in column
            *k*.
    
        [residuals, rank, singular_values, rcond] : list
            These values are only returned if ``full == True``
    
            - residuals -- sum of squared residuals of the least squares fit
            - rank -- the numerical rank of the scaled Vandermonde matrix
            - singular_values -- singular values of the scaled Vandermonde matrix
            - rcond -- value of `rcond`.
    
            For more details, see `numpy.linalg.lstsq`.
    
        Warns
        -----
        RankWarning
            The rank of the coefficient matrix in the least-squares fit is
            deficient. The warning is only raised if ``full == False``.  The
            warnings can be turned off by
    
            >>> import warnings
            >>> warnings.simplefilter('ignore', np.RankWarning)
    
        See Also
        --------
        numpy.polynomial.polynomial.polyfit
        numpy.polynomial.legendre.legfit
        numpy.polynomial.chebyshev.chebfit
        numpy.polynomial.hermite.hermfit
        numpy.polynomial.hermite_e.hermefit
        lagval : Evaluates a Laguerre series.
        lagvander : pseudo Vandermonde matrix of Laguerre series.
        lagweight : Laguerre weight function.
        numpy.linalg.lstsq : Computes a least-squares fit from the matrix.
        scipy.interpolate.UnivariateSpline : Computes spline fits.
    
        Notes
        -----
        The solution is the coefficients of the Laguerre series ``p`` that
        minimizes the sum of the weighted squared errors
    
        .. math:: E = \\sum_j w_j^2 * |y_j - p(x_j)|^2,
    
        where the :math:`w_j` are the weights. This problem is solved by
        setting up as the (typically) overdetermined matrix equation
    
        .. math:: V(x) * c = w * y,
    
        where ``V`` is the weighted pseudo Vandermonde matrix of `x`, ``c`` are the
        coefficients to be solved for, `w` are the weights, and `y` are the
        observed values.  This equation is then solved using the singular value
        decomposition of ``V``.
    
        If some of the singular values of `V` are so small that they are
        neglected, then a `RankWarning` will be issued. This means that the
        coefficient values may be poorly determined. Using a lower order fit
        will usually get rid of the warning.  The `rcond` parameter can also be
        set to a value smaller than its default, but the resulting fit may be
        spurious and have large contributions from roundoff error.
    
        Fits using Laguerre series are probably most useful when the data can
        be approximated by ``sqrt(w(x)) * p(x)``, where ``w(x)`` is the Laguerre
        weight. In that case the weight ``sqrt(w(x[i]))`` should be used
        together with data values ``y[i]/sqrt(w(x[i]))``. The weight function is
        available as `lagweight`.
    
        References
        ----------
        .. [1] Wikipedia, "Curve fitting",
               https://en.wikipedia.org/wiki/Curve_fitting
    
        Examples
        --------
        >>> from numpy.polynomial.laguerre import lagfit, lagval
        >>> x = np.linspace(0, 10)
        >>> err = np.random.randn(len(x))/10
        >>> y = lagval(x, [1, 2, 3]) + err
        >>> lagfit(x, y, 2)
        array([ 0.96971004,  2.00193749,  3.00288744]) # may vary
    
        
    """
def lagfromroots(roots):
    """
    
        Generate a Laguerre series with given roots.
    
        The function returns the coefficients of the polynomial
    
        .. math:: p(x) = (x - r_0) * (x - r_1) * ... * (x - r_n),
    
        in Laguerre form, where the `r_n` are the roots specified in `roots`.
        If a zero has multiplicity n, then it must appear in `roots` n times.
        For instance, if 2 is a root of multiplicity three and 3 is a root of
        multiplicity 2, then `roots` looks something like [2, 2, 2, 3, 3]. The
        roots can appear in any order.
    
        If the returned coefficients are `c`, then
    
        .. math:: p(x) = c_0 + c_1 * L_1(x) + ... +  c_n * L_n(x)
    
        The coefficient of the last term is not generally 1 for monic
        polynomials in Laguerre form.
    
        Parameters
        ----------
        roots : array_like
            Sequence containing the roots.
    
        Returns
        -------
        out : ndarray
            1-D array of coefficients.  If all roots are real then `out` is a
            real array, if some of the roots are complex, then `out` is complex
            even if all the coefficients in the result are real (see Examples
            below).
    
        See Also
        --------
        numpy.polynomial.polynomial.polyfromroots
        numpy.polynomial.legendre.legfromroots
        numpy.polynomial.chebyshev.chebfromroots
        numpy.polynomial.hermite.hermfromroots
        numpy.polynomial.hermite_e.hermefromroots
    
        Examples
        --------
        >>> from numpy.polynomial.laguerre import lagfromroots, lagval
        >>> coef = lagfromroots((-1, 0, 1))
        >>> lagval((-1, 0, 1), coef)
        array([0.,  0.,  0.])
        >>> coef = lagfromroots((-1j, 1j))
        >>> lagval((-1j, 1j), coef)
        array([0.+0.j, 0.+0.j])
    
        
    """
def laggauss(deg):
    """
    
        Gauss-Laguerre quadrature.
    
        Computes the sample points and weights for Gauss-Laguerre quadrature.
        These sample points and weights will correctly integrate polynomials of
        degree :math:`2*deg - 1` or less over the interval :math:`[0, \\inf]`
        with the weight function :math:`f(x) = \\exp(-x)`.
    
        Parameters
        ----------
        deg : int
            Number of sample points and weights. It must be >= 1.
    
        Returns
        -------
        x : ndarray
            1-D ndarray containing the sample points.
        y : ndarray
            1-D ndarray containing the weights.
    
        Notes
        -----
    
        .. versionadded:: 1.7.0
    
        The results have only been tested up to degree 100 higher degrees may
        be problematic. The weights are determined by using the fact that
    
        .. math:: w_k = c / (L'_n(x_k) * L_{n-1}(x_k))
    
        where :math:`c` is a constant independent of :math:`k` and :math:`x_k`
        is the k'th root of :math:`L_n`, and then scaling the results to get
        the right value when integrating 1.
    
        
    """
def laggrid2d(x, y, c):
    """
    
        Evaluate a 2-D Laguerre series on the Cartesian product of x and y.
    
        This function returns the values:
    
        .. math:: p(a,b) = \\sum_{i,j} c_{i,j} * L_i(a) * L_j(b)
    
        where the points `(a, b)` consist of all pairs formed by taking
        `a` from `x` and `b` from `y`. The resulting points form a grid with
        `x` in the first dimension and `y` in the second.
    
        The parameters `x` and `y` are converted to arrays only if they are
        tuples or a lists, otherwise they are treated as a scalars. In either
        case, either `x` and `y` or their elements must support multiplication
        and addition both with themselves and with the elements of `c`.
    
        If `c` has fewer than two dimensions, ones are implicitly appended to
        its shape to make it 2-D. The shape of the result will be c.shape[2:] +
        x.shape + y.shape.
    
        Parameters
        ----------
        x, y : array_like, compatible objects
            The two dimensional series is evaluated at the points in the
            Cartesian product of `x` and `y`.  If `x` or `y` is a list or
            tuple, it is first converted to an ndarray, otherwise it is left
            unchanged and, if it isn't an ndarray, it is treated as a scalar.
        c : array_like
            Array of coefficients ordered so that the coefficient of the term of
            multi-degree i,j is contained in `c[i,j]`. If `c` has dimension
            greater than two the remaining indices enumerate multiple sets of
            coefficients.
    
        Returns
        -------
        values : ndarray, compatible object
            The values of the two dimensional Chebyshev series at points in the
            Cartesian product of `x` and `y`.
    
        See Also
        --------
        lagval, lagval2d, lagval3d, laggrid3d
    
        Notes
        -----
    
        .. versionadded:: 1.7.0
    
        
    """
def laggrid3d(x, y, z, c):
    """
    
        Evaluate a 3-D Laguerre series on the Cartesian product of x, y, and z.
    
        This function returns the values:
    
        .. math:: p(a,b,c) = \\sum_{i,j,k} c_{i,j,k} * L_i(a) * L_j(b) * L_k(c)
    
        where the points `(a, b, c)` consist of all triples formed by taking
        `a` from `x`, `b` from `y`, and `c` from `z`. The resulting points form
        a grid with `x` in the first dimension, `y` in the second, and `z` in
        the third.
    
        The parameters `x`, `y`, and `z` are converted to arrays only if they
        are tuples or a lists, otherwise they are treated as a scalars. In
        either case, either `x`, `y`, and `z` or their elements must support
        multiplication and addition both with themselves and with the elements
        of `c`.
    
        If `c` has fewer than three dimensions, ones are implicitly appended to
        its shape to make it 3-D. The shape of the result will be c.shape[3:] +
        x.shape + y.shape + z.shape.
    
        Parameters
        ----------
        x, y, z : array_like, compatible objects
            The three dimensional series is evaluated at the points in the
            Cartesian product of `x`, `y`, and `z`.  If `x`,`y`, or `z` is a
            list or tuple, it is first converted to an ndarray, otherwise it is
            left unchanged and, if it isn't an ndarray, it is treated as a
            scalar.
        c : array_like
            Array of coefficients ordered so that the coefficients for terms of
            degree i,j are contained in ``c[i,j]``. If `c` has dimension
            greater than two the remaining indices enumerate multiple sets of
            coefficients.
    
        Returns
        -------
        values : ndarray, compatible object
            The values of the two dimensional polynomial at points in the Cartesian
            product of `x` and `y`.
    
        See Also
        --------
        lagval, lagval2d, laggrid2d, lagval3d
    
        Notes
        -----
    
        .. versionadded:: 1.7.0
    
        
    """
def lagint(c, m = 1, k = list(), lbnd = 0, scl = 1, axis = 0):
    """
    
        Integrate a Laguerre series.
    
        Returns the Laguerre series coefficients `c` integrated `m` times from
        `lbnd` along `axis`. At each iteration the resulting series is
        **multiplied** by `scl` and an integration constant, `k`, is added.
        The scaling factor is for use in a linear change of variable.  ("Buyer
        beware": note that, depending on what one is doing, one may want `scl`
        to be the reciprocal of what one might expect; for more information,
        see the Notes section below.)  The argument `c` is an array of
        coefficients from low to high degree along each axis, e.g., [1,2,3]
        represents the series ``L_0 + 2*L_1 + 3*L_2`` while [[1,2],[1,2]]
        represents ``1*L_0(x)*L_0(y) + 1*L_1(x)*L_0(y) + 2*L_0(x)*L_1(y) +
        2*L_1(x)*L_1(y)`` if axis=0 is ``x`` and axis=1 is ``y``.
    
    
        Parameters
        ----------
        c : array_like
            Array of Laguerre series coefficients. If `c` is multidimensional
            the different axis correspond to different variables with the
            degree in each axis given by the corresponding index.
        m : int, optional
            Order of integration, must be positive. (Default: 1)
        k : {[], list, scalar}, optional
            Integration constant(s).  The value of the first integral at
            ``lbnd`` is the first value in the list, the value of the second
            integral at ``lbnd`` is the second value, etc.  If ``k == []`` (the
            default), all constants are set to zero.  If ``m == 1``, a single
            scalar can be given instead of a list.
        lbnd : scalar, optional
            The lower bound of the integral. (Default: 0)
        scl : scalar, optional
            Following each integration the result is *multiplied* by `scl`
            before the integration constant is added. (Default: 1)
        axis : int, optional
            Axis over which the integral is taken. (Default: 0).
    
            .. versionadded:: 1.7.0
    
        Returns
        -------
        S : ndarray
            Laguerre series coefficients of the integral.
    
        Raises
        ------
        ValueError
            If ``m < 0``, ``len(k) > m``, ``np.ndim(lbnd) != 0``, or
            ``np.ndim(scl) != 0``.
    
        See Also
        --------
        lagder
    
        Notes
        -----
        Note that the result of each integration is *multiplied* by `scl`.
        Why is this important to note?  Say one is making a linear change of
        variable :math:`u = ax + b` in an integral relative to `x`.  Then
        :math:`dx = du/a`, so one will need to set `scl` equal to
        :math:`1/a` - perhaps not what one would have first thought.
    
        Also note that, in general, the result of integrating a C-series needs
        to be "reprojected" onto the C-series basis set.  Thus, typically,
        the result of this function is "unintuitive," albeit correct; see
        Examples section below.
    
        Examples
        --------
        >>> from numpy.polynomial.laguerre import lagint
        >>> lagint([1,2,3])
        array([ 1.,  1.,  1., -3.])
        >>> lagint([1,2,3], m=2)
        array([ 1.,  0.,  0., -4.,  3.])
        >>> lagint([1,2,3], k=1)
        array([ 2.,  1.,  1., -3.])
        >>> lagint([1,2,3], lbnd=-1)
        array([11.5,  1. ,  1. , -3. ])
        >>> lagint([1,2], m=2, k=[1,2], lbnd=-1)
        array([ 11.16666667,  -5.        ,  -3.        ,   2.        ]) # may vary
    
        
    """
def lagline(off, scl):
    """
    
        Laguerre series whose graph is a straight line.
    
        Parameters
        ----------
        off, scl : scalars
            The specified line is given by ``off + scl*x``.
    
        Returns
        -------
        y : ndarray
            This module's representation of the Laguerre series for
            ``off + scl*x``.
    
        See Also
        --------
        numpy.polynomial.polynomial.polyline
        numpy.polynomial.chebyshev.chebline
        numpy.polynomial.legendre.legline
        numpy.polynomial.hermite.hermline
        numpy.polynomial.hermite_e.hermeline
    
        Examples
        --------
        >>> from numpy.polynomial.laguerre import lagline, lagval
        >>> lagval(0,lagline(3, 2))
        3.0
        >>> lagval(1,lagline(3, 2))
        5.0
    
        
    """
def lagmul(c1, c2):
    """
    
        Multiply one Laguerre series by another.
    
        Returns the product of two Laguerre series `c1` * `c2`.  The arguments
        are sequences of coefficients, from lowest order "term" to highest,
        e.g., [1,2,3] represents the series ``P_0 + 2*P_1 + 3*P_2``.
    
        Parameters
        ----------
        c1, c2 : array_like
            1-D arrays of Laguerre series coefficients ordered from low to
            high.
    
        Returns
        -------
        out : ndarray
            Of Laguerre series coefficients representing their product.
    
        See Also
        --------
        lagadd, lagsub, lagmulx, lagdiv, lagpow
    
        Notes
        -----
        In general, the (polynomial) product of two C-series results in terms
        that are not in the Laguerre polynomial basis set.  Thus, to express
        the product as a Laguerre series, it is necessary to "reproject" the
        product onto said basis set, which may produce "unintuitive" (but
        correct) results; see Examples section below.
    
        Examples
        --------
        >>> from numpy.polynomial.laguerre import lagmul
        >>> lagmul([1, 2, 3], [0, 1, 2])
        array([  8., -13.,  38., -51.,  36.])
    
        
    """
def lagmulx(c):
    """
    Multiply a Laguerre series by x.
    
        Multiply the Laguerre series `c` by x, where x is the independent
        variable.
    
    
        Parameters
        ----------
        c : array_like
            1-D array of Laguerre series coefficients ordered from low to
            high.
    
        Returns
        -------
        out : ndarray
            Array representing the result of the multiplication.
    
        See Also
        --------
        lagadd, lagsub, lagmul, lagdiv, lagpow
    
        Notes
        -----
        The multiplication uses the recursion relationship for Laguerre
        polynomials in the form
    
        .. math::
    
            xP_i(x) = (-(i + 1)*P_{i + 1}(x) + (2i + 1)P_{i}(x) - iP_{i - 1}(x))
    
        Examples
        --------
        >>> from numpy.polynomial.laguerre import lagmulx
        >>> lagmulx([1, 2, 3])
        array([-1.,  -1.,  11.,  -9.])
    
        
    """
def lagpow(c, pow, maxpower = 16):
    """
    Raise a Laguerre series to a power.
    
        Returns the Laguerre series `c` raised to the power `pow`. The
        argument `c` is a sequence of coefficients ordered from low to high.
        i.e., [1,2,3] is the series  ``P_0 + 2*P_1 + 3*P_2.``
    
        Parameters
        ----------
        c : array_like
            1-D array of Laguerre series coefficients ordered from low to
            high.
        pow : integer
            Power to which the series will be raised
        maxpower : integer, optional
            Maximum power allowed. This is mainly to limit growth of the series
            to unmanageable size. Default is 16
    
        Returns
        -------
        coef : ndarray
            Laguerre series of power.
    
        See Also
        --------
        lagadd, lagsub, lagmulx, lagmul, lagdiv
    
        Examples
        --------
        >>> from numpy.polynomial.laguerre import lagpow
        >>> lagpow([1, 2, 3], 2)
        array([ 14., -16.,  56., -72.,  54.])
    
        
    """
def lagroots(c):
    """
    
        Compute the roots of a Laguerre series.
    
        Return the roots (a.k.a. "zeros") of the polynomial
    
        .. math:: p(x) = \\sum_i c[i] * L_i(x).
    
        Parameters
        ----------
        c : 1-D array_like
            1-D array of coefficients.
    
        Returns
        -------
        out : ndarray
            Array of the roots of the series. If all the roots are real,
            then `out` is also real, otherwise it is complex.
    
        See Also
        --------
        numpy.polynomial.polynomial.polyroots
        numpy.polynomial.legendre.legroots
        numpy.polynomial.chebyshev.chebroots
        numpy.polynomial.hermite.hermroots
        numpy.polynomial.hermite_e.hermeroots
    
        Notes
        -----
        The root estimates are obtained as the eigenvalues of the companion
        matrix, Roots far from the origin of the complex plane may have large
        errors due to the numerical instability of the series for such
        values. Roots with multiplicity greater than 1 will also show larger
        errors as the value of the series near such points is relatively
        insensitive to errors in the roots. Isolated roots near the origin can
        be improved by a few iterations of Newton's method.
    
        The Laguerre series basis polynomials aren't powers of `x` so the
        results of this function may seem unintuitive.
    
        Examples
        --------
        >>> from numpy.polynomial.laguerre import lagroots, lagfromroots
        >>> coef = lagfromroots([0, 1, 2])
        >>> coef
        array([  2.,  -8.,  12.,  -6.])
        >>> lagroots(coef)
        array([-4.4408921e-16,  1.0000000e+00,  2.0000000e+00])
    
        
    """
def lagsub(c1, c2):
    """
    
        Subtract one Laguerre series from another.
    
        Returns the difference of two Laguerre series `c1` - `c2`.  The
        sequences of coefficients are from lowest order term to highest, i.e.,
        [1,2,3] represents the series ``P_0 + 2*P_1 + 3*P_2``.
    
        Parameters
        ----------
        c1, c2 : array_like
            1-D arrays of Laguerre series coefficients ordered from low to
            high.
    
        Returns
        -------
        out : ndarray
            Of Laguerre series coefficients representing their difference.
    
        See Also
        --------
        lagadd, lagmulx, lagmul, lagdiv, lagpow
    
        Notes
        -----
        Unlike multiplication, division, etc., the difference of two Laguerre
        series is a Laguerre series (without having to "reproject" the result
        onto the basis set) so subtraction, just like that of "standard"
        polynomials, is simply "component-wise."
    
        Examples
        --------
        >>> from numpy.polynomial.laguerre import lagsub
        >>> lagsub([1, 2, 3, 4], [1, 2, 3])
        array([0.,  0.,  0.,  4.])
    
        
    """
def lagval(x, c, tensor = True):
    """
    
        Evaluate a Laguerre series at points x.
    
        If `c` is of length `n + 1`, this function returns the value:
    
        .. math:: p(x) = c_0 * L_0(x) + c_1 * L_1(x) + ... + c_n * L_n(x)
    
        The parameter `x` is converted to an array only if it is a tuple or a
        list, otherwise it is treated as a scalar. In either case, either `x`
        or its elements must support multiplication and addition both with
        themselves and with the elements of `c`.
    
        If `c` is a 1-D array, then `p(x)` will have the same shape as `x`.  If
        `c` is multidimensional, then the shape of the result depends on the
        value of `tensor`. If `tensor` is true the shape will be c.shape[1:] +
        x.shape. If `tensor` is false the shape will be c.shape[1:]. Note that
        scalars have shape (,).
    
        Trailing zeros in the coefficients will be used in the evaluation, so
        they should be avoided if efficiency is a concern.
    
        Parameters
        ----------
        x : array_like, compatible object
            If `x` is a list or tuple, it is converted to an ndarray, otherwise
            it is left unchanged and treated as a scalar. In either case, `x`
            or its elements must support addition and multiplication with
            themselves and with the elements of `c`.
        c : array_like
            Array of coefficients ordered so that the coefficients for terms of
            degree n are contained in c[n]. If `c` is multidimensional the
            remaining indices enumerate multiple polynomials. In the two
            dimensional case the coefficients may be thought of as stored in
            the columns of `c`.
        tensor : boolean, optional
            If True, the shape of the coefficient array is extended with ones
            on the right, one for each dimension of `x`. Scalars have dimension 0
            for this action. The result is that every column of coefficients in
            `c` is evaluated for every element of `x`. If False, `x` is broadcast
            over the columns of `c` for the evaluation.  This keyword is useful
            when `c` is multidimensional. The default value is True.
    
            .. versionadded:: 1.7.0
    
        Returns
        -------
        values : ndarray, algebra_like
            The shape of the return value is described above.
    
        See Also
        --------
        lagval2d, laggrid2d, lagval3d, laggrid3d
    
        Notes
        -----
        The evaluation uses Clenshaw recursion, aka synthetic division.
    
        Examples
        --------
        >>> from numpy.polynomial.laguerre import lagval
        >>> coef = [1,2,3]
        >>> lagval(1, coef)
        -0.5
        >>> lagval([[1,2],[3,4]], coef)
        array([[-0.5, -4. ],
               [-4.5, -2. ]])
    
        
    """
def lagval2d(x, y, c):
    """
    
        Evaluate a 2-D Laguerre series at points (x, y).
    
        This function returns the values:
    
        .. math:: p(x,y) = \\sum_{i,j} c_{i,j} * L_i(x) * L_j(y)
    
        The parameters `x` and `y` are converted to arrays only if they are
        tuples or a lists, otherwise they are treated as a scalars and they
        must have the same shape after conversion. In either case, either `x`
        and `y` or their elements must support multiplication and addition both
        with themselves and with the elements of `c`.
    
        If `c` is a 1-D array a one is implicitly appended to its shape to make
        it 2-D. The shape of the result will be c.shape[2:] + x.shape.
    
        Parameters
        ----------
        x, y : array_like, compatible objects
            The two dimensional series is evaluated at the points `(x, y)`,
            where `x` and `y` must have the same shape. If `x` or `y` is a list
            or tuple, it is first converted to an ndarray, otherwise it is left
            unchanged and if it isn't an ndarray it is treated as a scalar.
        c : array_like
            Array of coefficients ordered so that the coefficient of the term
            of multi-degree i,j is contained in ``c[i,j]``. If `c` has
            dimension greater than two the remaining indices enumerate multiple
            sets of coefficients.
    
        Returns
        -------
        values : ndarray, compatible object
            The values of the two dimensional polynomial at points formed with
            pairs of corresponding values from `x` and `y`.
    
        See Also
        --------
        lagval, laggrid2d, lagval3d, laggrid3d
    
        Notes
        -----
    
        .. versionadded:: 1.7.0
    
        
    """
def lagval3d(x, y, z, c):
    """
    
        Evaluate a 3-D Laguerre series at points (x, y, z).
    
        This function returns the values:
    
        .. math:: p(x,y,z) = \\sum_{i,j,k} c_{i,j,k} * L_i(x) * L_j(y) * L_k(z)
    
        The parameters `x`, `y`, and `z` are converted to arrays only if
        they are tuples or a lists, otherwise they are treated as a scalars and
        they must have the same shape after conversion. In either case, either
        `x`, `y`, and `z` or their elements must support multiplication and
        addition both with themselves and with the elements of `c`.
    
        If `c` has fewer than 3 dimensions, ones are implicitly appended to its
        shape to make it 3-D. The shape of the result will be c.shape[3:] +
        x.shape.
    
        Parameters
        ----------
        x, y, z : array_like, compatible object
            The three dimensional series is evaluated at the points
            `(x, y, z)`, where `x`, `y`, and `z` must have the same shape.  If
            any of `x`, `y`, or `z` is a list or tuple, it is first converted
            to an ndarray, otherwise it is left unchanged and if it isn't an
            ndarray it is  treated as a scalar.
        c : array_like
            Array of coefficients ordered so that the coefficient of the term of
            multi-degree i,j,k is contained in ``c[i,j,k]``. If `c` has dimension
            greater than 3 the remaining indices enumerate multiple sets of
            coefficients.
    
        Returns
        -------
        values : ndarray, compatible object
            The values of the multidimensional polynomial on points formed with
            triples of corresponding values from `x`, `y`, and `z`.
    
        See Also
        --------
        lagval, lagval2d, laggrid2d, laggrid3d
    
        Notes
        -----
    
        .. versionadded:: 1.7.0
    
        
    """
def lagvander(x, deg):
    """
    Pseudo-Vandermonde matrix of given degree.
    
        Returns the pseudo-Vandermonde matrix of degree `deg` and sample points
        `x`. The pseudo-Vandermonde matrix is defined by
    
        .. math:: V[..., i] = L_i(x)
    
        where `0 <= i <= deg`. The leading indices of `V` index the elements of
        `x` and the last index is the degree of the Laguerre polynomial.
    
        If `c` is a 1-D array of coefficients of length `n + 1` and `V` is the
        array ``V = lagvander(x, n)``, then ``np.dot(V, c)`` and
        ``lagval(x, c)`` are the same up to roundoff. This equivalence is
        useful both for least squares fitting and for the evaluation of a large
        number of Laguerre series of the same degree and sample points.
    
        Parameters
        ----------
        x : array_like
            Array of points. The dtype is converted to float64 or complex128
            depending on whether any of the elements are complex. If `x` is
            scalar it is converted to a 1-D array.
        deg : int
            Degree of the resulting matrix.
    
        Returns
        -------
        vander : ndarray
            The pseudo-Vandermonde matrix. The shape of the returned matrix is
            ``x.shape + (deg + 1,)``, where The last index is the degree of the
            corresponding Laguerre polynomial.  The dtype will be the same as
            the converted `x`.
    
        Examples
        --------
        >>> from numpy.polynomial.laguerre import lagvander
        >>> x = np.array([0, 1, 2])
        >>> lagvander(x, 3)
        array([[ 1.        ,  1.        ,  1.        ,  1.        ],
               [ 1.        ,  0.        , -0.5       , -0.66666667],
               [ 1.        , -1.        , -1.        , -0.33333333]])
    
        
    """
def lagvander2d(x, y, deg):
    """
    Pseudo-Vandermonde matrix of given degrees.
    
        Returns the pseudo-Vandermonde matrix of degrees `deg` and sample
        points `(x, y)`. The pseudo-Vandermonde matrix is defined by
    
        .. math:: V[..., (deg[1] + 1)*i + j] = L_i(x) * L_j(y),
    
        where `0 <= i <= deg[0]` and `0 <= j <= deg[1]`. The leading indices of
        `V` index the points `(x, y)` and the last index encodes the degrees of
        the Laguerre polynomials.
    
        If ``V = lagvander2d(x, y, [xdeg, ydeg])``, then the columns of `V`
        correspond to the elements of a 2-D coefficient array `c` of shape
        (xdeg + 1, ydeg + 1) in the order
    
        .. math:: c_{00}, c_{01}, c_{02} ... , c_{10}, c_{11}, c_{12} ...
    
        and ``np.dot(V, c.flat)`` and ``lagval2d(x, y, c)`` will be the same
        up to roundoff. This equivalence is useful both for least squares
        fitting and for the evaluation of a large number of 2-D Laguerre
        series of the same degrees and sample points.
    
        Parameters
        ----------
        x, y : array_like
            Arrays of point coordinates, all of the same shape. The dtypes
            will be converted to either float64 or complex128 depending on
            whether any of the elements are complex. Scalars are converted to
            1-D arrays.
        deg : list of ints
            List of maximum degrees of the form [x_deg, y_deg].
    
        Returns
        -------
        vander2d : ndarray
            The shape of the returned matrix is ``x.shape + (order,)``, where
            :math:`order = (deg[0]+1)*(deg[1]+1)`.  The dtype will be the same
            as the converted `x` and `y`.
    
        See Also
        --------
        lagvander, lagvander3d, lagval2d, lagval3d
    
        Notes
        -----
    
        .. versionadded:: 1.7.0
    
        
    """
def lagvander3d(x, y, z, deg):
    """
    Pseudo-Vandermonde matrix of given degrees.
    
        Returns the pseudo-Vandermonde matrix of degrees `deg` and sample
        points `(x, y, z)`. If `l, m, n` are the given degrees in `x, y, z`,
        then The pseudo-Vandermonde matrix is defined by
    
        .. math:: V[..., (m+1)(n+1)i + (n+1)j + k] = L_i(x)*L_j(y)*L_k(z),
    
        where `0 <= i <= l`, `0 <= j <= m`, and `0 <= j <= n`.  The leading
        indices of `V` index the points `(x, y, z)` and the last index encodes
        the degrees of the Laguerre polynomials.
    
        If ``V = lagvander3d(x, y, z, [xdeg, ydeg, zdeg])``, then the columns
        of `V` correspond to the elements of a 3-D coefficient array `c` of
        shape (xdeg + 1, ydeg + 1, zdeg + 1) in the order
    
        .. math:: c_{000}, c_{001}, c_{002},... , c_{010}, c_{011}, c_{012},...
    
        and  ``np.dot(V, c.flat)`` and ``lagval3d(x, y, z, c)`` will be the
        same up to roundoff. This equivalence is useful both for least squares
        fitting and for the evaluation of a large number of 3-D Laguerre
        series of the same degrees and sample points.
    
        Parameters
        ----------
        x, y, z : array_like
            Arrays of point coordinates, all of the same shape. The dtypes will
            be converted to either float64 or complex128 depending on whether
            any of the elements are complex. Scalars are converted to 1-D
            arrays.
        deg : list of ints
            List of maximum degrees of the form [x_deg, y_deg, z_deg].
    
        Returns
        -------
        vander3d : ndarray
            The shape of the returned matrix is ``x.shape + (order,)``, where
            :math:`order = (deg[0]+1)*(deg[1]+1)*(deg[2]+1)`.  The dtype will
            be the same as the converted `x`, `y`, and `z`.
    
        See Also
        --------
        lagvander, lagvander3d, lagval2d, lagval3d
    
        Notes
        -----
    
        .. versionadded:: 1.7.0
    
        
    """
def lagweight(x):
    """
    Weight function of the Laguerre polynomials.
    
        The weight function is :math:`exp(-x)` and the interval of integration
        is :math:`[0, \\inf]`. The Laguerre polynomials are orthogonal, but not
        normalized, with respect to this weight function.
    
        Parameters
        ----------
        x : array_like
           Values at which the weight function will be computed.
    
        Returns
        -------
        w : ndarray
           The weight function at `x`.
    
        Notes
        -----
    
        .. versionadded:: 1.7.0
    
        
    """
def poly2lag(pol):
    """
    
        poly2lag(pol)
    
        Convert a polynomial to a Laguerre series.
    
        Convert an array representing the coefficients of a polynomial (relative
        to the "standard" basis) ordered from lowest degree to highest, to an
        array of the coefficients of the equivalent Laguerre series, ordered
        from lowest to highest degree.
    
        Parameters
        ----------
        pol : array_like
            1-D array containing the polynomial coefficients
    
        Returns
        -------
        c : ndarray
            1-D array containing the coefficients of the equivalent Laguerre
            series.
    
        See Also
        --------
        lag2poly
    
        Notes
        -----
        The easy way to do conversions between polynomial basis sets
        is to use the convert method of a class instance.
    
        Examples
        --------
        >>> from numpy.polynomial.laguerre import poly2lag
        >>> poly2lag(np.arange(4))
        array([ 23., -63.,  58., -18.])
    
        
    """
lagdomain: numpy.ndarray  # value = array([0, 1])
lagone: numpy.ndarray  # value = array([1])
lagx: numpy.ndarray  # value = array([ 1, -1])
lagzero: numpy.ndarray  # value = array([0])
