"""
Tools for testing implementations of __array_function__ and ufunc overrides


"""
from __future__ import annotations
import numpy
from numpy.core import umath as _umath
import numpy.fft
import numpy.lib.scimath
import numpy.lib.stride_tricks
import numpy.linalg
from numpy import ufunc as _ufunc
__all__ = ['allows_array_function_override', 'allows_array_ufunc_override', 'get_overridable_numpy_array_functions', 'get_overridable_numpy_ufuncs']
def allows_array_function_override(func):
    """
    Determine if a Numpy function can be overridden via `__array_function__`
    
        Parameters
        ----------
        func : callable
            Function that may be overridable via `__array_function__`
    
        Returns
        -------
        bool
            `True` if `func` is a function in the Numpy API that is
            overridable via `__array_function__` and `False` otherwise.
        
    """
def allows_array_ufunc_override(func):
    """
    Determine if a function can be overridden via `__array_ufunc__`
    
        Parameters
        ----------
        func : callable
            Function that may be overridable via `__array_ufunc__`
    
        Returns
        -------
        bool
            `True` if `func` is overridable via `__array_ufunc__` and
            `False` otherwise.
    
        Notes
        -----
        This function is equivalent to ``isinstance(func, np.ufunc)`` and
        will work correctly for ufuncs defined outside of Numpy.
    
        
    """
def get_overridable_numpy_array_functions():
    """
    List all numpy functions overridable via `__array_function__`
    
        Parameters
        ----------
        None
    
        Returns
        -------
        set
            A set containing all functions in the public numpy API that are
            overridable via `__array_function__`.
    
        
    """
def get_overridable_numpy_ufuncs():
    """
    List all numpy ufuncs overridable via `__array_ufunc__`
    
        Parameters
        ----------
        None
    
        Returns
        -------
        set
            A set containing all overridable ufuncs in the public numpy API.
        
    """
_array_functions: set = {numpy.percentile, numpy.char.lower, numpy.quantile, numpy.char.lstrip, numpy.poly, numpy.trapz, numpy.char.partition, numpy.roots, numpy.save, numpy.char.replace, numpy.delete, numpy.polyint, numpy.char.rfind, numpy.insert, numpy.polyder, numpy.append, numpy.char.rindex, numpy.polyfit, numpy.digitize, numpy.polyval, numpy.char.rjust, numpy.lib.stride_tricks.sliding_window_view, numpy.polyadd, numpy.char.rpartition, numpy.polysub, numpy.char.rsplit, numpy.polymul, numpy.broadcast_to, numpy.asfarray, numpy.polydiv, numpy.char.rstrip, numpy.broadcast_arrays, numpy.char.split, numpy.pad, numpy.real, numpy.fliplr, numpy.fill_diagonal, numpy.fix, numpy.char.splitlines, numpy.flipud, numpy.diag_indices_from, numpy.char.startswith, numpy.isposinf, numpy.eye, numpy.char.strip, numpy.isneginf, numpy.diag, numpy.char.swapcase, numpy.diagflat, numpy.char.title, numpy.tri, numpy.atleast_1d, numpy.char.translate, numpy.imag, numpy.tril, numpy.char.equal, numpy.char.upper, numpy.iscomplex, numpy.triu, numpy.char.not_equal, numpy.char.zfill, numpy.vander, numpy.isreal, numpy.char.greater_equal, numpy.histogram2d, numpy.char.isnumeric, numpy.char.less_equal, numpy.tril_indices_from, numpy.iscomplexobj, numpy.char.isdecimal, numpy.triu_indices_from, numpy.char.greater, numpy.isrealobj, numpy.char.less, numpy.nan_to_num, numpy.char.str_len, numpy.real_if_close, numpy.char.add, numpy.common_type, numpy.char.multiply, numpy.nanmin, numpy.lib.scimath.sqrt, numpy.char.mod, numpy.lib.scimath.log, numpy.nanargmax, numpy.lib.scimath.log10, numpy.nanmax, numpy.char.capitalize, numpy.nanargmin, numpy.lib.scimath.logn, numpy.char.center, numpy.lib.scimath.log2, numpy.char.count, numpy.nansum, numpy.lib.scimath.power, numpy.char.decode, numpy.nanprod, numpy.concatenate, numpy.lib.scimath.arccos, numpy.char.encode, numpy.nancumsum, numpy.einsum_path, numpy.lib.scimath.arcsin, numpy.char.endswith, numpy.nancumprod, numpy.lib.scimath.arctanh, numpy.nanmean, numpy.char.expandtabs, numpy.einsum, numpy.fft.fft, numpy.nanmedian, numpy.char.find, numpy.char.index, numpy.nanpercentile, numpy.fft.ifft, numpy.char.isalnum, numpy.nanquantile, numpy.nanvar, numpy.fft.rfft, numpy.char.isalpha, numpy.inner, numpy.nanstd, numpy.fft.irfft, numpy.char.isdigit, numpy.fft.hfft, numpy.where, numpy.char.islower, numpy.fft.ihfft, numpy.lexsort, numpy.char.isspace, numpy.take, numpy.char.istitle, numpy.fft.fftn, numpy.can_cast, numpy.fft.ifftn, numpy.char.isupper, numpy.min_scalar_type, numpy.fft.fft2, numpy.result_type, numpy.char.join, numpy.fft.ifft2, numpy.char.ljust, numpy.nonzero, numpy.dot, numpy.flatnonzero, numpy.fft.rfftn, numpy.correlate, numpy.fft.rfft2, numpy.vdot, numpy.shape, numpy.fft.irfftn, numpy.convolve, numpy.bincount, numpy.compress, numpy.fft.irfft2, numpy.outer, numpy.ravel_multi_index, numpy.clip, numpy.unravel_index, numpy.tensordot, numpy.sum, numpy.roll, numpy.copyto, numpy.ix_, numpy.any, numpy.rollaxis, numpy.reshape, numpy.putmask, numpy.moveaxis, numpy.all, numpy.linalg.tensorsolve, numpy.packbits, numpy.cross, numpy.fft.fftshift, numpy.fromfunction, numpy.unpackbits, numpy.cumsum, numpy.identity, numpy.take_along_axis, numpy.fft.ifftshift, numpy.ptp, numpy.shares_memory, numpy.allclose, numpy.linalg.solve, numpy.isclose, numpy.put_along_axis, numpy.max, numpy.may_share_memory, numpy.array2string, numpy.apply_along_axis, numpy.linspace, numpy.is_busday, numpy.array_equal, numpy.amax, numpy.apply_over_axes, numpy.array_equiv, numpy.expand_dims, numpy.busday_offset, numpy.logspace, numpy.min, numpy.column_stack, numpy.geomspace, numpy.busday_count, numpy.linalg.tensorinv, numpy.dstack, numpy.amin, numpy.empty_like, numpy.array_split, numpy.datetime_as_string, numpy.split, numpy.prod, numpy.hsplit, numpy.vsplit, numpy.cumprod, numpy.linalg.inv, numpy.dsplit, numpy.array_repr, numpy.linalg.matrix_power, numpy.ndim, numpy.kron, numpy.linalg.cholesky, numpy.choose, numpy.tile, numpy.size, numpy.linalg.qr, numpy.histogram_bin_edges, numpy.array_str, numpy.linalg.eigvals, numpy.round, numpy.repeat, numpy.atleast_2d, numpy.histogram, numpy.linalg.eigvalsh, numpy.histogramdd, numpy.around, numpy.linalg.eig, numpy.put, numpy.linalg.eigh, numpy.mean, numpy.require, numpy.linalg.svd, numpy.swapaxes, numpy.meshgrid, numpy.linalg.cond, numpy.rot90, numpy.std, numpy.transpose, numpy.flip, numpy.linalg.matrix_rank, numpy.linalg.slogdet, numpy.average, numpy.linalg.pinv, numpy.var, numpy.ediff1d, numpy.partition, numpy.unique, numpy.intersect1d, numpy.setxor1d, numpy.piecewise, numpy.in1d, numpy.isin, numpy.union1d, numpy.round_, numpy.setdiff1d, numpy.argpartition, numpy.select, numpy.linalg.det, numpy.product, numpy.linalg.lstsq, numpy.copy, numpy.sort, numpy.gradient, numpy.cumproduct, numpy.linalg.norm, numpy.linalg.multi_dot, numpy.argsort, numpy.sometrue, numpy.diff, numpy.interp, numpy.alltrue, numpy.argmax, numpy.angle, numpy.unwrap, numpy.atleast_3d, numpy.argmin, numpy.sort_complex, numpy.vstack, numpy.trim_zeros, numpy.hstack, numpy.searchsorted, numpy.savez, numpy.extract, numpy.argwhere, numpy.stack, numpy.resize, numpy.savez_compressed, numpy.place, numpy.loadtxt, numpy.block, numpy.squeeze, numpy.cov, numpy.savetxt, numpy.ones_like, numpy.genfromtxt, numpy.corrcoef, numpy.diagonal, numpy.full, numpy.i0, numpy.trace, numpy.full_like, numpy.sinc, numpy.count_nonzero, numpy.ravel, numpy.msort, numpy.ones, numpy.zeros_like, numpy.median}
