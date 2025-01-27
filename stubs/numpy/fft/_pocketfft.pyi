"""

Discrete Fourier Transforms

Routines in this module:

fft(a, n=None, axis=-1, norm="backward")
ifft(a, n=None, axis=-1, norm="backward")
rfft(a, n=None, axis=-1, norm="backward")
irfft(a, n=None, axis=-1, norm="backward")
hfft(a, n=None, axis=-1, norm="backward")
ihfft(a, n=None, axis=-1, norm="backward")
fftn(a, s=None, axes=None, norm="backward")
ifftn(a, s=None, axes=None, norm="backward")
rfftn(a, s=None, axes=None, norm="backward")
irfftn(a, s=None, axes=None, norm="backward")
fft2(a, s=None, axes=(-2,-1), norm="backward")
ifft2(a, s=None, axes=(-2, -1), norm="backward")
rfft2(a, s=None, axes=(-2,-1), norm="backward")
irfft2(a, s=None, axes=(-2, -1), norm="backward")

i = inverse transform
r = transform of purely real data
h = Hermite transform
n = n-dimensional transform
2 = 2-dimensional transform
(Note: 2D routines are just nD routines with different default
behavior.)

"""
from __future__ import annotations
import functools as functools
import numpy
from numpy import asarray
from numpy.core._multiarray_umath import normalize_axis_index
from numpy.core import overrides
import numpy.fft
from numpy.fft import _pocketfft_internal as pfi
from numpy.fft import fft
from numpy.fft import fft2
from numpy.fft import fftn
from numpy.fft import hfft
from numpy.fft import ifft
from numpy.fft import ifft2
from numpy.fft import ifftn
from numpy.fft import ihfft
from numpy.fft import irfft
from numpy.fft import irfft2
from numpy.fft import irfftn
from numpy.fft import rfft
from numpy.fft import rfft2
from numpy.fft import rfftn
from numpy import swapaxes
from numpy import take
from numpy import zeros
__all__: list = ['fft', 'ifft', 'rfft', 'irfft', 'hfft', 'ihfft', 'rfftn', 'irfftn', 'rfft2', 'irfft2', 'fft2', 'ifft2', 'fftn', 'ifftn']
def _cook_nd_args(a, s = None, axes = None, invreal = 0):
    ...
def _fft_dispatcher(a, n = None, axis = None, norm = None):
    ...
def _fftn_dispatcher(a, s = None, axes = None, norm = None):
    ...
def _get_backward_norm(n, norm):
    ...
def _get_forward_norm(n, norm):
    ...
def _raw_fft(a, n, axis, is_real, is_forward, inv_norm):
    ...
def _raw_fftnd(a, s = None, axes = None, function = numpy.fft.fft, norm = None):
    ...
def _swap_direction(norm):
    ...
_SWAP_DIRECTION_MAP: dict = {'backward': 'forward', None: 'forward', 'ortho': 'ortho', 'forward': 'backward'}
array_function_dispatch: functools.partial  # value = functools.partial(<function array_function_dispatch at 0x00000230DBD7F640>, module='numpy.fft')
conjugate: numpy.ufunc  # value = <ufunc 'conjugate'>
sqrt: numpy.ufunc  # value = <ufunc 'sqrt'>
