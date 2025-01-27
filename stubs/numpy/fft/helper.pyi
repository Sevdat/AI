"""

Discrete Fourier Transforms - helper.py

"""
from __future__ import annotations
import numpy
from numpy._utils import set_module
from numpy import arange
from numpy import asarray
from numpy.core.overrides import array_function_dispatch
from numpy import empty
from numpy.fft import fftfreq
from numpy.fft import fftshift
from numpy.fft import ifftshift
from numpy.fft import rfftfreq
from numpy import integer
from numpy import roll
__all__: list = ['fftshift', 'ifftshift', 'fftfreq', 'rfftfreq']
def _fftshift_dispatcher(x, axes = None):
    ...
integer_types: tuple = (int, numpy.integer)
