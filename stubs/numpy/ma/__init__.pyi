"""

=============
Masked Arrays
=============

Arrays sometimes contain invalid or missing data.  When doing operations
on such arrays, we wish to suppress invalid values, which is the purpose masked
arrays fulfill (an example of typical use is given below).

For example, examine the following array:

>>> x = np.array([2, 1, 3, np.nan, 5, 2, 3, np.nan])

When we try to calculate the mean of the data, the result is undetermined:

>>> np.mean(x)
nan

The mean is calculated using roughly ``np.sum(x)/len(x)``, but since
any number added to ``NaN`` [1]_ produces ``NaN``, this doesn't work.  Enter
masked arrays:

>>> m = np.ma.masked_array(x, np.isnan(x))
>>> m
masked_array(data = [2.0 1.0 3.0 -- 5.0 2.0 3.0 --],
      mask = [False False False  True False False False  True],
      fill_value=1e+20)

Here, we construct a masked array that suppress all ``NaN`` values.  We
may now proceed to calculate the mean of the other values:

>>> np.mean(m)
2.6666666666666665

.. [1] Not-a-Number, a floating point value that is the result of an
       invalid operation.

.. moduleauthor:: Pierre Gerard-Marchant
.. moduleauthor:: Jarrod Millman

"""
from __future__ import annotations
import numpy
import numpy._pytesttester
from numpy import amax
from numpy import amin
from numpy import bool_ as MaskType
from numpy import bool_
from numpy import expand_dims
from numpy.ma.core import MAError
from numpy.ma.core import MaskError
from numpy.ma.core import MaskedArray
from numpy.ma.core import MaskedArray as masked_array
from numpy.ma.core._MaskedBinaryOperation import reduce as sometrue
from numpy.ma.core._MaskedBinaryOperation import reduce as alltrue
from numpy.ma.core import allclose
from numpy.ma.core import allequal
from numpy.ma.core import append
from numpy.ma.core import argsort
from numpy.ma.core import array
from numpy.ma.core import asanyarray
from numpy.ma.core import asarray
from numpy.ma.core import choose
from numpy.ma.core import common_fill_value
from numpy.ma.core import compressed
from numpy.ma.core import concatenate
from numpy.ma.core import convolve
from numpy.ma.core import correlate
from numpy.ma.core import default_fill_value
from numpy.ma.core import diag
from numpy.ma.core import diff
from numpy.ma.core import dot
from numpy.ma.core import filled
from numpy.ma.core import fix_invalid
from numpy.ma.core import flatten_mask
from numpy.ma.core import flatten_structured_array
from numpy.ma.core import fromflex
from numpy.ma.core import getdata
from numpy.ma.core import getmask
from numpy.ma.core import getmaskarray
from numpy.ma.core import inner as innerproduct
from numpy.ma.core import inner
from numpy.ma.core import isMaskedArray as isMA
from numpy.ma.core import isMaskedArray
from numpy.ma.core import isMaskedArray as isarray
from numpy.ma.core import is_mask
from numpy.ma.core import is_masked
from numpy.ma.core import left_shift
from numpy.ma.core import make_mask
from numpy.ma.core import make_mask_descr
from numpy.ma.core import make_mask_none
from numpy.ma.core import mask_or
from numpy.ma.core import masked_equal
from numpy.ma.core import masked_greater
from numpy.ma.core import masked_greater_equal
from numpy.ma.core import masked_inside
from numpy.ma.core import masked_invalid
from numpy.ma.core import masked_less
from numpy.ma.core import masked_less_equal
from numpy.ma.core import masked_not_equal
from numpy.ma.core import masked_object
from numpy.ma.core import masked_outside
from numpy.ma.core import masked_values
from numpy.ma.core import masked_where
from numpy.ma.core import max
from numpy.ma.core import maximum_fill_value
from numpy.ma.core import min
from numpy.ma.core import minimum_fill_value
from numpy.ma.core import mvoid
from numpy.ma.core import ndim
from numpy.ma.core import outer as outerproduct
from numpy.ma.core import outer
from numpy.ma.core import power
from numpy.ma.core import ptp
from numpy.ma.core import put
from numpy.ma.core import putmask
from numpy.ma.core import reshape
from numpy.ma.core import resize
from numpy.ma.core import right_shift
from numpy.ma.core import round_ as round
from numpy.ma.core import round_
from numpy.ma.core import set_fill_value
from numpy.ma.core import shape
from numpy.ma.core import size
from numpy.ma.core import sort
from numpy.ma.core import take
from numpy.ma.core import transpose
from numpy.ma.core import where
from numpy.ma.extras import apply_along_axis
from numpy.ma.extras import apply_over_axes
from numpy.ma.extras import average
from numpy.ma.extras import clump_masked
from numpy.ma.extras import clump_unmasked
from numpy.ma.extras import compress_cols
from numpy.ma.extras import compress_nd
from numpy.ma.extras import compress_rowcols
from numpy.ma.extras import compress_rows
from numpy.ma.extras import corrcoef
from numpy.ma.extras import count_masked
from numpy.ma.extras import cov
from numpy.ma.extras import ediff1d
from numpy.ma.extras import flatnotmasked_contiguous
from numpy.ma.extras import flatnotmasked_edges
from numpy.ma.extras import in1d
from numpy.ma.extras import intersect1d
from numpy.ma.extras import isin
from numpy.ma.extras import mask_cols
from numpy.ma.extras import mask_rowcols
from numpy.ma.extras import mask_rows
from numpy.ma.extras import masked_all
from numpy.ma.extras import masked_all_like
from numpy.ma.extras import median
from numpy.ma.extras import ndenumerate
from numpy.ma.extras import notmasked_contiguous
from numpy.ma.extras import notmasked_edges
from numpy.ma.extras import polyfit
from numpy.ma.extras import setdiff1d
from numpy.ma.extras import setxor1d
from numpy.ma.extras import union1d
from numpy.ma.extras import unique
from numpy.ma.extras import vander
from . import core
from . import extras
__all__: list = ['core', 'extras', 'MAError', 'MaskError', 'MaskType', 'MaskedArray', 'abs', 'absolute', 'add', 'all', 'allclose', 'allequal', 'alltrue', 'amax', 'amin', 'angle', 'anom', 'anomalies', 'any', 'append', 'arange', 'arccos', 'arccosh', 'arcsin', 'arcsinh', 'arctan', 'arctan2', 'arctanh', 'argmax', 'argmin', 'argsort', 'around', 'array', 'asanyarray', 'asarray', 'bitwise_and', 'bitwise_or', 'bitwise_xor', 'bool_', 'ceil', 'choose', 'clip', 'common_fill_value', 'compress', 'compressed', 'concatenate', 'conjugate', 'convolve', 'copy', 'correlate', 'cos', 'cosh', 'count', 'cumprod', 'cumsum', 'default_fill_value', 'diag', 'diagonal', 'diff', 'divide', 'empty', 'empty_like', 'equal', 'exp', 'expand_dims', 'fabs', 'filled', 'fix_invalid', 'flatten_mask', 'flatten_structured_array', 'floor', 'floor_divide', 'fmod', 'frombuffer', 'fromflex', 'fromfunction', 'getdata', 'getmask', 'getmaskarray', 'greater', 'greater_equal', 'harden_mask', 'hypot', 'identity', 'ids', 'indices', 'inner', 'innerproduct', 'isMA', 'isMaskedArray', 'is_mask', 'is_masked', 'isarray', 'left_shift', 'less', 'less_equal', 'log', 'log10', 'log2', 'logical_and', 'logical_not', 'logical_or', 'logical_xor', 'make_mask', 'make_mask_descr', 'make_mask_none', 'mask_or', 'masked', 'masked_array', 'masked_equal', 'masked_greater', 'masked_greater_equal', 'masked_inside', 'masked_invalid', 'masked_less', 'masked_less_equal', 'masked_not_equal', 'masked_object', 'masked_outside', 'masked_print_option', 'masked_singleton', 'masked_values', 'masked_where', 'max', 'maximum', 'maximum_fill_value', 'mean', 'min', 'minimum', 'minimum_fill_value', 'mod', 'multiply', 'mvoid', 'ndim', 'negative', 'nomask', 'nonzero', 'not_equal', 'ones', 'ones_like', 'outer', 'outerproduct', 'power', 'prod', 'product', 'ptp', 'put', 'putmask', 'ravel', 'remainder', 'repeat', 'reshape', 'resize', 'right_shift', 'round', 'round_', 'set_fill_value', 'shape', 'sin', 'sinh', 'size', 'soften_mask', 'sometrue', 'sort', 'sqrt', 'squeeze', 'std', 'subtract', 'sum', 'swapaxes', 'take', 'tan', 'tanh', 'trace', 'transpose', 'true_divide', 'var', 'where', 'zeros', 'zeros_like', 'apply_along_axis', 'apply_over_axes', 'atleast_1d', 'atleast_2d', 'atleast_3d', 'average', 'clump_masked', 'clump_unmasked', 'column_stack', 'compress_cols', 'compress_nd', 'compress_rowcols', 'compress_rows', 'count_masked', 'corrcoef', 'cov', 'diagflat', 'dot', 'dstack', 'ediff1d', 'flatnotmasked_contiguous', 'flatnotmasked_edges', 'hsplit', 'hstack', 'isin', 'in1d', 'intersect1d', 'mask_cols', 'mask_rowcols', 'mask_rows', 'masked_all', 'masked_all_like', 'median', 'mr_', 'ndenumerate', 'notmasked_contiguous', 'notmasked_edges', 'polyfit', 'row_stack', 'setdiff1d', 'setxor1d', 'stack', 'unique', 'union1d', 'vander', 'vstack']
abs: core._MaskedUnaryOperation  # value = <numpy.ma.core._MaskedUnaryOperation object>
absolute: core._MaskedUnaryOperation  # value = <numpy.ma.core._MaskedUnaryOperation object>
add: core._MaskedBinaryOperation  # value = <numpy.ma.core._MaskedBinaryOperation object>
all: core._frommethod  # value = <numpy.ma.core._frommethod object>
angle: core._MaskedUnaryOperation  # value = <numpy.ma.core._MaskedUnaryOperation object>
anom: core._frommethod  # value = <numpy.ma.core._frommethod object>
anomalies: core._frommethod  # value = <numpy.ma.core._frommethod object>
any: core._frommethod  # value = <numpy.ma.core._frommethod object>
arange: core._convert2ma  # value = <numpy.ma.core._convert2ma object>
arccos: core._MaskedUnaryOperation  # value = <numpy.ma.core._MaskedUnaryOperation object>
arccosh: core._MaskedUnaryOperation  # value = <numpy.ma.core._MaskedUnaryOperation object>
arcsin: core._MaskedUnaryOperation  # value = <numpy.ma.core._MaskedUnaryOperation object>
arcsinh: core._MaskedUnaryOperation  # value = <numpy.ma.core._MaskedUnaryOperation object>
arctan: core._MaskedUnaryOperation  # value = <numpy.ma.core._MaskedUnaryOperation object>
arctan2: core._MaskedBinaryOperation  # value = <numpy.ma.core._MaskedBinaryOperation object>
arctanh: core._MaskedUnaryOperation  # value = <numpy.ma.core._MaskedUnaryOperation object>
argmax: core._frommethod  # value = <numpy.ma.core._frommethod object>
argmin: core._frommethod  # value = <numpy.ma.core._frommethod object>
around: core._MaskedUnaryOperation  # value = <numpy.ma.core._MaskedUnaryOperation object>
atleast_1d: extras._fromnxfunction_allargs  # value = <numpy.ma.extras._fromnxfunction_allargs object>
atleast_2d: extras._fromnxfunction_allargs  # value = <numpy.ma.extras._fromnxfunction_allargs object>
atleast_3d: extras._fromnxfunction_allargs  # value = <numpy.ma.extras._fromnxfunction_allargs object>
bitwise_and: core._MaskedBinaryOperation  # value = <numpy.ma.core._MaskedBinaryOperation object>
bitwise_or: core._MaskedBinaryOperation  # value = <numpy.ma.core._MaskedBinaryOperation object>
bitwise_xor: core._MaskedBinaryOperation  # value = <numpy.ma.core._MaskedBinaryOperation object>
ceil: core._MaskedUnaryOperation  # value = <numpy.ma.core._MaskedUnaryOperation object>
clip: core._convert2ma  # value = <numpy.ma.core._convert2ma object>
column_stack: extras._fromnxfunction_seq  # value = <numpy.ma.extras._fromnxfunction_seq object>
compress: core._frommethod  # value = <numpy.ma.core._frommethod object>
conjugate: core._MaskedUnaryOperation  # value = <numpy.ma.core._MaskedUnaryOperation object>
copy: core._frommethod  # value = <numpy.ma.core._frommethod object>
cos: core._MaskedUnaryOperation  # value = <numpy.ma.core._MaskedUnaryOperation object>
cosh: core._MaskedUnaryOperation  # value = <numpy.ma.core._MaskedUnaryOperation object>
count: core._frommethod  # value = <numpy.ma.core._frommethod object>
cumprod: core._frommethod  # value = <numpy.ma.core._frommethod object>
cumsum: core._frommethod  # value = <numpy.ma.core._frommethod object>
diagflat: extras._fromnxfunction_single  # value = <numpy.ma.extras._fromnxfunction_single object>
diagonal: core._frommethod  # value = <numpy.ma.core._frommethod object>
divide: core._DomainedBinaryOperation  # value = <numpy.ma.core._DomainedBinaryOperation object>
dstack: extras._fromnxfunction_seq  # value = <numpy.ma.extras._fromnxfunction_seq object>
empty: core._convert2ma  # value = <numpy.ma.core._convert2ma object>
empty_like: core._convert2ma  # value = <numpy.ma.core._convert2ma object>
equal: core._MaskedBinaryOperation  # value = <numpy.ma.core._MaskedBinaryOperation object>
exp: core._MaskedUnaryOperation  # value = <numpy.ma.core._MaskedUnaryOperation object>
fabs: core._MaskedUnaryOperation  # value = <numpy.ma.core._MaskedUnaryOperation object>
floor: core._MaskedUnaryOperation  # value = <numpy.ma.core._MaskedUnaryOperation object>
floor_divide: core._DomainedBinaryOperation  # value = <numpy.ma.core._DomainedBinaryOperation object>
fmod: core._DomainedBinaryOperation  # value = <numpy.ma.core._DomainedBinaryOperation object>
frombuffer: core._convert2ma  # value = <numpy.ma.core._convert2ma object>
fromfunction: core._convert2ma  # value = <numpy.ma.core._convert2ma object>
greater: core._MaskedBinaryOperation  # value = <numpy.ma.core._MaskedBinaryOperation object>
greater_equal: core._MaskedBinaryOperation  # value = <numpy.ma.core._MaskedBinaryOperation object>
harden_mask: core._frommethod  # value = <numpy.ma.core._frommethod object>
hsplit: extras._fromnxfunction_single  # value = <numpy.ma.extras._fromnxfunction_single object>
hstack: extras._fromnxfunction_seq  # value = <numpy.ma.extras._fromnxfunction_seq object>
hypot: core._MaskedBinaryOperation  # value = <numpy.ma.core._MaskedBinaryOperation object>
identity: core._convert2ma  # value = <numpy.ma.core._convert2ma object>
ids: core._frommethod  # value = <numpy.ma.core._frommethod object>
indices: core._convert2ma  # value = <numpy.ma.core._convert2ma object>
less: core._MaskedBinaryOperation  # value = <numpy.ma.core._MaskedBinaryOperation object>
less_equal: core._MaskedBinaryOperation  # value = <numpy.ma.core._MaskedBinaryOperation object>
log: core._MaskedUnaryOperation  # value = <numpy.ma.core._MaskedUnaryOperation object>
log10: core._MaskedUnaryOperation  # value = <numpy.ma.core._MaskedUnaryOperation object>
log2: core._MaskedUnaryOperation  # value = <numpy.ma.core._MaskedUnaryOperation object>
logical_and: core._MaskedBinaryOperation  # value = <numpy.ma.core._MaskedBinaryOperation object>
logical_not: core._MaskedUnaryOperation  # value = <numpy.ma.core._MaskedUnaryOperation object>
logical_or: core._MaskedBinaryOperation  # value = <numpy.ma.core._MaskedBinaryOperation object>
logical_xor: core._MaskedBinaryOperation  # value = <numpy.ma.core._MaskedBinaryOperation object>
masked: core.MaskedConstant  # value = masked
masked_print_option: core._MaskedPrintOption  # value = --
masked_singleton: core.MaskedConstant  # value = masked
maximum: core._extrema_operation  # value = <numpy.ma.core._extrema_operation object>
mean: core._frommethod  # value = <numpy.ma.core._frommethod object>
minimum: core._extrema_operation  # value = <numpy.ma.core._extrema_operation object>
mod: core._DomainedBinaryOperation  # value = <numpy.ma.core._DomainedBinaryOperation object>
mr_: extras.mr_class  # value = <numpy.ma.extras.mr_class object>
multiply: core._MaskedBinaryOperation  # value = <numpy.ma.core._MaskedBinaryOperation object>
negative: core._MaskedUnaryOperation  # value = <numpy.ma.core._MaskedUnaryOperation object>
nomask: numpy.bool_  # value = False
nonzero: core._frommethod  # value = <numpy.ma.core._frommethod object>
not_equal: core._MaskedBinaryOperation  # value = <numpy.ma.core._MaskedBinaryOperation object>
ones: core._convert2ma  # value = <numpy.ma.core._convert2ma object>
ones_like: core._convert2ma  # value = <numpy.ma.core._convert2ma object>
prod: core._frommethod  # value = <numpy.ma.core._frommethod object>
product: core._frommethod  # value = <numpy.ma.core._frommethod object>
ravel: core._frommethod  # value = <numpy.ma.core._frommethod object>
remainder: core._DomainedBinaryOperation  # value = <numpy.ma.core._DomainedBinaryOperation object>
repeat: core._frommethod  # value = <numpy.ma.core._frommethod object>
row_stack: extras._fromnxfunction_seq  # value = <numpy.ma.extras._fromnxfunction_seq object>
sin: core._MaskedUnaryOperation  # value = <numpy.ma.core._MaskedUnaryOperation object>
sinh: core._MaskedUnaryOperation  # value = <numpy.ma.core._MaskedUnaryOperation object>
soften_mask: core._frommethod  # value = <numpy.ma.core._frommethod object>
sqrt: core._MaskedUnaryOperation  # value = <numpy.ma.core._MaskedUnaryOperation object>
squeeze: core._convert2ma  # value = <numpy.ma.core._convert2ma object>
stack: extras._fromnxfunction_seq  # value = <numpy.ma.extras._fromnxfunction_seq object>
std: core._frommethod  # value = <numpy.ma.core._frommethod object>
subtract: core._MaskedBinaryOperation  # value = <numpy.ma.core._MaskedBinaryOperation object>
sum: core._frommethod  # value = <numpy.ma.core._frommethod object>
swapaxes: core._frommethod  # value = <numpy.ma.core._frommethod object>
tan: core._MaskedUnaryOperation  # value = <numpy.ma.core._MaskedUnaryOperation object>
tanh: core._MaskedUnaryOperation  # value = <numpy.ma.core._MaskedUnaryOperation object>
test: numpy._pytesttester.PytestTester  # value = <numpy._pytesttester.PytestTester object>
trace: core._frommethod  # value = <numpy.ma.core._frommethod object>
true_divide: core._DomainedBinaryOperation  # value = <numpy.ma.core._DomainedBinaryOperation object>
var: core._frommethod  # value = <numpy.ma.core._frommethod object>
vstack: extras._fromnxfunction_seq  # value = <numpy.ma.extras._fromnxfunction_seq object>
zeros: core._convert2ma  # value = <numpy.ma.core._convert2ma object>
zeros_like: core._convert2ma  # value = <numpy.ma.core._convert2ma object>
