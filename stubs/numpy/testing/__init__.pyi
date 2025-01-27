"""
Common test support for all numpy test scripts.

This single module should provide all the common functionality for numpy tests
in a single location, so that test scripts can just import it and work right
away.

"""
from __future__ import annotations
import numpy._pytesttester
from numpy.testing._private import extbuild
from numpy.testing._private.utils import IgnoreException
from numpy.testing._private.utils import KnownFailureException
from numpy.testing._private.utils import _assert_valid_refcount
from numpy.testing._private.utils import _gen_alignment_data
from numpy.testing._private.utils import assert_
from numpy.testing._private.utils import assert_allclose
from numpy.testing._private.utils import assert_almost_equal
from numpy.testing._private.utils import assert_approx_equal
from numpy.testing._private.utils import assert_array_almost_equal
from numpy.testing._private.utils import assert_array_almost_equal_nulp
from numpy.testing._private.utils import assert_array_compare
from numpy.testing._private.utils import assert_array_equal
from numpy.testing._private.utils import assert_array_less
from numpy.testing._private.utils import assert_array_max_ulp
from numpy.testing._private.utils import assert_equal
from numpy.testing._private.utils import assert_no_gc_cycles
from numpy.testing._private.utils import assert_no_warnings
from numpy.testing._private.utils import assert_raises
from numpy.testing._private.utils import assert_raises_regex
from numpy.testing._private.utils import assert_string_equal
from numpy.testing._private.utils import assert_warns
from numpy.testing._private.utils import break_cycles
from numpy.testing._private.utils import build_err_msg
from numpy.testing._private.utils import clear_and_catch_warnings
from numpy.testing._private.utils import decorate_methods
from numpy.testing._private.utils import jiffies
from numpy.testing._private.utils import measure
from numpy.testing._private.utils import memusage
from numpy.testing._private.utils import print_assert_equal
from numpy.testing._private.utils import rundocs
from numpy.testing._private.utils import runstring
from numpy.testing._private.utils import suppress_warnings
from numpy.testing._private.utils import tempdir
from numpy.testing._private.utils import temppath
from unittest.case import SkipTest
from unittest.case import TestCase
from . import _private
from . import overrides
__all__: list = ['assert_equal', 'assert_almost_equal', 'assert_approx_equal', 'assert_array_equal', 'assert_array_less', 'assert_string_equal', 'assert_array_almost_equal', 'assert_raises', 'build_err_msg', 'decorate_methods', 'jiffies', 'memusage', 'print_assert_equal', 'rundocs', 'runstring', 'verbose', 'measure', 'assert_', 'assert_array_almost_equal_nulp', 'assert_raises_regex', 'assert_array_max_ulp', 'assert_warns', 'assert_no_warnings', 'assert_allclose', 'IgnoreException', 'clear_and_catch_warnings', 'SkipTest', 'KnownFailureException', 'temppath', 'tempdir', 'IS_PYPY', 'HAS_REFCOUNT', 'IS_WASM', 'suppress_warnings', 'assert_array_compare', 'assert_no_gc_cycles', 'break_cycles', 'HAS_LAPACK64', 'IS_PYSTON', '_OLD_PROMOTION', 'IS_MUSL', '_SUPPORTS_SVE', 'TestCase', 'overrides']
HAS_LAPACK64: bool = True
HAS_REFCOUNT: bool = True
IS_MUSL: bool = False
IS_PYPY: bool = False
IS_PYSTON: bool = False
IS_WASM: bool = False
_SUPPORTS_SVE: bool = False
test: numpy._pytesttester.PytestTester  # value = <numpy._pytesttester.PytestTester object>
verbose: int = 0
