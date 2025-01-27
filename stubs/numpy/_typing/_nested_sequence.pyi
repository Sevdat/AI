"""
A module containing the `_NestedSequence` protocol.
"""
from __future__ import annotations
from collections.abc import Iterator
import typing
from typing import Protocol
from typing import TypeVar
from typing import runtime_checkable
__all__: list = ['_NestedSequence']
class _NestedSequence(typing.Protocol):
    """
    A protocol for representing nested sequences.
    
        Warning
        -------
        `_NestedSequence` currently does not work in combination with typevars,
        *e.g.* ``def func(a: _NestedSequnce[T]) -> T: ...``.
    
        See Also
        --------
        collections.abc.Sequence
            ABCs for read-only and mutable :term:`sequences`.
    
        Examples
        --------
        .. code-block:: python
    
            >>> from __future__ import annotations
    
            >>> from typing import TYPE_CHECKING
            >>> import numpy as np
            >>> from numpy._typing import _NestedSequence
    
            >>> def get_dtype(seq: _NestedSequence[float]) -> np.dtype[np.float64]:
            ...     return np.asarray(seq).dtype
    
            >>> a = get_dtype([1.0])
            >>> b = get_dtype([[1.0]])
            >>> c = get_dtype([[[1.0]]])
            >>> d = get_dtype([[[[1.0]]]])
    
            >>> if TYPE_CHECKING:
            ...     reveal_locals()
            ...     # note: Revealed local types are:
            ...     # note:     a: numpy.dtype[numpy.floating[numpy._typing._64Bit]]
            ...     # note:     b: numpy.dtype[numpy.floating[numpy._typing._64Bit]]
            ...     # note:     c: numpy.dtype[numpy.floating[numpy._typing._64Bit]]
            ...     # note:     d: numpy.dtype[numpy.floating[numpy._typing._64Bit]]
    
        
    """
    __abstractmethods__: typing.ClassVar[frozenset]  # value = frozenset()
    __orig_bases__: typing.ClassVar[tuple]  # value = (typing.Protocol[+_T_co])
    __parameters__: typing.ClassVar[tuple]  # value = (+_T_co)
    _abc_impl: typing.ClassVar[_abc._abc_data]  # value = <_abc._abc_data object>
    _is_protocol: typing.ClassVar[bool] = True
    _is_runtime_protocol: typing.ClassVar[bool] = True
    @staticmethod
    def __subclasshook__(other):
        ...
    def __contains__(self, x: typing.Any) -> bool:
        """
        Implement ``x in self``.
        """
    def __getitem__(self, index: int) -> _T_co | _NestedSequence[_T_co]:
        """
        Implement ``self[x]``.
        """
    def __init__(self, *args, **kwargs):
        ...
    def __iter__(self) -> typing.Iterator[_T_co | _NestedSequence[_T_co]]:
        """
        Implement ``iter(self)``.
        """
    def __len__(self) -> int:
        """
        Implement ``len(self)``.
        """
    def __reversed__(self) -> typing.Iterator[_T_co | _NestedSequence[_T_co]]:
        """
        Implement ``reversed(self)``.
        """
    def count(self, value: typing.Any) -> int:
        """
        Return the number of occurrences of `value`.
        """
    def index(self, value: typing.Any) -> int:
        """
        Return the first index of `value`.
        """
_T_co: typing.TypeVar  # value = +_T_co
