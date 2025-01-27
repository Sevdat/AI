from __future__ import annotations
from collections.abc import Callable
from collections.abc import Collection
from collections.abc import Sequence
from numpy._typing._nested_sequence import _NestedSequence
from numpy import bool_
from numpy import bytes_
from numpy import complexfloating
from numpy import datetime64
from numpy import dtype
from numpy import floating
from numpy import generic
from numpy import integer
from numpy import ndarray
from numpy import ndarray as NDArray
from numpy import number
from numpy import object_
from numpy import str_
from numpy import timedelta64
from numpy import unsignedinteger
from numpy import void
import sys as sys
import typing
from typing import Protocol
from typing import TypeVar
from typing import runtime_checkable
__all__ = ['ArrayLike', 'Callable', 'Collection', 'NDArray', 'Protocol', 'Sequence', 'TypeVar', 'bool_', 'bytes_', 'complexfloating', 'datetime64', 'dtype', 'floating', 'generic', 'integer', 'ndarray', 'number', 'object_', 'runtime_checkable', 'str_', 'sys', 'timedelta64', 'unsignedinteger', 'void']
class _SupportsArray(typing.Protocol):
    __abstractmethods__: typing.ClassVar[frozenset]  # value = frozenset()
    __orig_bases__: typing.ClassVar[tuple]  # value = (typing.Protocol[+_DType_co])
    __parameters__: typing.ClassVar[tuple]  # value = (+_DType_co)
    _abc_impl: typing.ClassVar[_abc._abc_data]  # value = <_abc._abc_data object>
    _is_protocol: typing.ClassVar[bool] = True
    _is_runtime_protocol: typing.ClassVar[bool] = True
    @staticmethod
    def __subclasshook__(other):
        ...
    def __array__(self) -> ndarray[typing.Any, _DType_co]:
        ...
    def __init__(self, *args, **kwargs):
        ...
class _SupportsArrayFunc(typing.Protocol):
    """
    A protocol class representing `~class.__array_function__`.
    """
    __abstractmethods__: typing.ClassVar[frozenset]  # value = frozenset()
    __parameters__: typing.ClassVar[tuple] = tuple()
    _abc_impl: typing.ClassVar[_abc._abc_data]  # value = <_abc._abc_data object>
    _is_protocol: typing.ClassVar[bool] = True
    _is_runtime_protocol: typing.ClassVar[bool] = True
    @staticmethod
    def __subclasshook__(other):
        ...
    def __array_function__(self, func: typing.Callable[..., typing.Any], types: Collection[type[typing.Any]], args: tuple[typing.Any, ...], kwargs: dict[str, typing.Any]) -> typing.Any:
        ...
    def __init__(self, *args, **kwargs):
        ...
class _UnknownType:
    pass
ArrayLike: typing._UnionGenericAlias  # value = typing.Union[numpy._typing._array_like._SupportsArray[numpy.dtype[typing.Any]], numpy._typing._nested_sequence._NestedSequence[numpy._typing._array_like._SupportsArray[numpy.dtype[typing.Any]]], bool, int, float, complex, str, bytes, numpy._typing._nested_sequence._NestedSequence[typing.Union[bool, int, float, complex, str, bytes]]]
_ArrayLike: typing._UnionGenericAlias  # value = typing.Union[numpy._typing._array_like._SupportsArray[numpy.dtype[~_ScalarType]], numpy._typing._nested_sequence._NestedSequence[numpy._typing._array_like._SupportsArray[numpy.dtype[~_ScalarType]]]]
_ArrayLikeBool_co: typing._UnionGenericAlias  # value = typing.Union[numpy._typing._array_like._SupportsArray[numpy.dtype[numpy.bool_]], numpy._typing._nested_sequence._NestedSequence[numpy._typing._array_like._SupportsArray[numpy.dtype[numpy.bool_]]], bool, numpy._typing._nested_sequence._NestedSequence[bool]]
_ArrayLikeBytes_co: typing._UnionGenericAlias  # value = typing.Union[numpy._typing._array_like._SupportsArray[numpy.dtype[numpy.bytes_]], numpy._typing._nested_sequence._NestedSequence[numpy._typing._array_like._SupportsArray[numpy.dtype[numpy.bytes_]]], bytes, numpy._typing._nested_sequence._NestedSequence[bytes]]
_ArrayLikeComplex_co: typing._UnionGenericAlias  # value = typing.Union[numpy._typing._array_like._SupportsArray[numpy.dtype[typing.Union[numpy.bool_, numpy.integer[typing.Any], numpy.floating[typing.Any], numpy.complexfloating[typing.Any, typing.Any]]]], numpy._typing._nested_sequence._NestedSequence[numpy._typing._array_like._SupportsArray[numpy.dtype[typing.Union[numpy.bool_, numpy.integer[typing.Any], numpy.floating[typing.Any], numpy.complexfloating[typing.Any, typing.Any]]]]], bool, int, float, complex, numpy._typing._nested_sequence._NestedSequence[typing.Union[bool, int, float, complex]]]
_ArrayLikeDT64_co: typing._UnionGenericAlias  # value = typing.Union[numpy._typing._array_like._SupportsArray[numpy.dtype[numpy.datetime64]], numpy._typing._nested_sequence._NestedSequence[numpy._typing._array_like._SupportsArray[numpy.dtype[numpy.datetime64]]]]
_ArrayLikeFloat_co: typing._UnionGenericAlias  # value = typing.Union[numpy._typing._array_like._SupportsArray[numpy.dtype[typing.Union[numpy.bool_, numpy.integer[typing.Any], numpy.floating[typing.Any]]]], numpy._typing._nested_sequence._NestedSequence[numpy._typing._array_like._SupportsArray[numpy.dtype[typing.Union[numpy.bool_, numpy.integer[typing.Any], numpy.floating[typing.Any]]]]], bool, int, float, numpy._typing._nested_sequence._NestedSequence[typing.Union[bool, int, float]]]
_ArrayLikeInt: typing._UnionGenericAlias  # value = typing.Union[numpy._typing._array_like._SupportsArray[numpy.dtype[numpy.integer[typing.Any]]], numpy._typing._nested_sequence._NestedSequence[numpy._typing._array_like._SupportsArray[numpy.dtype[numpy.integer[typing.Any]]]], int, numpy._typing._nested_sequence._NestedSequence[int]]
_ArrayLikeInt_co: typing._UnionGenericAlias  # value = typing.Union[numpy._typing._array_like._SupportsArray[numpy.dtype[typing.Union[numpy.bool_, numpy.integer[typing.Any]]]], numpy._typing._nested_sequence._NestedSequence[numpy._typing._array_like._SupportsArray[numpy.dtype[typing.Union[numpy.bool_, numpy.integer[typing.Any]]]]], bool, int, numpy._typing._nested_sequence._NestedSequence[typing.Union[bool, int]]]
_ArrayLikeNumber_co: typing._UnionGenericAlias  # value = typing.Union[numpy._typing._array_like._SupportsArray[numpy.dtype[typing.Union[numpy.bool_, numpy.number[typing.Any]]]], numpy._typing._nested_sequence._NestedSequence[numpy._typing._array_like._SupportsArray[numpy.dtype[typing.Union[numpy.bool_, numpy.number[typing.Any]]]]], bool, int, float, complex, numpy._typing._nested_sequence._NestedSequence[typing.Union[bool, int, float, complex]]]
_ArrayLikeObject_co: typing._UnionGenericAlias  # value = typing.Union[numpy._typing._array_like._SupportsArray[numpy.dtype[numpy.object_]], numpy._typing._nested_sequence._NestedSequence[numpy._typing._array_like._SupportsArray[numpy.dtype[numpy.object_]]]]
_ArrayLikeStr_co: typing._UnionGenericAlias  # value = typing.Union[numpy._typing._array_like._SupportsArray[numpy.dtype[numpy.str_]], numpy._typing._nested_sequence._NestedSequence[numpy._typing._array_like._SupportsArray[numpy.dtype[numpy.str_]]], str, numpy._typing._nested_sequence._NestedSequence[str]]
_ArrayLikeTD64_co: typing._UnionGenericAlias  # value = typing.Union[numpy._typing._array_like._SupportsArray[numpy.dtype[typing.Union[numpy.bool_, numpy.integer[typing.Any], numpy.timedelta64]]], numpy._typing._nested_sequence._NestedSequence[numpy._typing._array_like._SupportsArray[numpy.dtype[typing.Union[numpy.bool_, numpy.integer[typing.Any], numpy.timedelta64]]]], bool, int, numpy._typing._nested_sequence._NestedSequence[typing.Union[bool, int]]]
_ArrayLikeUInt_co: typing._UnionGenericAlias  # value = typing.Union[numpy._typing._array_like._SupportsArray[numpy.dtype[typing.Union[numpy.bool_, numpy.unsignedinteger[typing.Any]]]], numpy._typing._nested_sequence._NestedSequence[numpy._typing._array_like._SupportsArray[numpy.dtype[typing.Union[numpy.bool_, numpy.unsignedinteger[typing.Any]]]]], bool, numpy._typing._nested_sequence._NestedSequence[bool]]
_ArrayLikeUnknown: typing._UnionGenericAlias  # value = typing.Union[numpy._typing._array_like._SupportsArray[numpy.dtype[numpy._typing._array_like._UnknownType]], numpy._typing._nested_sequence._NestedSequence[numpy._typing._array_like._SupportsArray[numpy.dtype[numpy._typing._array_like._UnknownType]]], numpy._typing._array_like._UnknownType, numpy._typing._nested_sequence._NestedSequence[numpy._typing._array_like._UnknownType]]
_ArrayLikeVoid_co: typing._UnionGenericAlias  # value = typing.Union[numpy._typing._array_like._SupportsArray[numpy.dtype[numpy.void]], numpy._typing._nested_sequence._NestedSequence[numpy._typing._array_like._SupportsArray[numpy.dtype[numpy.void]]]]
_DType: typing.TypeVar  # value = ~_DType
_DType_co: typing.TypeVar  # value = +_DType_co
_DualArrayLike: typing._UnionGenericAlias  # value = typing.Union[numpy._typing._array_like._SupportsArray[~_DType], numpy._typing._nested_sequence._NestedSequence[numpy._typing._array_like._SupportsArray[~_DType]], ~_T, numpy._typing._nested_sequence._NestedSequence[~_T]]
_FiniteNestedSequence: typing._UnionGenericAlias  # value = typing.Union[~_T, collections.abc.Sequence[~_T], collections.abc.Sequence[collections.abc.Sequence[~_T]], collections.abc.Sequence[collections.abc.Sequence[collections.abc.Sequence[~_T]]], collections.abc.Sequence[collections.abc.Sequence[collections.abc.Sequence[collections.abc.Sequence[~_T]]]]]
_ScalarType: typing.TypeVar  # value = ~_ScalarType
_ScalarType_co: typing.TypeVar  # value = +_ScalarType_co
_T: typing.TypeVar  # value = ~_T
