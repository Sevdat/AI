from __future__ import annotations
import numpy
import typing
import typing_extensions
from . import random
__all__ = ['Debug', 'DoubleVector', 'Error', 'Info', 'IntVector', 'Matrix3dVector', 'Matrix4dVector', 'Vector2dVector', 'Vector2iVector', 'Vector3dVector', 'Vector3iVector', 'Vector4iVector', 'VerbosityContextManager', 'VerbosityLevel', 'Warning', 'get_verbosity_level', 'random', 'reset_print_function', 'set_verbosity_level']
class DoubleVector:
    """
    Convert float64 numpy array of shape ``(n,)`` to Open3D format.
    """
    __hash__: typing.ClassVar[None] = None
    def __bool__(self) -> bool:
        """
        Check whether the list is nonempty
        """
    def __buffer__(self, flags):
        """
        Return a buffer object that exposes the underlying memory of the object.
        """
    def __contains__(self, x: float) -> bool:
        """
        Return true the container contains ``x``
        """
    def __copy__(self) -> DoubleVector:
        ...
    def __deepcopy__(self, arg0: dict) -> DoubleVector:
        ...
    @typing.overload
    def __delitem__(self, arg0: int) -> None:
        """
        Delete the list elements at index ``i``
        """
    @typing.overload
    def __delitem__(self, arg0: slice) -> None:
        """
        Delete list elements using a slice object
        """
    def __eq__(self, arg0: DoubleVector) -> bool:
        ...
    @typing.overload
    def __getitem__(self, s: slice) -> DoubleVector:
        """
        Retrieve list elements using a slice object
        """
    @typing.overload
    def __getitem__(self, arg0: int) -> float:
        ...
    @typing.overload
    def __init__(self, arg0: typing_extensions.Buffer) -> None:
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: DoubleVector) -> None:
        """
        Copy constructor
        """
    @typing.overload
    def __init__(self, arg0: typing.Iterable) -> None:
        ...
    def __iter__(self) -> typing.Iterator[float]:
        ...
    def __len__(self) -> int:
        ...
    def __ne__(self, arg0: DoubleVector) -> bool:
        ...
    def __release_buffer__(self, buffer):
        """
        Release the buffer object that exposes the underlying memory of the object.
        """
    def __repr__(self) -> str:
        """
        Return the canonical string representation of this list.
        """
    @typing.overload
    def __setitem__(self, arg0: int, arg1: float) -> None:
        ...
    @typing.overload
    def __setitem__(self, arg0: slice, arg1: DoubleVector) -> None:
        """
        Assign list elements using a slice object
        """
    def append(self, x: float) -> None:
        """
        Add an item to the end of the list
        """
    def clear(self) -> None:
        """
        Clear the contents
        """
    def count(self, x: float) -> int:
        """
        Return the number of times ``x`` appears in the list
        """
    @typing.overload
    def extend(self, L: DoubleVector) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    @typing.overload
    def extend(self, L: typing.Iterable) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    def insert(self, i: int, x: float) -> None:
        """
        Insert an item at a given position.
        """
    @typing.overload
    def pop(self) -> float:
        """
        Remove and return the last item
        """
    @typing.overload
    def pop(self, i: int) -> float:
        """
        Remove and return the item at index ``i``
        """
    def remove(self, x: float) -> None:
        """
        Remove the first item from the list whose value is x. It is an error if there is no such item.
        """
class IntVector:
    """
    Convert int32 numpy array of shape ``(n,)`` to Open3D format.
    """
    __hash__: typing.ClassVar[None] = None
    def __bool__(self) -> bool:
        """
        Check whether the list is nonempty
        """
    def __buffer__(self, flags):
        """
        Return a buffer object that exposes the underlying memory of the object.
        """
    def __contains__(self, x: int) -> bool:
        """
        Return true the container contains ``x``
        """
    def __copy__(self) -> IntVector:
        ...
    def __deepcopy__(self, arg0: dict) -> IntVector:
        ...
    @typing.overload
    def __delitem__(self, arg0: int) -> None:
        """
        Delete the list elements at index ``i``
        """
    @typing.overload
    def __delitem__(self, arg0: slice) -> None:
        """
        Delete list elements using a slice object
        """
    def __eq__(self, arg0: IntVector) -> bool:
        ...
    @typing.overload
    def __getitem__(self, s: slice) -> IntVector:
        """
        Retrieve list elements using a slice object
        """
    @typing.overload
    def __getitem__(self, arg0: int) -> int:
        ...
    @typing.overload
    def __init__(self, arg0: typing_extensions.Buffer) -> None:
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: IntVector) -> None:
        """
        Copy constructor
        """
    @typing.overload
    def __init__(self, arg0: typing.Iterable) -> None:
        ...
    def __iter__(self) -> typing.Iterator[int]:
        ...
    def __len__(self) -> int:
        ...
    def __ne__(self, arg0: IntVector) -> bool:
        ...
    def __release_buffer__(self, buffer):
        """
        Release the buffer object that exposes the underlying memory of the object.
        """
    def __repr__(self) -> str:
        """
        Return the canonical string representation of this list.
        """
    @typing.overload
    def __setitem__(self, arg0: int, arg1: int) -> None:
        ...
    @typing.overload
    def __setitem__(self, arg0: slice, arg1: IntVector) -> None:
        """
        Assign list elements using a slice object
        """
    def append(self, x: int) -> None:
        """
        Add an item to the end of the list
        """
    def clear(self) -> None:
        """
        Clear the contents
        """
    def count(self, x: int) -> int:
        """
        Return the number of times ``x`` appears in the list
        """
    @typing.overload
    def extend(self, L: IntVector) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    @typing.overload
    def extend(self, L: typing.Iterable) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    def insert(self, i: int, x: int) -> None:
        """
        Insert an item at a given position.
        """
    @typing.overload
    def pop(self) -> int:
        """
        Remove and return the last item
        """
    @typing.overload
    def pop(self, i: int) -> int:
        """
        Remove and return the item at index ``i``
        """
    def remove(self, x: int) -> None:
        """
        Remove the first item from the list whose value is x. It is an error if there is no such item.
        """
class Matrix3dVector:
    """
    Convert float64 numpy array of shape ``(n, 3, 3)`` to Open3D format.
    """
    __hash__: typing.ClassVar[None] = None
    def __bool__(self) -> bool:
        """
        Check whether the list is nonempty
        """
    def __buffer__(self, flags):
        """
        Return a buffer object that exposes the underlying memory of the object.
        """
    def __contains__(self, x: numpy.ndarray[numpy.float64[3, 3]]) -> bool:
        """
        Return true the container contains ``x``
        """
    def __copy__(self) -> Matrix3dVector:
        ...
    def __deepcopy__(self, arg0: dict) -> Matrix3dVector:
        ...
    @typing.overload
    def __delitem__(self, arg0: int) -> None:
        """
        Delete the list elements at index ``i``
        """
    @typing.overload
    def __delitem__(self, arg0: slice) -> None:
        """
        Delete list elements using a slice object
        """
    def __eq__(self, arg0: Matrix3dVector) -> bool:
        ...
    @typing.overload
    def __getitem__(self, s: slice) -> Matrix3dVector:
        """
        Retrieve list elements using a slice object
        """
    @typing.overload
    def __getitem__(self, arg0: int) -> numpy.ndarray[numpy.float64[3, 3]]:
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: Matrix3dVector) -> None:
        """
        Copy constructor
        """
    @typing.overload
    def __init__(self, arg0: typing.Iterable) -> None:
        ...
    def __iter__(self) -> typing.Iterator[numpy.ndarray[numpy.float64[3, 3]]]:
        ...
    def __len__(self) -> int:
        ...
    def __ne__(self, arg0: Matrix3dVector) -> bool:
        ...
    def __release_buffer__(self, buffer):
        """
        Release the buffer object that exposes the underlying memory of the object.
        """
    def __repr__(self) -> str:
        ...
    @typing.overload
    def __setitem__(self, arg0: int, arg1: numpy.ndarray[numpy.float64[3, 3]]) -> None:
        ...
    @typing.overload
    def __setitem__(self, arg0: slice, arg1: Matrix3dVector) -> None:
        """
        Assign list elements using a slice object
        """
    def append(self, x: numpy.ndarray[numpy.float64[3, 3]]) -> None:
        """
        Add an item to the end of the list
        """
    def clear(self) -> None:
        """
        Clear the contents
        """
    def count(self, x: numpy.ndarray[numpy.float64[3, 3]]) -> int:
        """
        Return the number of times ``x`` appears in the list
        """
    @typing.overload
    def extend(self, L: Matrix3dVector) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    @typing.overload
    def extend(self, L: typing.Iterable) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    def insert(self, i: int, x: numpy.ndarray[numpy.float64[3, 3]]) -> None:
        """
        Insert an item at a given position.
        """
    @typing.overload
    def pop(self) -> numpy.ndarray[numpy.float64[3, 3]]:
        """
        Remove and return the last item
        """
    @typing.overload
    def pop(self, i: int) -> numpy.ndarray[numpy.float64[3, 3]]:
        """
        Remove and return the item at index ``i``
        """
    def remove(self, x: numpy.ndarray[numpy.float64[3, 3]]) -> None:
        """
        Remove the first item from the list whose value is x. It is an error if there is no such item.
        """
class Matrix4dVector:
    """
    Convert float64 numpy array of shape ``(n, 4, 4)`` to Open3D format.
    """
    __hash__: typing.ClassVar[None] = None
    def __bool__(self) -> bool:
        """
        Check whether the list is nonempty
        """
    def __buffer__(self, flags):
        """
        Return a buffer object that exposes the underlying memory of the object.
        """
    def __contains__(self, x: numpy.ndarray[numpy.float64[4, 4]]) -> bool:
        """
        Return true the container contains ``x``
        """
    def __copy__(self) -> Matrix4dVector:
        ...
    def __deepcopy__(self, arg0: dict) -> Matrix4dVector:
        ...
    @typing.overload
    def __delitem__(self, arg0: int) -> None:
        """
        Delete the list elements at index ``i``
        """
    @typing.overload
    def __delitem__(self, arg0: slice) -> None:
        """
        Delete list elements using a slice object
        """
    def __eq__(self, arg0: Matrix4dVector) -> bool:
        ...
    @typing.overload
    def __getitem__(self, s: slice) -> Matrix4dVector:
        """
        Retrieve list elements using a slice object
        """
    @typing.overload
    def __getitem__(self, arg0: int) -> numpy.ndarray[numpy.float64[4, 4]]:
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: Matrix4dVector) -> None:
        """
        Copy constructor
        """
    @typing.overload
    def __init__(self, arg0: typing.Iterable) -> None:
        ...
    def __iter__(self) -> typing.Iterator[numpy.ndarray[numpy.float64[4, 4]]]:
        ...
    def __len__(self) -> int:
        ...
    def __ne__(self, arg0: Matrix4dVector) -> bool:
        ...
    def __release_buffer__(self, buffer):
        """
        Release the buffer object that exposes the underlying memory of the object.
        """
    def __repr__(self) -> str:
        ...
    @typing.overload
    def __setitem__(self, arg0: int, arg1: numpy.ndarray[numpy.float64[4, 4]]) -> None:
        ...
    @typing.overload
    def __setitem__(self, arg0: slice, arg1: Matrix4dVector) -> None:
        """
        Assign list elements using a slice object
        """
    def append(self, x: numpy.ndarray[numpy.float64[4, 4]]) -> None:
        """
        Add an item to the end of the list
        """
    def clear(self) -> None:
        """
        Clear the contents
        """
    def count(self, x: numpy.ndarray[numpy.float64[4, 4]]) -> int:
        """
        Return the number of times ``x`` appears in the list
        """
    @typing.overload
    def extend(self, L: Matrix4dVector) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    @typing.overload
    def extend(self, L: typing.Iterable) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    def insert(self, i: int, x: numpy.ndarray[numpy.float64[4, 4]]) -> None:
        """
        Insert an item at a given position.
        """
    @typing.overload
    def pop(self) -> numpy.ndarray[numpy.float64[4, 4]]:
        """
        Remove and return the last item
        """
    @typing.overload
    def pop(self, i: int) -> numpy.ndarray[numpy.float64[4, 4]]:
        """
        Remove and return the item at index ``i``
        """
    def remove(self, x: numpy.ndarray[numpy.float64[4, 4]]) -> None:
        """
        Remove the first item from the list whose value is x. It is an error if there is no such item.
        """
class Vector2dVector:
    """
    Convert float64 numpy array of shape ``(n, 2)`` to Open3D format.
    """
    __hash__: typing.ClassVar[None] = None
    def __bool__(self) -> bool:
        """
        Check whether the list is nonempty
        """
    def __buffer__(self, flags):
        """
        Return a buffer object that exposes the underlying memory of the object.
        """
    def __contains__(self, x: numpy.ndarray[numpy.float64[2, 1]]) -> bool:
        """
        Return true the container contains ``x``
        """
    def __copy__(self) -> Vector2dVector:
        ...
    def __deepcopy__(self, arg0: dict) -> Vector2dVector:
        ...
    @typing.overload
    def __delitem__(self, arg0: int) -> None:
        """
        Delete the list elements at index ``i``
        """
    @typing.overload
    def __delitem__(self, arg0: slice) -> None:
        """
        Delete list elements using a slice object
        """
    def __eq__(self, arg0: Vector2dVector) -> bool:
        ...
    @typing.overload
    def __getitem__(self, s: slice) -> Vector2dVector:
        """
        Retrieve list elements using a slice object
        """
    @typing.overload
    def __getitem__(self, arg0: int) -> numpy.ndarray[numpy.float64[2, 1]]:
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: numpy.ndarray[numpy.float64]) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: Vector2dVector) -> None:
        """
        Copy constructor
        """
    @typing.overload
    def __init__(self, arg0: typing.Iterable) -> None:
        ...
    def __iter__(self) -> typing.Iterator[numpy.ndarray[numpy.float64[2, 1]]]:
        ...
    def __len__(self) -> int:
        ...
    def __ne__(self, arg0: Vector2dVector) -> bool:
        ...
    def __release_buffer__(self, buffer):
        """
        Release the buffer object that exposes the underlying memory of the object.
        """
    def __repr__(self) -> str:
        ...
    @typing.overload
    def __setitem__(self, arg0: int, arg1: numpy.ndarray[numpy.float64[2, 1]]) -> None:
        ...
    @typing.overload
    def __setitem__(self, arg0: slice, arg1: Vector2dVector) -> None:
        """
        Assign list elements using a slice object
        """
    def append(self, x: numpy.ndarray[numpy.float64[2, 1]]) -> None:
        """
        Add an item to the end of the list
        """
    def clear(self) -> None:
        """
        Clear the contents
        """
    def count(self, x: numpy.ndarray[numpy.float64[2, 1]]) -> int:
        """
        Return the number of times ``x`` appears in the list
        """
    @typing.overload
    def extend(self, L: Vector2dVector) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    @typing.overload
    def extend(self, L: typing.Iterable) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    def insert(self, i: int, x: numpy.ndarray[numpy.float64[2, 1]]) -> None:
        """
        Insert an item at a given position.
        """
    @typing.overload
    def pop(self) -> numpy.ndarray[numpy.float64[2, 1]]:
        """
        Remove and return the last item
        """
    @typing.overload
    def pop(self, i: int) -> numpy.ndarray[numpy.float64[2, 1]]:
        """
        Remove and return the item at index ``i``
        """
    def remove(self, x: numpy.ndarray[numpy.float64[2, 1]]) -> None:
        """
        Remove the first item from the list whose value is x. It is an error if there is no such item.
        """
class Vector2iVector:
    """
    Convert int32 numpy array of shape ``(n, 2)`` to Open3D format.
    """
    __hash__: typing.ClassVar[None] = None
    def __bool__(self) -> bool:
        """
        Check whether the list is nonempty
        """
    def __buffer__(self, flags):
        """
        Return a buffer object that exposes the underlying memory of the object.
        """
    def __contains__(self, x: numpy.ndarray[numpy.int32[2, 1]]) -> bool:
        """
        Return true the container contains ``x``
        """
    def __copy__(self) -> Vector2iVector:
        ...
    def __deepcopy__(self, arg0: dict) -> Vector2iVector:
        ...
    @typing.overload
    def __delitem__(self, arg0: int) -> None:
        """
        Delete the list elements at index ``i``
        """
    @typing.overload
    def __delitem__(self, arg0: slice) -> None:
        """
        Delete list elements using a slice object
        """
    def __eq__(self, arg0: Vector2iVector) -> bool:
        ...
    @typing.overload
    def __getitem__(self, s: slice) -> Vector2iVector:
        """
        Retrieve list elements using a slice object
        """
    @typing.overload
    def __getitem__(self, arg0: int) -> numpy.ndarray[numpy.int32[2, 1]]:
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: numpy.ndarray[numpy.int32]) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: Vector2iVector) -> None:
        """
        Copy constructor
        """
    @typing.overload
    def __init__(self, arg0: typing.Iterable) -> None:
        ...
    def __iter__(self) -> typing.Iterator[numpy.ndarray[numpy.int32[2, 1]]]:
        ...
    def __len__(self) -> int:
        ...
    def __ne__(self, arg0: Vector2iVector) -> bool:
        ...
    def __release_buffer__(self, buffer):
        """
        Release the buffer object that exposes the underlying memory of the object.
        """
    def __repr__(self) -> str:
        ...
    @typing.overload
    def __setitem__(self, arg0: int, arg1: numpy.ndarray[numpy.int32[2, 1]]) -> None:
        ...
    @typing.overload
    def __setitem__(self, arg0: slice, arg1: Vector2iVector) -> None:
        """
        Assign list elements using a slice object
        """
    def append(self, x: numpy.ndarray[numpy.int32[2, 1]]) -> None:
        """
        Add an item to the end of the list
        """
    def clear(self) -> None:
        """
        Clear the contents
        """
    def count(self, x: numpy.ndarray[numpy.int32[2, 1]]) -> int:
        """
        Return the number of times ``x`` appears in the list
        """
    @typing.overload
    def extend(self, L: Vector2iVector) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    @typing.overload
    def extend(self, L: typing.Iterable) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    def insert(self, i: int, x: numpy.ndarray[numpy.int32[2, 1]]) -> None:
        """
        Insert an item at a given position.
        """
    @typing.overload
    def pop(self) -> numpy.ndarray[numpy.int32[2, 1]]:
        """
        Remove and return the last item
        """
    @typing.overload
    def pop(self, i: int) -> numpy.ndarray[numpy.int32[2, 1]]:
        """
        Remove and return the item at index ``i``
        """
    def remove(self, x: numpy.ndarray[numpy.int32[2, 1]]) -> None:
        """
        Remove the first item from the list whose value is x. It is an error if there is no such item.
        """
class Vector3dVector:
    """
    Convert float64 numpy array of shape ``(n, 3)`` to Open3D format.
    
    Example usage
    
    .. code-block:: python
    
        import open3d
        import numpy as np
    
        pcd = open3d.geometry.PointCloud()
        np_points = np.random.rand(100, 3)
    
        # From numpy to Open3D
        pcd.points = open3d.utility.Vector3dVector(np_points)
    
        # From Open3D to numpy
        np_points = np.asarray(pcd.points)
    """
    __hash__: typing.ClassVar[None] = None
    def __bool__(self) -> bool:
        """
        Check whether the list is nonempty
        """
    def __buffer__(self, flags):
        """
        Return a buffer object that exposes the underlying memory of the object.
        """
    def __contains__(self, x: numpy.ndarray[numpy.float64[3, 1]]) -> bool:
        """
        Return true the container contains ``x``
        """
    def __copy__(self) -> Vector3dVector:
        ...
    def __deepcopy__(self, arg0: dict) -> Vector3dVector:
        ...
    @typing.overload
    def __delitem__(self, arg0: int) -> None:
        """
        Delete the list elements at index ``i``
        """
    @typing.overload
    def __delitem__(self, arg0: slice) -> None:
        """
        Delete list elements using a slice object
        """
    def __eq__(self, arg0: Vector3dVector) -> bool:
        ...
    @typing.overload
    def __getitem__(self, s: slice) -> Vector3dVector:
        """
        Retrieve list elements using a slice object
        """
    @typing.overload
    def __getitem__(self, arg0: int) -> numpy.ndarray[numpy.float64[3, 1]]:
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: numpy.ndarray[numpy.float64]) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: Vector3dVector) -> None:
        """
        Copy constructor
        """
    @typing.overload
    def __init__(self, arg0: typing.Iterable) -> None:
        ...
    def __iter__(self) -> typing.Iterator[numpy.ndarray[numpy.float64[3, 1]]]:
        ...
    def __len__(self) -> int:
        ...
    def __ne__(self, arg0: Vector3dVector) -> bool:
        ...
    def __release_buffer__(self, buffer):
        """
        Release the buffer object that exposes the underlying memory of the object.
        """
    def __repr__(self) -> str:
        ...
    @typing.overload
    def __setitem__(self, arg0: int, arg1: numpy.ndarray[numpy.float64[3, 1]]) -> None:
        ...
    @typing.overload
    def __setitem__(self, arg0: slice, arg1: Vector3dVector) -> None:
        """
        Assign list elements using a slice object
        """
    def append(self, x: numpy.ndarray[numpy.float64[3, 1]]) -> None:
        """
        Add an item to the end of the list
        """
    def clear(self) -> None:
        """
        Clear the contents
        """
    def count(self, x: numpy.ndarray[numpy.float64[3, 1]]) -> int:
        """
        Return the number of times ``x`` appears in the list
        """
    @typing.overload
    def extend(self, L: Vector3dVector) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    @typing.overload
    def extend(self, L: typing.Iterable) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    def insert(self, i: int, x: numpy.ndarray[numpy.float64[3, 1]]) -> None:
        """
        Insert an item at a given position.
        """
    @typing.overload
    def pop(self) -> numpy.ndarray[numpy.float64[3, 1]]:
        """
        Remove and return the last item
        """
    @typing.overload
    def pop(self, i: int) -> numpy.ndarray[numpy.float64[3, 1]]:
        """
        Remove and return the item at index ``i``
        """
    def remove(self, x: numpy.ndarray[numpy.float64[3, 1]]) -> None:
        """
        Remove the first item from the list whose value is x. It is an error if there is no such item.
        """
class Vector3iVector:
    """
    Convert int32 numpy array of shape ``(n, 3)`` to Open3D format..
    
    Example usage
    
    .. code-block:: python
    
        import open3d
        import numpy as np
    
        # Example mesh
        # x, y coordinates:
        # [0: (-1, 2)]__________[1: (1, 2)]
        #             \\        /\\
        #              \\  (0) /  \\
        #               \\    / (1)\\
        #                \\  /      \\
        #      [2: (0, 0)]\\/________\\[3: (2, 0)]
        #
        # z coordinate: 0
    
        mesh = open3d.geometry.TriangleMesh()
        np_vertices = np.array([[-1, 2, 0],
                                [1, 2, 0],
                                [0, 0, 0],
                                [2, 0, 0]])
        np_triangles = np.array([[0, 2, 1],
                                 [1, 2, 3]]).astype(np.int32)
        mesh.vertices = open3d.Vector3dVector(np_vertices)
    
        # From numpy to Open3D
        mesh.triangles = open3d.Vector3iVector(np_triangles)
    
        # From Open3D to numpy
        np_triangles = np.asarray(mesh.triangles)
    """
    __hash__: typing.ClassVar[None] = None
    def __bool__(self) -> bool:
        """
        Check whether the list is nonempty
        """
    def __buffer__(self, flags):
        """
        Return a buffer object that exposes the underlying memory of the object.
        """
    def __contains__(self, x: numpy.ndarray[numpy.int32[3, 1]]) -> bool:
        """
        Return true the container contains ``x``
        """
    def __copy__(self) -> Vector3iVector:
        ...
    def __deepcopy__(self, arg0: dict) -> Vector3iVector:
        ...
    @typing.overload
    def __delitem__(self, arg0: int) -> None:
        """
        Delete the list elements at index ``i``
        """
    @typing.overload
    def __delitem__(self, arg0: slice) -> None:
        """
        Delete list elements using a slice object
        """
    def __eq__(self, arg0: Vector3iVector) -> bool:
        ...
    @typing.overload
    def __getitem__(self, s: slice) -> Vector3iVector:
        """
        Retrieve list elements using a slice object
        """
    @typing.overload
    def __getitem__(self, arg0: int) -> numpy.ndarray[numpy.int32[3, 1]]:
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: numpy.ndarray[numpy.int32]) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: Vector3iVector) -> None:
        """
        Copy constructor
        """
    @typing.overload
    def __init__(self, arg0: typing.Iterable) -> None:
        ...
    def __iter__(self) -> typing.Iterator[numpy.ndarray[numpy.int32[3, 1]]]:
        ...
    def __len__(self) -> int:
        ...
    def __ne__(self, arg0: Vector3iVector) -> bool:
        ...
    def __release_buffer__(self, buffer):
        """
        Release the buffer object that exposes the underlying memory of the object.
        """
    def __repr__(self) -> str:
        ...
    @typing.overload
    def __setitem__(self, arg0: int, arg1: numpy.ndarray[numpy.int32[3, 1]]) -> None:
        ...
    @typing.overload
    def __setitem__(self, arg0: slice, arg1: Vector3iVector) -> None:
        """
        Assign list elements using a slice object
        """
    def append(self, x: numpy.ndarray[numpy.int32[3, 1]]) -> None:
        """
        Add an item to the end of the list
        """
    def clear(self) -> None:
        """
        Clear the contents
        """
    def count(self, x: numpy.ndarray[numpy.int32[3, 1]]) -> int:
        """
        Return the number of times ``x`` appears in the list
        """
    @typing.overload
    def extend(self, L: Vector3iVector) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    @typing.overload
    def extend(self, L: typing.Iterable) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    def insert(self, i: int, x: numpy.ndarray[numpy.int32[3, 1]]) -> None:
        """
        Insert an item at a given position.
        """
    @typing.overload
    def pop(self) -> numpy.ndarray[numpy.int32[3, 1]]:
        """
        Remove and return the last item
        """
    @typing.overload
    def pop(self, i: int) -> numpy.ndarray[numpy.int32[3, 1]]:
        """
        Remove and return the item at index ``i``
        """
    def remove(self, x: numpy.ndarray[numpy.int32[3, 1]]) -> None:
        """
        Remove the first item from the list whose value is x. It is an error if there is no such item.
        """
class Vector4iVector:
    """
    Convert int numpy array of shape ``(n, 4)`` to Open3D format.
    """
    __hash__: typing.ClassVar[None] = None
    def __bool__(self) -> bool:
        """
        Check whether the list is nonempty
        """
    def __buffer__(self, flags):
        """
        Return a buffer object that exposes the underlying memory of the object.
        """
    def __contains__(self, x: numpy.ndarray[numpy.int32[4, 1]]) -> bool:
        """
        Return true the container contains ``x``
        """
    def __copy__(self) -> Vector4iVector:
        ...
    def __deepcopy__(self, arg0: dict) -> Vector4iVector:
        ...
    @typing.overload
    def __delitem__(self, arg0: int) -> None:
        """
        Delete the list elements at index ``i``
        """
    @typing.overload
    def __delitem__(self, arg0: slice) -> None:
        """
        Delete list elements using a slice object
        """
    def __eq__(self, arg0: Vector4iVector) -> bool:
        ...
    @typing.overload
    def __getitem__(self, s: slice) -> Vector4iVector:
        """
        Retrieve list elements using a slice object
        """
    @typing.overload
    def __getitem__(self, arg0: int) -> numpy.ndarray[numpy.int32[4, 1]]:
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: numpy.ndarray[numpy.int32]) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: Vector4iVector) -> None:
        """
        Copy constructor
        """
    @typing.overload
    def __init__(self, arg0: typing.Iterable) -> None:
        ...
    def __iter__(self) -> typing.Iterator[numpy.ndarray[numpy.int32[4, 1]]]:
        ...
    def __len__(self) -> int:
        ...
    def __ne__(self, arg0: Vector4iVector) -> bool:
        ...
    def __release_buffer__(self, buffer):
        """
        Release the buffer object that exposes the underlying memory of the object.
        """
    def __repr__(self) -> str:
        ...
    @typing.overload
    def __setitem__(self, arg0: int, arg1: numpy.ndarray[numpy.int32[4, 1]]) -> None:
        ...
    @typing.overload
    def __setitem__(self, arg0: slice, arg1: Vector4iVector) -> None:
        """
        Assign list elements using a slice object
        """
    def append(self, x: numpy.ndarray[numpy.int32[4, 1]]) -> None:
        """
        Add an item to the end of the list
        """
    def clear(self) -> None:
        """
        Clear the contents
        """
    def count(self, x: numpy.ndarray[numpy.int32[4, 1]]) -> int:
        """
        Return the number of times ``x`` appears in the list
        """
    @typing.overload
    def extend(self, L: Vector4iVector) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    @typing.overload
    def extend(self, L: typing.Iterable) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    def insert(self, i: int, x: numpy.ndarray[numpy.int32[4, 1]]) -> None:
        """
        Insert an item at a given position.
        """
    @typing.overload
    def pop(self) -> numpy.ndarray[numpy.int32[4, 1]]:
        """
        Remove and return the last item
        """
    @typing.overload
    def pop(self, i: int) -> numpy.ndarray[numpy.int32[4, 1]]:
        """
        Remove and return the item at index ``i``
        """
    def remove(self, x: numpy.ndarray[numpy.int32[4, 1]]) -> None:
        """
        Remove the first item from the list whose value is x. It is an error if there is no such item.
        """
class VerbosityContextManager:
    """
    A context manager to temporally change the verbosity level of Open3D
    """
    def __enter__(self) -> None:
        """
        Enter the context manager
        """
    def __exit__(self, arg0: typing.Any, arg1: typing.Any, arg2: typing.Any) -> None:
        """
        Exit the context manager
        """
    def __init__(self, level: VerbosityLevel) -> None:
        """
        Create a VerbosityContextManager with a given VerbosityLevel
        """
class VerbosityLevel:
    """
    Enum class for VerbosityLevel.
    """
    Debug: typing.ClassVar[VerbosityLevel]  # value = <VerbosityLevel.Debug: 3>
    Error: typing.ClassVar[VerbosityLevel]  # value = <VerbosityLevel.Error: 0>
    Info: typing.ClassVar[VerbosityLevel]  # value = <VerbosityLevel.Info: 2>
    Warning: typing.ClassVar[VerbosityLevel]  # value = <VerbosityLevel.Warning: 1>
    __members__: typing.ClassVar[dict[str, VerbosityLevel]]  # value = {'Error': <VerbosityLevel.Error: 0>, 'Warning': <VerbosityLevel.Warning: 1>, 'Info': <VerbosityLevel.Info: 2>, 'Debug': <VerbosityLevel.Debug: 3>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __ge__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __gt__(self, other: typing.Any) -> bool:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __le__(self, other: typing.Any) -> bool:
        ...
    def __lt__(self, other: typing.Any) -> bool:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
def get_verbosity_level() -> VerbosityLevel:
    """
    Get global verbosity level of Open3D
    
    Returns:
        open3d.cpu.pybind.utility.VerbosityLevel
    """
def reset_print_function() -> None:
    ...
def set_verbosity_level(verbosity_level: VerbosityLevel) -> None:
    """
    Set global verbosity level of Open3D
    
    Args:
        verbosity_level (open3d.cpu.pybind.utility.VerbosityLevel): Messages with equal or less than ``verbosity_level`` verbosity will be printed.
    
    Returns:
        None
    """
Debug: VerbosityLevel  # value = <VerbosityLevel.Debug: 3>
Error: VerbosityLevel  # value = <VerbosityLevel.Error: 0>
Info: VerbosityLevel  # value = <VerbosityLevel.Info: 2>
Warning: VerbosityLevel  # value = <VerbosityLevel.Warning: 1>
