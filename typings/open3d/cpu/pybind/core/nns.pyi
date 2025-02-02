from __future__ import annotations
import open3d.cpu.pybind.core
import typing
__all__ = ['NearestNeighborSearch']
class NearestNeighborSearch:
    """
    NearestNeighborSearch class for nearest neighbor search. 
    
    This class holds multiple index types to accelerate various nearest neighbor 
    search operations for a dataset of points with shape {n,d} with `n` as the number
    of points and `d` as the dimension of the points. The class supports knn search,
    fixed-radius search, multi-radius search, and hybrid search.
    
    Example:
        The following example demonstrates how to perform knn search using the class::
            
            import open3d as o3d
            import numpy as np
    
            dataset = np.random.rand(10,3)
            query_points = np.random.rand(5,3)
    
            nns = o3d.core.nns.NearestNeighborSearch(dataset)
    
            # initialize the knn_index before we can use knn_search
            nns.knn_index()
    
            # perform knn search to get the 3 closest points with respect to the 
            # Euclidean distance. The returned distance is given as the squared
            # distances
            indices, squared_distances = nns.knn_search(query_points, knn=3)
    
                    
    """
    def __init__(self, dataset_points: open3d.cpu.pybind.core.Tensor, index_dtype: open3d.cpu.pybind.core.Dtype = ...) -> None:
        ...
    def fixed_radius_index(self, radius: float | None = None) -> bool:
        """
        Initialize the index for fixed-radius search.
        
        This function needs to be called once before performing search operations.
        
        Args:
            radius (float, optional): Radius value for fixed-radius search. Required
                for GPU fixed radius index.
        
        Returns:
            True on success.
        """
    def fixed_radius_search(self, query_points: open3d.cpu.pybind.core.Tensor, radius: float, sort: bool | None = None) -> tuple[open3d.cpu.pybind.core.Tensor, open3d.cpu.pybind.core.Tensor, open3d.cpu.pybind.core.Tensor]:
        """
        Perform fixed-radius search.
        
        Note:
            To use fixed_radius_search initialize the index using fixed_radius_index before calling this function.
        
        Args:
                query_points (open3d.core.Tensor): Query points with shape {n, d}.
                radius (float): Radius value for fixed-radius search. Note that this
                    parameter can differ from the radius used to initialize the index
                    for convenience, which may cause the index to be rebuilt for GPU 
                    devices.
                sort (bool, optional): Sort the results by distance. Default is True.
        
        Returns:
                Tuple of Tensors (indices, splits, distances).
                    - indices: The indices of the neighbors.
                    - distances: The squared L2 distances.
                    - splits: The splits of the indices and distances defining the start 
                        and exclusive end of each query point's neighbors. The shape is {num_queries+1}
        
        Example:
            The following searches the neighbors within a radius of 1.0 a set of data and query points::
        
                # define data and query points
                points = np.array([
                [0.1,0.1,0.1],
                [0.9,0.9,0.9],
                [0.5,0.5,0.5],
                [1.7,1.7,1.7],
                [1.8,1.8,1.8],
                [0.3,2.4,1.4]], dtype=np.float32)
        
                queries = np.array([
                [1.0,1.0,1.0],
                [0.5,2.0,2.0],
                [0.5,2.1,2.1],
                [100,100,100],
                ], dtype=np.float32)
        
                nns = o3d.core.nns.NearestNeighborSearch(points)
                nns.fixed_radius_index(radius=1.0)
        
                neighbors_index, neighbors_distance, neighbors_splits = nns.fixed_radius_search(queries, radius=1.0, sort=True)
                # returns neighbors_index      = [1, 2, 5, 5]
                #         neighbors_distance   = [0.03 0.75 0.56000006 0.62]
                #         neighbors_splits = [0, 2, 3, 4, 4]
        
                for i, (start_i, end_i) in enumerate(zip(neighbors_splits, neighbors_splits[1:])):
                start_i = start_i.item()
                end_i = end_i.item()
                print(f"query_point {i} has the neighbors {neighbors_index[start_i:end_i].numpy()} "
                      f"with squared distances {neighbors_distance[start_i:end_i].numpy()}")
        """
    def hybrid_index(self, radius: float | None = None) -> bool:
        """
        Initialize the index for hybrid search.
        
        This function needs to be called once before performing search operations.
        
        Args:
            radius (float, optional): Radius value for hybrid search. Required
                for GPU hybrid index.
        
        Returns:
            True on success.
        """
    def hybrid_search(self, query_points: open3d.cpu.pybind.core.Tensor, radius: float, max_knn: int) -> tuple[open3d.cpu.pybind.core.Tensor, open3d.cpu.pybind.core.Tensor, open3d.cpu.pybind.core.Tensor]:
        """
        Perform hybrid search.
        
        Hybrid search behaves similarly to fixed-radius search, but with a maximum number of neighbors to search per query point.
        
        Note:
            To use hybrid_search initialize the index using hybrid_index before calling this function.
        
        Args:
            query_points (open3d.core.Tensor): Query points with shape {n, d}.
            radius (float): Radius value for hybrid search.
            max_knn (int): Maximum number of neighbor to search per query.
        
        Returns:
                Tuple of Tensors (indices, distances, counts)
                    - indices: The indices of the neighbors with shape {n, max_knn}.
                        If there are less than max_knn neighbors within the radius then the
                        last entries are padded with -1.
                    - distances: The squared L2 distances with shape {n, max_knn}.
                        If there are less than max_knn neighbors within the radius then the
                        last entries are padded with 0.
                    - counts: Counts of neighbour for each query points with shape {n}.
        """
    def knn_index(self) -> bool:
        """
        Initialize the index for knn search. 
        
        This function needs to be called once before performing search operations.
        
        Returns:
            True on success.
        """
    def knn_search(self, query_points: open3d.cpu.pybind.core.Tensor, knn: int) -> tuple[open3d.cpu.pybind.core.Tensor, open3d.cpu.pybind.core.Tensor]:
        """
        Perform knn search.
        
        Note:
            To use knn_search initialize the index using knn_index before calling this function.
        
        Args:
            query_points (open3d.core.Tensor): Query points with shape {n, d}.
            knn (int): Number of neighbors to search per query point.
        
        Example:
            The following searches the 3 nearest neighbors for random dataset and query points::
        
                import open3d as o3d
                import numpy as np
        
                dataset = np.random.rand(10,3)
                query_points = np.random.rand(5,3)
        
                nns = o3d.core.nns.NearestNeighborSearch(dataset)
        
                # initialize the knn_index before we can use knn_search
                nns.knn_index()
        
                # perform knn search to get the 3 closest points with respect to the 
                # Euclidean distance. The returned distance is given as the squared
                # distances
                indices, squared_distances = nns.knn_search(query_points, knn=3)
        
        Returns:
            Tuple of Tensors (indices, squared_distances).
                - indices: Tensor of shape {n, knn}.
                - squared_distances: Tensor of shape {n, knn}. The distances are squared L2 distances.
        """
    def multi_radius_index(self) -> bool:
        """
        Initialize the index for multi-radius search.
        
        This function needs to be called once before performing search operations.
        
        Returns:
                True on success.
        """
    def multi_radius_search(self, query_points: open3d.cpu.pybind.core.Tensor, radii: open3d.cpu.pybind.core.Tensor) -> tuple[open3d.cpu.pybind.core.Tensor, open3d.cpu.pybind.core.Tensor, open3d.cpu.pybind.core.Tensor]:
        """
        Perform multi-radius search. Each query point has an independent radius.
        
        Note:
            To use multi_radius_search initialize the index using multi_radius_index before calling this function.
        
        Args:
            query_points (open3d.core.Tensor): Query points with shape {n, d}.
            radii (open3d.core.Tensor): Radii of query points. Each query point has one radius.
        
        Returns:
                Tuple of Tensors (indices, splits, distances).
                    - indices: The indices of the neighbors.
                    - distances: The squared L2 distances.
                    - splits: The splits of the indices and distances defining the start 
                        and exclusive end of each query point's neighbors. The shape is {num_queries+1}
        
        Example:
            The following searches the neighbors with an individual radius for each query point::
        
                # define data, query points and radii
                points = np.array([
                [0.1,0.1,0.1],
                [0.9,0.9,0.9],
                [0.5,0.5,0.5],
                [1.7,1.7,1.7],
                [1.8,1.8,1.8],
                [0.3,2.4,1.4]], dtype=np.float32)
        
                queries = np.array([
                    [1.0,1.0,1.0],
                    [0.5,2.0,2.0],
                    [0.5,2.1,2.1],
                    [100,100,100],
                ], dtype=np.float32)
        
                radii = np.array([0.5, 1.0, 1.5, 2], dtype=np.float32)
        
                nns = o3d.core.nns.NearestNeighborSearch(points)
        
                nns.multi_radius_index()
        
                neighbors_index, neighbors_distance, neighbors_splits = nns.multi_radius_search(queries, radii)
                # returns neighbors_index      = [1 5 5 3 4]
                #         neighbors_distance   = [0.03 0.56 0.62 1.76 1.87]
                #         neighbors_splits = [0 1 2 5 5]
        
        
                for i, (start_i, end_i) in enumerate(zip(neighbors_splits, neighbors_splits[1:])):
                    start_i = start_i.item()
                    end_i = end_i.item()
                    print(f"query_point {i} has the neighbors {neighbors_index[start_i:end_i].numpy()} "
                        f"with squared distances {neighbors_distance[start_i:end_i].numpy()}")
        """
