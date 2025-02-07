import cupy as cp
import numpy as np
import numba
from numba import cuda

# Define the array of vectors
V1 = cp.array([
    [0, 1, 3],
    [2, 2, 7],
    [6, 5, 1]
], dtype=cp.float32)

# Define the array of vectors
V2 = cp.array([
    [0, 1, 3],
    [4, 5, 6],
    [7, 8, 9]
], dtype=cp.float32)

@cuda.jit
def compute_distances_kernel(index, V1, V2, distances):
    i = cuda.grid(1)  # Thread index

    element1 = V1[index]  # Get the element to compare
    element2 = V2[index]  # Get the element to compare
    if i < V1.shape[0]:  # Check if the thread is within bounds
        if ((element1[0] != element2[0]) and (element1[1] != element2[1]) and (element1[2] != element2[2])):
            dist1 = (((V1[i, 0] - element1[0]) ** 2) + 
                    ((V1[i, 1] - element1[1]) ** 2) + 
                    ((V1[i, 2] - element1[2]) ** 2)) ** 0.5
            
            dist2 = (((V2[i, 0] - element2[0]) ** 2) + 
                    ((V2[i, 1] - element2[1]) ** 2) + 
                    ((V2[i, 2] - element2[2]) ** 2)) ** 0.5
            
            distances[i] = abs(dist1 - dist2) 
        else:
            distances[i] = cp.inf

def replaceArrayIndex(index,distances,possibleIndexes):
    size = range(len(distances))
    substract = 0
    for i in size: 
        if ((distances[i] != index)) :
            possibleIndexes[index][i+substract] = distances[i] 
        else :
            substract -= 1

def getDistancesFromElement(V1,V2):
    N, D = V2.shape
    possibleIndexes = cp.full((N,10), -1, dtype=cp.int32)
    threads_per_block = 128
    blocks_per_grid = (N + threads_per_block - 1) // threads_per_block
    
    for index in range(N):
        distances = cp.zeros(N, dtype=cp.float32)
        compute_distances_kernel[blocks_per_grid, threads_per_block](index,V1,V2, distances)
        sorted = cp.argsort(distances)[:10]
        distances = sorted[distances[sorted] != cp.inf]
        print(distances)
        replaceArrayIndex(index,distances,possibleIndexes)

    return possibleIndexes

# Example usage
if __name__ == "__main__":
    index = 0  # Compare the second element (index 1) to all others
    distances = getDistancesFromElement(V2, V1)
    print("Distances from element", index, "to all others:\n", distances)