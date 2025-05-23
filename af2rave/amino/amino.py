"""AMINO: generating a minimally redundant set of order parameters through
clustering of mutual information based distances. Method by Ravindra, Smith,
and Tiwary. Code maintained by Ravindra and Smith. Tested and reorganized by
Da Teng in September 2024.

This is the kernel density estimation version.

Read and cite the following when using this method:
https://doi.org/10.1039/C9ME00115H
"""

from __future__ import annotations

import numpy as np
import multiprocessing as mp
from itertools import combinations_with_replacement

from timeit import default_timer as timer
from tqdm import tqdm

from numpy.typing import ArrayLike

def product_without_self(arr: np.array) -> np.array:

    # Compute the prefix products
    left_products = np.cumprod(arr)
    # Compute the suffix products (reversing and then computing cumulative product)
    right_products = np.cumprod(arr[::-1])[::-1]

    # Prepare the output by multiplying left and right products
    output = np.ones_like(arr)
    output[1:] = left_products[:-1]
    output[:-1] *= right_products[1:]

    return output


class OrderParameter:
    """Order Parameter (OP) class - stores OP name and trajectory

    Attributes
    ----------
    name : str
        Name of OP.

    traj : list of floats or np.array
        Trajectory of OP values. This will be normalized to have std = 1.

    """

    # name should be unique to the Order Parameter being defined
    def __init__(self, name: str, traj: ArrayLike):
        self.name = name
        self.traj = np.array(traj).reshape([-1, 1]) / np.std(traj)

    def __eq__(self, other):
        return self.name == other.name

    # This is needed for the sets construction used in clustering
    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        return str(self.name)


# Memoizes distance computation between OP's to prevent re-calculations
class DistanceMatrix:
    """Memoizes distance computation between OP's to prevent re-calculations.

    Attributes
    ----------
    bins : int
        Number of values along each axis for the joint probability.
        The probability will be a bins x bins grid.
    """

    def __init__(self, bins: int):
        self.memo = {}
        self.bins = bins

    @classmethod
    def from_matrix(cls, matrix: np.ndarray, names: list[str]) -> DistanceMatrix:

        # check dimensions
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Matrix must be square.")
        if matrix.shape[0] != len(names):
            raise ValueError("Matrix and names must have the same length.")

        # check for symmetry
        if not np.allclose(matrix, matrix.T):
            raise ValueError("Matrix must be symmetric.")
        if not np.allclose(np.diag(matrix), 0):
            raise ValueError("Diagonal of the matrix must be zero.")
        if not np.all(matrix >= 0) or not np.all(np.isreal(matrix)):
            raise ValueError("Matrix must be positive semi-definite.")

        instance = cls(bins=0)
        for i, j in combinations_with_replacement(range(len(names)), 2):
            idx = frozenset((names[i], names[j]))
            instance.memo[idx] = matrix[i, j]
        
        return instance

    def initialize_distances(self, ops: list[OrderParameter], verbose=False) -> None:

        self.ops = ops
        n = len(ops)

        pairs = list(combinations_with_replacement(range(n), 2))
        with mp.Pool(processes=mp.cpu_count()) as p:
            if verbose:
                result = p.starmap(self._compute_pair, pairs)
            else:
                result = p.starmap(self._compute_pair, 
                                tqdm(pairs, desc="Computing distances", total=n*(n+1)//2)
                )

        for (i, j), r in zip(pairs, result):
            idx = frozenset((self.ops[i].name, self.ops[j].name))
            self.memo[idx] = r

    def _compute_pair(self, i: int, j: int) -> float:
        """Compute distance for a single pair"""
        x = self.ops[i].traj.flatten()
        y = self.ops[j].traj.flatten()
        
        # Compute joint probability
        p_xy = np.histogram2d(x, y, bins=self.bins)[0]
        p_xy = p_xy / np.sum(p_xy)
        return self._calculate_mi_distance(p_xy)

    def _calculate_mi_distance(self, p_xy: np.ndarray) -> float:
        """Mutual information distance calculation"""
        p_x = np.sum(p_xy, axis=1)
        p_y = np.sum(p_xy, axis=0)

        log_p_x_times_p_y = np.ma.log(np.tensordot(p_x, p_y, axes=0))
        log_p_xy = np.ma.log(p_xy)

        info = np.sum(p_xy * (log_p_xy - log_p_x_times_p_y))
        entropy = np.sum(-1 * p_xy * log_p_xy)

        return max(0.0, (1 - (info / entropy)))

    def distance(self, op1: str, op2: str) -> float:
        idx = frozenset((op1, op2))
        try:
            return self.memo[idx]
        except KeyError:
            raise ValueError(f"Distance between {op1.name} and {op2.name} not found")

    def distortion(self, centers: list[str], ops: list[str]) -> float:
        """Computes the distortion between a set of centeroids and OPs.
        When multiple centoids are used, the minimum distortion grouping
        will be used to calculate the total distortion.

        Parameters
        ----------
        centers : OrderParameters
            Cluster centroids.

        ops : list of OrderParameters
            All OPs belonging to the centroid's clusters.

        Returns
        -------
        float
            Minimum total distortion given the centroids and OPs.

        """

        dis = 0.0
        for i in ops:
            min_val = np.min([self.distance(i, c) for c in centers])
            dis += (min_val * min_val)
        return 1 + np.sqrt(dis)


# Dissimilarity Matrix (DM) construction
class DissimilarityMatrix:
    """Matrix containing distances for initial centroid determination.

    Attributes
    ----------
    size : int
        Maximum number of OPs contained.

    mut : DistanceMatrix
        The DistanceMatrix used for distance calculation and storage.
    """

    # Initializes DM based on size
    def __init__(self, size: int, mut: DistanceMatrix):
        self.size = size
        self.matrix = np.zeros((size, size))
        self.mut = mut
        self.OPs = []

    # Checks a new OP against OP's in DM to see if it should be added to DM
    def add_OP(self, new_op: OrderParameter) -> None:
        """Adds OPs to the matrix if they should be added.
        OPs should be added when there are fewer OPs than self.size
        or if they can increase the geometric mean of distances
        by being swapped with a different OP.

        Parameters
        ----------
        OP : OrderParameter
            OP to potentially be added.

        Returns
        -------
        None
            self.matrix is updated directly without a return.

        """

        self_distance = self.mut.distance(new_op, new_op)
        assert self_distance < 1e-10, "The dissimilarity between an OP and itself is not close to zero."

        if len(self.OPs) == self.size:  # matrix is full, check for swaps

            # distance of the new OP to all existing OPs
            mut_info = np.array([self.mut.distance(new_op, i) for i in self.OPs])

            # product of distances between all existing OPs, excluding the diagonal element (Eqn. 3)
            # Excluding the diagonal element 0 is the same as filling them with 1
            # Taking the power of 1/(|S|-1) is not necessary for comparison purposes
            product = np.prod(self.matrix + np.eye(self.size), axis=1)

            product_row = product_without_self(mut_info)   # Leetcode 238

            diff = product - product_row
            n = np.argmin(diff)

            if diff[n] < 0:
                mut_info[n] = self_distance
                self.matrix[n, :] = mut_info
                self.matrix[:, n] = mut_info
                self.OPs[n] = new_op  # Update the list of OPs

        else:  # adding an OP when there are fewer than self.size

            mut_info = np.zeros(self.size)

            n = len(self.OPs)
            for i, op in enumerate(self.OPs):
                mut_info[i] = self.mut.distance(new_op, op)
            mut_info[n] = self_distance

            self.matrix[n, :] = mut_info
            self.matrix[:, n] = mut_info

            self.OPs.append(new_op)


# Running clustering on `ops` starting with `seeds`
def cluster(ops: list[str], seeds: list[str], mut: DistanceMatrix) -> list[str]:
    """Clusters OPs startng with centroids from a DissimilarityMatrix.

    Parameters
    ----------
    ops : list of OrderParameters
        All OPs.

    seeds : list of OrderParameters
        Starting centroids from a DissimilarityMatrix.

    mut : DistanceMatrix
        The DistanceMatrix used for distance calculation and storage.

    Returns
    -------
    centers : list of OPs
        Final centroids after clustering.
    """

    # we will only keep track of the indices of the centers
    centers = np.array([ops.index(c) for c in seeds])
    new_centers = np.zeros_like(centers)

    max_iteration = 500
    it = 0

    while np.any(set(centers) != set(new_centers)):

        it += 1
        if it > max_iteration:
            print("Maximum iteration reached.")
            break

        # first put in all cluster centers
        group = np.full(len(ops), -1)
        for i, s in enumerate(centers):
            group[s] = i

        # copy the centers
        centers = np.array(new_centers)

        # Assign all OP to nearest cluster
        for i, op in enumerate(ops):

            # technically this if statement is redudant as or any cluster center the distance is 0.
            # I felt like adding this as you never know if there are other OPs close enough, and with
            # the numerical precision, there will end up having a few empty clusters.
            if group[i] == -1:
                dist = [mut.distance(op, ops[c]) for c in centers]
                group[i] = np.argmin(dist)

        # Calculates the new centroid minimizing distortion for a set of OPs.
        for i, _ in enumerate(centers):

            # index of all members in my center
            my_group = np.nonzero(group == i)[0]

            distortion = [np.sum([mut.distance(ops[g], ops[j]) ** 2 for j in my_group]) for g in my_group]
            new_centers[i] = my_group[np.argmin(distortion)]

    return [ops[i] for i in centers]


# This is the general workflow for AMINO
def find_ops(all_ops: list[OrderParameter] | None = None,
             distance_matrix: np.ndarray | None = None,
             names: list[str] | None = None,
             max_outputs: int = 20,
             bins: int = 20,
             distortion_filename: str = None,
             verbose: bool = True) -> list[OrderParameter]:
    """Main function performing clustering and finding the optimal number of OPs.

    Parameters
    ----------
    all_ops : list of OrderParameters
        All OPs for clustering.

    max_outputs : int
        The maximum number of clusters/centroids.

    bins : int or None
        Number of values along each axis for the joint probability.
        The probability will be a bins x bins grid.
        If None this is set with a rule of thumb.

    distortion_filename : str or None
        The filename to save distortion jumps.

    verbose : bool
        If True, print out the progress.

    Returns
    -------
    list of OPs
        The centrioids for the optimal clustering.
    """

    # selecting bins
    if bins is None:
        try:
            bins = np.ceil(np.sqrt(len(all_ops[0].traj)))
        except NameError:
            bins = 50
            print("Using default bins = 50.")

    start = timer()
    # If a disance matrix is provided, use it
    if distance_matrix is not None:
        if names is None:
            raise ValueError("Names must be provided when using a distance matrix.")
        mut = DistanceMatrix.from_matrix(distance_matrix, names)
        if verbose:
            print("Using provided distance matrix.")
    else:
        # Otherwise, create a new distance matrix
        if all_ops is None:
            raise ValueError("All OPs must be provided when not using a distance matrix.")
        mut = DistanceMatrix(bins)
        mut.initialize_distances(all_ops, verbose=verbose)
        names = [op.name for op in all_ops]
        if verbose:
            print(f"DM construction time: {timer() - start:.2f} s")

    num_array = np.arange(1, max_outputs + 1)[::-1]
    distortion_array = np.zeros_like(num_array)
    selected_op = {}

    # This loops through each number of clusters
    for f, n in enumerate(num_array):

        if verbose:
            print(f"Checking {n} order parameters...")

        # DM construction
        matrix = DissimilarityMatrix(n, mut)
        for i in names:
            matrix.add_OP(i)
        for i in names[::-1]:
            matrix.add_OP(i)

        # Clustering
        selected_op[n] = cluster(names, matrix.OPs, mut)
        distortion_array[f] = mut.distortion(selected_op[n], names)

    # Determining number of clusters
    num_ops = 0

    for dim in range(1, 11):

        neg_expo = np.array(distortion_array) ** (-0.5 * dim)
        jumps = neg_expo[:-1] - neg_expo[1:]
        min_index = np.argmax(jumps)

        if num_array[min_index] > num_ops:
            num_ops = num_array[min_index]

    if distortion_filename is not None:
        np.save(distortion_filename, distortion_array[::-1])

    return selected_op[num_ops]
