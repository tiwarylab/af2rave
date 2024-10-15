"""AMINO: generating a minimally redundant set of order parameters through
clustering of mutual information based distances. Method by Ravindra, Smith,
and Tiwary. Code maintained by Ravindra and Smith. Tested and reorganized by
Da Teng in September 2024.

This is the kernel density estimation version.

Read and cite the following when using this method:
https://doi.org/10.1039/C9ME00115H
"""

import numpy as np
from sklearn.neighbors import KernelDensity
from numpy.typing import ArrayLike
import multiprocessing as mp
from itertools import combinations_with_replacement

from timeit import default_timer as timer


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
        self.traj = np.array(traj).reshape([-1, 1])/np.std(traj)

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

    bandwidth : float
        Bandwidth parameter for kernel denensity estimation.

    kernel : str
        Kernel name for kernel density estimation.

    """

    def __init__(self, bins: int, bandwidth: float, kernel: str, weights: ArrayLike = None):
        self.memo = {}
        self.bins = bins
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.weights = weights

    def initialize_distances(self, ops: list[OrderParameter]) -> None:

        self.ops = ops
        n = len(ops)

        with mp.Pool(processes=mp.cpu_count()) as p:
            result = p.starmap(self._distance_kernel, combinations_with_replacement(range(n), 2))

        for (i, j), r in zip(combinations_with_replacement(range(n), 2), result):
            idx = frozenset((self.ops[i].name, self.ops[j].name))
            self.memo[idx] = r

    # Binning two OP's in 2D space
    def _d2_bin(self, x, y):
        """ Calculate a joint probability distribution for two trajectories.

        Parameters
        ----------
        x : np.array
            Trajectory of first OP.

        y : np.array
            Trajcetory of second OP.

        Returns
        -------
        p : np.array
            self.bins by self.bins array of joint probabilities from KDE.

        """

        KD = KernelDensity(bandwidth=self.bandwidth, kernel=self.kernel)
        KD.fit(np.column_stack((x, y)), sample_weight=self.weights)

        grid1 = np.linspace(np.min(x), np.max(x), self.bins)
        grid2 = np.linspace(np.min(y), np.max(y), self.bins)
        mesh = np.meshgrid(grid1, grid2)

        data = np.column_stack((mesh[0].reshape(-1, 1), mesh[1].reshape(-1, 1)))
        samp = KD.score_samples(data)
        samp = samp.reshape(self.bins, self.bins)
        p = np.exp(samp)/np.sum(np.exp(samp))

        return p

    # Checks if distance has been computed before, otherwise computes distance
    def distance(self, OP1: OrderParameter, OP2: OrderParameter) -> float:
        """Returns the mutual information distance between two OPs.

        Parameters
        ----------
        OP1 : OrderParameter
            The first order parameter for distance calculation.

        OP2 : OrderParameter
            The second order parameter for distance calculation.

        Returns
        -------
        float
            The mutual information distance.

        """

        idx = frozenset((OP1.name, OP2.name))
        try:
            d = self.memo[idx]
        except KeyError:
            raise ValueError(f"Distance between {OP1.name} and {OP2.name} not found.")

        return d

    def _distance_kernel(self, i: int, j: int) -> float:
        '''
        Calculates the mutual information distance between the i-th and j-th OP,
        given the joint distribution.
        '''

        p_xy = self._d2_bin(self.ops[i].traj, self.ops[j].traj)

        p_x = np.sum(p_xy, axis=1)
        p_y = np.sum(p_xy, axis=0)

        log_p_x_times_p_y = np.ma.log(np.tensordot(p_x, p_y, axes=0))
        log_p_xy = np.ma.log(p_xy)

        info = np.sum(p_xy * (log_p_xy - log_p_x_times_p_y))
        entropy = np.sum(-1 * p_xy * log_p_xy)

        distance = max(0.0, (1 - (info / entropy)))

        return distance

    def distortion(self, centers: list[OrderParameter], ops: list[OrderParameter]) -> float:
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
        assert self_distance < 1e-10, f"The dissimilarity between an OP and itself is not close to zero. Decrease bandwidth. Current: {self.mut.bandwidth:.4f}"

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
def cluster(ops: list[OrderParameter], seeds: list[OrderParameter], mut: DistanceMatrix) -> list[OrderParameter]:
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

    while np.any(set(centers) != set(new_centers)):

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
def find_ops(all_ops: list[OrderParameter],
             max_outputs: int = 20,
             bins: int = 20,
             bandwidth: float = None,
             kernel: str = 'epanechnikov',
             distortion_filename: str = None,
             weights: ArrayLike = None,
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

    bandwidth : float or None
        Bandwidth parameter for kernel denensity estimation.
        If None this is set with a rule of thumb.

    kernel : str
        Kernel name for kernel density estimation.
        It is recommended to use either epanechnikov (parabolic) or gaussian.
        These are currently the only two implemented in bandwidth rule of thumb.

    distortion_filename : str or None
        The filename to save distortion jumps.

    verbose : bool
        If True, print out the progress.

    Returns
    -------
    list of OPs
        The centrioids for the optimal clustering.
    """

    # selecting bandwidth
    if bandwidth is None:
        if kernel == 'parabolic':
            kernel = 'epanechnikov'
        if kernel == 'epanechnikov':
            bw_constant = 2.2
        else:
            bw_constant = 1

        n = np.shape(all_ops[0].traj)[0]
        bandwidth = bw_constant * n ** (-1/6)
        print('Selected bandwidth: ' + str(bandwidth) + '\n')

    # selecting bins
    if bins is None:
        bins = np.ceil(np.sqrt(len(all_ops[0].traj)))
        print(f"Using {bins} bins for KDE.")

    start = timer()
    mut = DistanceMatrix(bins, bandwidth, kernel, weights)
    mut.initialize_distances(all_ops)
    print(f"DM construction time: {timer()-start:.2f} s")

    num_array = np.arange(1, max_outputs + 1)[::-1]
    distortion_array = np.zeros_like(num_array)
    selected_op = {}

    # This loops through each number of clusters
    for f, n in enumerate(num_array):

        if verbose:
            print(f"Checking {n} order parameters...")

        # DM construction
        matrix = DissimilarityMatrix(n, mut)
        for i in all_ops:
            matrix.add_OP(i)
        for i in all_ops[::-1]:
            matrix.add_OP(i)

        # Clustering
        selected_op[n] = cluster(all_ops, matrix.OPs, mut)
        distortion_array[f] = mut.distortion(selected_op[n], all_ops)

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
