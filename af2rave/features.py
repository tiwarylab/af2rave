import numpy as np
import mdtraj as md

from numba import njit
from itertools import combinations

from numpy.typing import ArrayLike

class rMSAAnalysis:

    def __init__(self,
                 pdb_name: list[str],
                 ref_pdb: str = None) -> None:
        '''
        Initialize the rMSAAnalysis object.

        :param pdb_name: ArrayLike[str]: The name(s) of the PDB file from reduced MSA.
        :param ref_pdb: str: The name of the reference structure. If none is provided, the first frame of the input PDB file will be used as the reference.
        :param align_by: str: The selection string used to align the trajectories.
        '''

        # Store the pdb name as labels
        self.pdb_name = pdb_name
        if ref_pdb is None:
            ref_pdb = pdb_name[0]
        else:
            self.ref_pdb = ref_pdb

        # MDtraj objects
        self.traj = md.load(pdb_name)
        self.ref = md.load(ref_pdb)

    def get_rmsd(self, selection: str = "name CA") -> np.ndarray:
        '''
        Get the RMSD of the atoms in the selection for each frame in the trajectory.

        :param selection: str: The selection string to use to select the atoms.
        :return: np.ndarray: The RMSD of the atoms in the selection for each frame in the trajectory.
        '''

        self.rmsd = md.rmsd(self.traj, self.ref, atom_indices=self.ref.top.select(selection)) * 10
        
        return self.rmsd
    
    def drop_unphysical_structures(self, rmsd_cutoff: float = 4.5) -> None:
        '''
        Drop structures with RMSD above the cutoff. This modifies the trajectory in place.

        :param rmsd_cutoff: float: The RMSD cutoff value in Angstrom.
        :return: None
        '''

        self.traj = self.traj[np.where(self.rmsd < rmsd_cutoff)]

    def select_features(self, 
                        selection: str = "name CA", 
                        n_features: int = 100,
                        return_all: bool = False) -> None:

        atom_index = self.traj.top.select(selection)

        name = [str(self.traj.top.atom(i)) for i in atom_index]
        coords = np.array(self.traj.xyz[:, atom_index, :])

        pd, label = self._pairwise_distance(coords, name)

        cv = np.std(pd, axis=0)/np.mean(pd, axis=0)
        rank = np.argsort(cv)
        self.feature = pd[:,rank]
        self.label = [label[i] for i in rank]
        self.n_features = n_features

        if return_all:
            return self.feature, self.label
        else:
            return self.feature[:,-n_features:], self.label[-n_features:]
    
    def reduce_features(self, 
                        n_features: int, 
                        max_outputs: int = 20, 
                        bins: int = 30, 
                        kde_bandwidth: float = 0.02,
                        **kwargs: dict) -> tuple[list[str], np.ndarray]:
        '''
        Reduce the number of features using AMINO. Please see and cite https://doi.org/10.1039/C9ME00115H for a description of the method.

        :param n_features: int: The number features to work with. Picked using the highest coefficient of variation.
        :param max_outputs: int: The maximum number of OPs to output.
        :param bins: int: The number of bins for the histogram.
        :param kde_bandwidth: float: The bandwidth for the KDE.
        :param kwargs: dict: Additional keyword arguments to pass to the AMINO functions.
        :return: tuple(list[str], np.ndarray): The names of the selected features and the corresponding features.
        '''

        from . import amino

        names = self.label[-n_features:]
        trajs = self.feature[:,-n_features:]

        ops = [amino.OrderParameter(n, trajs[i]) for i, n in enumerate(names)]
        final_ops = amino.find_ops(ops, max_outputs=max_outputs, bins=bins, bandwidth=kde_bandwidth, verbose=False, **kwargs)

        op_names = [op.name for op in final_ops]
        op_index = [names.index(n) for n in op_names]
        op_features = trajs[:,op_index]

        return op_names, op_features

    def get_chimera_plotscript(self, labels: list[str] = None) -> str:
        '''
        Generate a Chimera plotscript to visualize the selected features.

        :param labels: list[str]: The names of the features to visualize.
        :return: str: The Chimera plotscript.
        '''

        import re

        plotscript = f"open {self.ref_pdb}\n"
        for l in labels:
            resid_i, resid_j = re.findall(r"\d+", l)
            atom_i, atom_j = re.findall(r"-\w+", l)
            plotscript += f"distance :{resid_i}@{atom_i[1:]} :{resid_j}@{atom_j[1:]}\n"

        return plotscript

    @staticmethod
    @njit
    def _pairwise_distance(coord: np.ndarray, atom_name: list[str]) -> np.ndarray:
        '''
        Calculate pairwise distances between all atoms in a frame.
        
        :param coord: np.ndarray, shape=(nframes, natoms, 3)
        :return: np.ndarray, shape=(nframes, natoms*(natoms-1)//2)
        '''

        nframes = coord.shape[0]
        natoms = coord.shape[1]
        pairwise_distances = np.zeros((nframes, natoms*(natoms-1)//2))

        label = ["" for i in range(natoms*(natoms-1)//2)]

        idx = 0
        for j in range(natoms):
            for k in range(j+1, natoms):
                label[idx] = f"{atom_name[j]}/{atom_name[k]}"
                for i in range(nframes):
                    pairwise_distances[i, idx] = np.linalg.norm(coord[i,j] - coord[i,k])
                idx += 1

        return pairwise_distances, label

    @staticmethod
    @njit(parallel=True)
    def _get_distance_to_center(center, coord):

        ncenters = center.shape[0]
        npoints = coord.shape[0]
        ndims = center.shape[1]

        distance_mat = np.zeros((ncenters, npoints))

        for i in np.arange(ncenters):
            for j in np.arange(npoints):
                distance_mat[i, j] = np.linalg.norm(coord[j] - center[i])/np.sqrt(ndims)

        return distance_mat

    def regular_space_clustering(self, 
                                 n_features: int,
                                 min_dist: float, 
                                 max_centers: int = 100, 
                                 batch_size: int = 100, 
                                 randomseed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        '''
        Performs regular space clustering on the selected dimensions of features.

        :param n_features: int: The number of features to use for clustering.
        :param min_dist: float: The minimum distance between cluster centers.
        :param max_centers: int: The maximum number of cluster centers.
        :param batch_size: int: The number of points to process in each batch. 
        :param randomseed: int: The random seed to use for the permutation.
        :return center: np.ndarray: The cluster center coordinates.
        :return center_id: np.ndarray: The indices of the cluster centers.
        '''

        z = self.feature[:,-n_features:]
        npoints, d = z.shape

        # Reshuffle the data with a random permutation, but keep the first element fixed
        p = np.hstack((0, np.random.RandomState(seed=randomseed).permutation(npoints-1)+1))
        data = z[p]

        # The first element is always a cluster center
        center_list = data[0, :].copy().reshape(1, d)
        center_id = np.array([-1 for i in np.arange(max_centers)])
        center_id[0] = np.array(p[0]+1)

        i = 1
        ncenter = 1
        while i < npoints:

            x_active = data[i:i+batch_size]

            # All indices of points that are at least min_dist away from all cluster centers
            distances = self._get_distance_to_center(center_list, x_active)
            indice = np.nonzero(np.all((distances > min_dist/10), axis=0))[0]

            if len(indice) > 0:

                # the first element will be added as cluster center
                center_list = np.append(center_list, x_active[indice[0]][np.newaxis,:], axis=0)
                center_id[ncenter] = p[i + indice[0]] + 1
                ncenter += 1
                i += indice[0]
            else:
                i += batch_size
            if ncenter >= max_centers:
                raise ValueError(f"{i}/{npoints} clustered. \
                                 {center_id.size} centers exceeded the maximum number of cluster centers {max_centers}. \
                                 Please increase min_dist.")
        
        center_id = center_id[center_id != -1]

        return center_list, center_id
