import numpy as np
import mdtraj as md

from numba import njit
from . import simulation

class rMSAAnalysis:

    def __init__(self,
                 pdb_name: list[str],
                 ref_pdb: str = None) -> None:
        '''
        Initialize the FeatureAnalysis object. This also computes the CA RMSD to the reference structure and sorts the trajectory by the RMSD.

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

        self.ca_rmsd: np.ndarray = np.array(md.rmsd(self.traj, self.ref, atom_indices=self.ref.top.select("name CA")) * 10)
        rmsd_rank = np.argsort(self.ca_rmsd)
        self.traj = md.join([self.traj[i] for i in rmsd_rank])
        self.pdb_name = [self.pdb_name[i] for i in rmsd_rank]

    def get_rmsd(self, selection: str = "name CA") -> np.ndarray:
        '''
        Get the RMSD of the atoms in the selection for each frame in the trajectory.

        :param selection: str: The selection string to use to select the atoms.
        :return: np.ndarray: The RMSD of the atoms in the selection for each frame in the trajectory.
        '''

        rmsd = md.rmsd(self.traj, self.ref, atom_indices=self.ref.top.select(selection)) * 10
        return np.array(rmsd)
    
    def drop_unphysical_structures(self, selection = "name CA", rmsd_cutoff: float = 10.0) -> np.ndarray:
        '''
        Drop structures with RMSD above the cutoff. This modifies the trajectory in place.

        :param rmsd_cutoff: The RMSD cutoff value in Angstrom. Default: 10.0 Angstrom
        :type rmsd_cutoff: float
        :param selection: The selection string to the atoms to calculate the RMSD. Default: "name CA"
        :type selection: str
        :return: The RMSD of the atoms in the selection for each frame in the trajectory.
        :rtype: np.ndarray
        '''

        rmsd = self.get_rmsd(selection)

        mask = (rmsd < rmsd_cutoff).nonzero()[0]
        try:
            assert(len(mask) > 0)
        except:
            raise ValueError(f"No structures are below the RMSD cutoff of {rmsd_cutoff} Angstrom.")
        
        self.traj = md.join([self.traj[i] for i in mask])
        self.pdb_name = [self.pdb_name[i] for i in mask]
        self.ca_rmsd = self.ca_rmsd[mask]

        return rmsd[mask]

    def rank_features(self, selection: str = "name CA") -> tuple[np.ndarray, list[set[int, int]]]:
        '''
        Rank the features by the coefficient of variation.

        :param selection: str: The selection string to use to select the atoms.
        :return feature: np.ndarray, shape=(nframes, nfeatures). The feature vector. Unit: Angstrom
        :return atom_pairs: list[tuple[int, int]]: The list of atom pairs.

        '''

        from itertools import combinations

        atom_index = self.traj.top.select(selection)
        atom_pairs = list(combinations(atom_index, 2))

        pd = md.compute_distances(self.traj, atom_pairs, periodic=False) * 10

        # sort the features by coefficient of variation
        cv = np.std(pd, axis=0)/np.mean(pd, axis=0)
        rank = np.argsort(cv)[::-1]
        cv = cv[rank]
        self.atom_pairs = [atom_pairs[i] for i in rank]
        self.feature = pd[:,rank]

        # set up a backmap to look up things from the atom pairs 
        self.feature_dict = {}
        for i, ap in enumerate(self.atom_pairs):
            self.feature_dict[frozenset(ap)] = i

        return self.feature, self.atom_pairs, cv

    def get_feature(self, atom_pairs: list[set[int, int]]) -> np.ndarray:
        '''
        Get the features for the selected atom pairs.

        :param atom_pairs: list[tuple[int, int]]: The list of atom pairs to get the features for.
        :return: np.ndarray: The features for the selected atom pairs.
        '''

        if not isinstance(atom_pairs, list):
            atom_pairs = [atom_pairs]

        indices = [self.feature_dict[ap] for ap in atom_pairs]
        return self.feature[:,indices]

    def get_feature_name(self, atom_pairs: list[set[int, int]]) -> list[str]:
        '''
        Get the names of the features.

        :param atom_pairs: The list of atom pairs to get the names for.
        :type atom_pairs: list[tuple[int, int]]
        :return: The names of the features.
        :rtype: list[str]
        '''

        if not isinstance(atom_pairs, list):
            atom_pairs = [atom_pairs]
        
        feature_names = ["" for _ in atom_pairs]
        
        for n, (i, j) in enumerate(atom_pairs):
            resname_i = str(self.traj.top.atom(i))
            resname_j = str(self.traj.top.atom(j))
            feature_names[n] = f"{resname_i}_{resname_j}"

        return feature_names

    def reduce_features(self, 
                        atom_pairs: list[set[int, int]], 
                        max_outputs: int = 20, 
                        bins: int = 50, 
                        kde_bandwidth: float = 0.02,
                        **kwargs: dict) -> list[tuple[int, int]]:
        '''
        Reduce the number of features using AMINO. Please see and cite https://doi.org/10.1039/C9ME00115H for a description of the method.

        :param n_features: int: The number features to work with. Picked using the highest coefficient of variation.
        :param max_outputs: int: The maximum number of OPs to output.
        :param bins: int: The number of bins for the histogram.
        :param kde_bandwidth: float: The bandwidth for the KDE.
        :param kwargs: dict: Additional keyword arguments to pass to the AMINO functions.
        :return: list[tuple[int, int]]: The names of the selected features and the corresponding features.
        '''

        from . import amino

        indices = [self.feature_dict[frozenset(ap)] for ap in atom_pairs]
        # Get these feature matrices for these atom pairs
        trajs = self.feature[:,indices]

        # This is a pretty weird feature in AMINO. The original code distinguish the features by their names (a string).
        # So the only way we can incorporate AMINO in is to work around a string representation
        names = self.get_feature_name(atom_pairs)
        ops = [amino.OrderParameter(n, trajs[:,i]) for i, n in enumerate(names)]
        selected_ops = amino.find_ops(ops, max_outputs=max_outputs, bins=bins, bandwidth=kde_bandwidth, verbose=False, **kwargs)

        selected_name = [op.name for op in selected_ops]
        selected_ap = [atom_pairs[names.index(n)] for n in selected_name]

        return selected_ap

    def get_chimera_plotscript(self, atom_pairs: list[set[int, int]] = None) -> str:
        '''
        Generate a Chimera plotscript to visualize the selected features.

        :param labels: list[str]: The names of the features to visualize.
        :return: str: The Chimera plotscript.
        '''

        plotscript = f"open {self.ref_pdb}\n"
        for i, j in atom_pairs:
            resid_i = self.traj.top.atom(i).residue.index
            resid_j = self.traj.top.atom(j).residue.index
            atom_name_i = self.traj.top.atom(i).name
            atom_name_j = self.traj.top.atom(j).name
            plotscript += f"distance :{resid_i + 1}@{atom_name_i} :{resid_j + 1}@{atom_name_j}\n"

        return plotscript

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

        feature_dimensions = self.feature.shape[1]
        if n_features is None:
            n_features = self.n_features
        elif n_features > feature_dimensions:
            raise ValueError(f"Number of features {n_features} is greater than the number of features in the dataset {feature_dimensions}.")
        
        z = self.feature[:,:n_features]
        npoints, d = z.shape

        # Reshuffle the data with a random permutation, but keep the first element fixed
        p = np.hstack((0, np.random.RandomState(seed=randomseed).permutation(npoints - 1) + 1))
        data = z[p]

        # The first element is always a cluster center
        center_id = np.full(max_centers, -1)
        center_id[0] = p[0]

        i = 1
        ncenter = 1
        while i < npoints:

            x_active = data[i:i+batch_size]

            # All indices of points that are at least min_dist away from all cluster centers
            center_list = data[center_id[center_id != -1]]
            distances = self._get_distance_to_center(center_list, x_active)
            indice = np.nonzero(np.all((distances > min_dist), axis=0))[0]

            if len(indice) > 0:

                # the first element will be added as cluster center
                center_id[ncenter] = p[i + indice[0]]
                ncenter += 1
                i += indice[0] + 1
            else:
                i += batch_size
            if ncenter >= max_centers:
                raise ValueError(f"{i}/{npoints} clustered. \
                                 Exceeded the maximum number of cluster centers {max_centers}. \
                                 Please increase min_dist.")
        
        center_id = center_id[center_id != -1]

        return center_list, center_id

    def get_openmm_reporter(self, 
                            n_features: int,
                            file: str = "COLVAR.dat", 
                            reportInterval: int = 100,
                            ) -> simulation.CVReporter:
        '''
        Generate a CVReporter object to write the features to a file.

        :param file: str: The name of the file to write the CVs to. Default: COLVAR.dat
        :param reportInterval: int: The interval at which to write the CVs. Default: 100
        :param list_of_indexes: list[tuple[int, int]]: The list of indexes to calculate the CVs. Default: None
        :return: None
        '''

        feature_dimensions = len(self.feature.keys())
        if n_features is None:
            n_features = self.n_features
        elif n_features > feature_dimensions:
            raise ValueError(f"Number of features {n_features} is greater than the number of features in the dataset {feature_dimensions}.")
        
        label = self.label[-n_features:]

        return simulation.CVReporter(file, reportInterval, list_of_indexes)
