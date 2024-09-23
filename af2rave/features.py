import numpy as np
import mdtraj as md
from functools import cached_property

from numba import njit
from numpy.typing import ArrayLike


class Feature(object):

    def __init__(self, atom_pair: set[int], top: md.Topology, ts: ArrayLike):
        n_atoms = len(atom_pair)
        if n_atoms != 2:
            raise ValueError(f"Accept only two atoms as atom_pairs. Got {n_atoms}")
        else:
            self._ap = atom_pair
        self._top = top
        self._ts = ts

    @cached_property
    def name(self):
        i, j = self._ap
        resname_i = str(self._top.atom(i))
        resname_j = str(self._top.atom(j))
        return f"{resname_i}_{resname_j}"

    @property
    def ts(self):
        return self._ts

    @cached_property
    def mean(self):
        return np.mean(self._ts)

    @cached_property
    def std(self):
        return np.std(self._ts)

    @cached_property
    def cv(self):
        return self.std/self.mean

    @cached_property
    def __len__(self):
        return self._ts.size

    def __eq__(self, value) -> bool:
        return self._ap == value._ap

    def get_plot_script(self):

        script = ""

        i, j = self._ap
        resid_i = self._top.atom(i).residue.index + 1
        resid_j = self._top.atom(j).residue.index + 1
        atom_name_i = self._top.atom(i).name
        atom_name_j = self._top.atom(j).name

        script += f"distance :{resid_i}@{atom_name_i} :{resid_j}@{atom_name_j}\n"

        if atom_name_i != "CA":
            script += f"show :{resid_i} a\n"
        if atom_name_j != "CA":
            script += f"show :{resid_j} a\n"

        return script


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

        ca_rmsd = self.get_rmsd("name CA")
        rmsd_rank = np.argsort(ca_rmsd)
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

    def drop_unphysical_structures(self, selection="name CA", rmsd_cutoff: float = 10.0) -> np.ndarray:
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
        assert len(mask) > 0, f"No structures are below the RMSD cutoff of {rmsd_cutoff} Angstrom."

        self.traj = md.join([self.traj[i] for i in mask])
        self.pdb_name = [self.pdb_name[i] for i in mask]

        return rmsd[mask]

    def rank_features(self, selection: str = "name CA") -> tuple[dict[Feature], list[str], np.array]:
        '''
        Rank the features by the coefficient of variation.

        :param selection: str: The selection string to use to select the atoms.
        :return features: np.ndarray, shape=(nframes, nfeatures). The feature vector. Unit: Angstrom
        :return names: list[str], The list of feature names.
        :return cv: list[float], list of coefficient of variance
        '''

        from itertools import combinations

        atom_index = self.traj.top.select(selection)
        atom_pairs = list(combinations(atom_index, 2))

        pw_dist = md.compute_distances(self.traj, atom_pairs, periodic=False) * 10
        print(pw_dist.shape)

        self.features = {}
        self.names = ["" for _ in atom_pairs]
        for i, ap in enumerate(atom_pairs):
            f = Feature(ap, self.traj.top, pw_dist[:, i])
            self.features[f.name] = f
            self.names[i] = f.name

        # sort the features by coefficient of variation
        cv = np.std(pw_dist, axis=0)/np.mean(pw_dist, axis=0)
        rank = np.argsort(cv)[::-1]
        self.names = [self.names[i] for i in rank]

        return self.features, self.names, cv[rank]

    def reduce_features(self,
                        feature_name: list[str],
                        max_outputs: int = 20,
                        bins: int = 50,
                        kde_bandwidth: float = 0.02,
                        **kwargs: dict) -> list[str]:
        '''
        Reduce the number of features using AMINO. Please see and cite https://doi.org/10.1039/C9ME00115H for a description of the method.

        :param n_features: int: The number features to work with. Picked using the highest coefficient of variation.
        :param max_outputs: int: The maximum number of OPs to output.
        :param bins: int: The number of bins for the histogram.
        :param kde_bandwidth: float: The bandwidth for the KDE.
        :param kwargs: dict: Additional keyword arguments to pass to the AMINO functions.
        :return: list[str]: The names of the selected features and the corresponding features.
        '''

        from . import amino

        # This is a pretty weird feature in AMINO. The original code distinguish
        # the features by their names (a string). So the only way we can incorporate
        # AMINO in is to work around a string representation
        ops = [amino.OrderParameter(n, self.features[n].ts) for n in feature_name]
        selected_ops = amino.find_ops(ops,
                                      max_outputs=max_outputs,
                                      bins=bins,
                                      bandwidth=kde_bandwidth,
                                      verbose=False,
                                      **kwargs)
        selected_name = [op.name for op in selected_ops]
        return selected_name

    def get_chimera_plotscript(self, feature_name: list[str] = None) -> str:
        '''
        Generate a Chimera plotscript to visualize the selected features.

        :param labels: list[str]: The names of the features to visualize.
        :return: str: The Chimera plotscript.
        '''

        plotscript = f"open {self.ref_pdb}\n"
        for fn in feature_name:
            plotscript += self.features[fn].get_plot_script()

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
                                 feature_name: list[str],
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

        z = np.array([self.features[fn].ts for fn in feature_name]).T
        npoints = z.shape[0]

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

    @property
    def feature_array(self):
        return np.array([self.features[fn].ts for fn in self.names]).T

    def pca(self, n_components: int = 2, **kwargs):
        '''
        Perform PCA on the selected features.

        :param n_components: int: The number of components to keep.
        :return: np.ndarray: The PCA components.
        '''

        from sklearn.decomposition import PCA

        z = np.array([self.features[fn].ts for fn in self.names]).T
        pca = PCA(n_components=n_components, **kwargs)
        pca.fit(z)
        return pca, pca.transform(z)
