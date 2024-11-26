'''Feature analysis module for AF2RAVE'''
import os
import glob
from natsort import natsorted
import numpy as np
import mdtraj as md
from pathlib import Path

from .feature import Feature
from numpy.typing import NDArray

class FeatureSelection(object):

    def __init__(self,
                 pdb_name: str | list[str],
                 ref_pdb: str = None) -> None:
        '''
        Initialize the FeatureSelection object.

        :param pdb_name: The name(s) of the PDB file from reduced MSA.
        :type pdb_name: list[str]
        :param ref_pdb: The name of the reference structure.
            If none is provided, the first frame of the input PDB file will be used as the reference.
        :type ref_pdb: str
        :param align_by: str: The selection string used to align the trajectories.
        '''

        if not isinstance(pdb_name, list):
            p = Path(pdb_name)
            if p.is_dir():   # If a folder is provided, load all in the folder
                self.pdb_name = natsorted(glob.glob(p / "*.pdb"))
            else:
                self.pdb_name = [pdb_name]
        else:
            self.pdb_name = pdb_name

        if ref_pdb is None:
            self.ref_pdb = self.pdb_name[0]
        else:
            self.ref_pdb = ref_pdb
        self.ref = md.load(self.ref_pdb)

        # MDtraj objects
        self.traj = md.load(self.pdb_name)

        # these two will be populated by the rank_features method
        self.names = []
        self.features = {}

    @property
    def feature_array(self) -> NDArray:
        return np.array([self.features[fn].ts for fn in self.names]).T

    def __len__(self) -> int:
        return len(self.pdb_name)

    # ===== Preprocessing =====

    def get_rmsd(self, selection: str = "name CA") -> NDArray:
        '''
        Get the RMSD of the atoms in the selection for each frame in the trajectory.

        :param selection: str: The selection string to use to select the atoms.
        :return: np.ndarray: RMSD. Unit: Angstrom
        :rtype: np.ndarray
        '''

        # get the atom indices from selection and check if it is valid
        try:
            atom_indices = self.ref.top.select(selection)
        except:
            raise ValueError("Selection is invalid.")
        assert len(atom_indices) > 1, f"Selection does not contain enough atoms ({len(atom_indices)})."

        rmsd = md.rmsd(self.traj, self.ref, atom_indices=atom_indices) * 10
        return np.array(rmsd)

    def filter_by_rmsd(self, selection="name CA", rmsd_cutoff: float = 10.0) -> NDArray:
        '''
        Filter structures with a RMSD cutoff.

        Remove structures that are too irrelavant by dropping those with RMSD
        larger than a cutoff (in Angstrom). This modifies the trajectory in place.

        :param rmsd_cutoff: The RMSD cutoff value. Default: 10.0 Angstrom
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

    # ===== Feature selection =====
    
    def _get_atom_index_from_selection(self, selection: str) -> NDArray:
        '''
        Get the atom indices from the selection string.

        :param selection: str: The selection string.
        :return: np.ndarray: The atom indices.
        '''

        try:
            atom_index = self.traj.top.select(selection)
        except:
            raise ValueError("Selection is invalid.")
        return atom_index
    
    def _get_atom_pairs(self, selection: str | tuple[str, str]) -> NDArray:
        '''
        Get the atom pairs from the selection string.

        :param selection: str: The selection string.
        :return: np.ndarray: The atom pairs.
        '''

        from itertools import combinations, product

        if isinstance(selection, str):
            atom_index = self._get_atom_index_from_selection(selection)
            if len(atom_index) < 2:
                raise ValueError(f"Selection '{selection}' does not contain enough atoms ({len(atom_index)}).")
            atom_pairs = np.array(list(combinations(atom_index, 2)))
        elif isinstance(selection, tuple):
            if len(selection) != 2:
                raise ValueError("Selection must be a tuple of two strings.")
            a, b = selection
            idx_a = self._get_atom_index_from_selection(a)
            idx_b = self._get_atom_index_from_selection(b)
            if len(idx_a) == 0 or len(idx_b) == 0:
                raise ValueError(f"Selection '{selection}' does not contain enough atoms.")
            atom_pairs = np.array(list(product(idx_a, idx_b)))
        else:
            raise ValueError("Selection must be a string or a tuple of two strings.")
        return atom_pairs

    def rank_feature(self,
                     selection: str | tuple[str] | list[str | tuple[str]] = "name CA"
                     ) -> tuple[dict[Feature], list[str], NDArray]:
        '''
        Rank the features by the coefficient of variation.

        :param selection: str: The selection string to use to select the atoms.
        :return features: np.ndarray, shape=(nframes, nfeatures). The feature vector. Unit: Angstrom
        :return names: list[str], The list of feature names.
        :return cv: list[float], list of coefficient of variance
        '''

        if isinstance(selection, (str, tuple)):
            atom_pairs = self._get_atom_pairs(selection)
        elif isinstance(selection, list):
            atom_pairs = np.vstack([self._get_atom_pairs(s) for s in selection])
        else:
            raise ValueError("Selection must be a string, a tuple of two strings, or a list of them.")

        pw_dist = md.compute_distances(self.traj, atom_pairs, periodic=False) * 10

        names = ["" for _ in atom_pairs]
        for i, ap in enumerate(atom_pairs):
            f = Feature(ap, self.ref.top, pw_dist[:, i])
            self.features[f.name] = f
            names[i] = f.name
        self.names += names

        # sort the features by coefficient of variation
        cv = np.std(pw_dist, axis=0) / np.mean(pw_dist, axis=0)
        rank = np.argsort(cv)[::-1]
        names = [names[i] for i in rank]

        return names, cv[rank]

    # ===== AMINO interface =====

    def amino(self,
              feature_name: list[str],
              max_outputs: int = 20,
              bins: int = 50,
              kde_bandwidth: float = 0.02,
              **kwargs: dict
              ) -> list[str]:
        '''
        Reduce the number of features using AMINO.

        Please see and cite https://doi.org/10.1039/C9ME00115H for a description of the method.

        :param n_features: int: The number features to work with. Picked using the highest coefficient of variation.
        :param max_outputs: int: The maximum number of OPs to output.
        :param bins: int: The number of bins for the histogram.
        :param kde_bandwidth: float: The bandwidth for the KDE.
        :param kwargs: dict: Additional keyword arguments to pass to the AMINO functions.
        :return: list[str]: The names of the selected features and the corresponding features.
        '''

        from .. import amino

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

    # ===== Format conversion =====

    def get_chimera_plotscript(self,
                               feature_name: list[str] = None,
                               add_header: bool = True
                               ) -> str:
        '''
        Generate a Chimera plotscript to visualize the selected features.

        :param labels: list[str]: The names of the features to visualize.
        :param add_header: bool: Add the "open xxx.pdb" header to the plotscript.
        :return: str: The Chimera plotscript.
        '''

        plotscript = ""
        for fn in feature_name:
            plotscript += self.features[fn].get_plot_script()

        # reduce redudancy
        plotscript = list(set(plotscript.strip().split("\n")))
        plotscript = "\n".join(plotscript) + "\n"

        # add header
        if add_header:
            plotscript = f"open {self.ref_pdb}\n" + plotscript

        return plotscript

    def get_index(self, feature_name: list[str]) -> list[set[int]]:
        '''
        Get the atom indices of features by their names.
        '''

        try:
            index = [self.features[fn].ap for fn in feature_name]
        except KeyError as e:
            raise ValueError(f"Feature {e} does not exist.") from e

        return index

    # ===== Clustering =====

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

        # Reshuffle the data with a random permutation, but keep the first element/reference fixed
        if self.ref_pdb in self.pdb_name:
            idx = self.pdb_name.index(self.ref_pdb)
        else: 
            idx = 0
        p = np.random.RandomState(seed=randomseed).permutation(npoints)
        lookup = (np.arange(npoints) + np.where(p == idx)[0][0]) % npoints
        p = p[lookup]
        data = z[p]

        # The first element is always a cluster center
        center_id = np.full(max_centers, -1)
        center_id[0] = p[0]

        i = 1
        ncenter = 1
        ndim = data.shape[1]
        while i < npoints:

            x_active = data[i:i + batch_size]

            # All indices of points that are at least min_dist away from all cluster centers
            center_list = data[center_id[center_id != -1]]
            distances = np.linalg.norm(x_active[:, np.newaxis, :] - center_list[np.newaxis, :, :], axis=2) / np.sqrt(ndim)
            indice = np.nonzero(np.all((distances > min_dist).reshape(ncenter, -1), axis=0))[0]

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

    def pca(self, n_components: int = 2, **kwargs):
        '''
        Perform PCA on the selected features.

        :param n_components: The number of output principle components
        :type n_components: int
        :param kwargs: Additional keyword arguments to pass to the PCA constructor.
        :return pca, z: The PCA object and the transformed data.
        '''

        from sklearn.decomposition import PCA

        z = self.feature_array
        pca = PCA(n_components=n_components, **kwargs)
        pca.fit(z)
        return pca, pca.transform(z)
