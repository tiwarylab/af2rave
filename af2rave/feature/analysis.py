'''
Feature analysis module for af2rave.
'''

import glob
from natsort import natsorted
import numpy as np
import mdtraj as md
from pathlib import Path

from numpy.typing import NDArray
from sklearn.decomposition import PCA


class FeatureSelection:
    """
    Reads an ensemble of PDB files and performs feature selection.

    :param pdb_name: The name(s) of the PDB file from reduced MSA.
    :param ref_pdb: The name of the reference structure. If none is provided,
                    the first frame of the input PDB file will be used as the reference.
    """

    def __init__(self, pdb_name: str | list[str], ref_pdb: str | None = None) -> None:
        if isinstance(pdb_name, str):
            p = Path(pdb_name)
            self._pdb_name = (
                natsorted(glob.glob(f"{p}/*.pdb")) if p.is_dir() else [pdb_name]
            )
        else:
            self._pdb_name = pdb_name

        if not self._pdb_name:
            raise ValueError("No valid PDB files found.")

        self._ref_pdb = ref_pdb if ref_pdb else self._pdb_name[0]

        # Load reference structure
        self._ref = md.load(self._ref_pdb)
        self._top = self._ref.topology

        # Load trajectory
        self._traj = md.load(self._pdb_name)

        # Feature storage
        self._features: dict = {}
        self._atom_pairs: dict = {}

    # ===== Properties =====
    @property
    def traj(self) -> md.Trajectory:
        '''
        Return a MDTraj object of all structures.

        :return: The MDTraj object.
        :rtype: md.Trajectory
        '''
        return self._traj

    @property
    def pdb_name(self) -> list[str]:
        '''
        The list of pdb names.
        '''
        return self._pdb_name

    @property
    def ref_pdb(self) -> str:
        '''
        The reference pdb name.
        '''
        return self._ref_pdb

    @property
    def top(self) -> md.Topology:
        '''
        The topology of the reference structure.
        '''
        return self._top

    @property
    def features(self) -> dict[str, NDArray[np.float_]]:
        '''
        The features dictionary. The key is the feature name and the value is the feature array.
        '''
        return self._features

    @property
    def atom_pairs(self) -> dict[str, NDArray[np.int_]]:
        '''
        The atom pairs dictionary. The key is the feature name and the value is the atom pairs.
        '''
        return self._atom_pairs

    @property
    def feature_array(self) -> NDArray[np.float_]:
        '''
        The feature array, with each feature stacked column-wise.

        :return: The feature array.
        :rtype: np.ndarray[float]
        '''
        return np.column_stack(list(self.features.values())) if self.features else np.empty((0, 0))

    def __len__(self) -> int:
        '''
        The number of structures in the trajectory.
        '''
        return len(self.pdb_name)
    
    def __getitem__(self, key):
        try:
            idx = self._pdb_name.index(key)
        except ValueError:
            raise KeyError(f"Structure '{key}' not found in the trajectory.")
        return self._traj[idx]

    # ===== Preprocessing =====

    def _select_and_validate(self,
                             selection: str,
                             min_atoms: int | None = 1) -> NDArray[np.int_]:
        """
        Select atoms from the trajectory and validate the selection
        to ensure it contains at least `min_atoms` atoms.

        :param str selection:
            The selection string.
        :param int min_atoms:
            The minimum number of atoms required.
            Default: 1.
            Set to `None` to disable the check.
        :return: An array of atom indices.
        :raises ValueError:
            If the selection is invalid or does not contain enough atoms.
        """

        try:
            atom_indices = self._top.select(selection)
        except Exception as e:
            raise ValueError(f"Invalid selection: {selection}. Error: {e}")

        if min_atoms is not None and len(atom_indices) < min_atoms:
            raise ValueError(f"Selection '{selection}' contains only {len(atom_indices)} atoms, "
                             f"which is less than the required {min_atoms}.")

        return atom_indices

    def get_rmsd(self, selection: str = "name CA") -> dict[str, float]:
        '''
        Get the RMSD of the atoms in the selection for each frame in the trajectory.
        The reference structure is provided in the constructor.

        :param selection: str:
            The selection string to use to select the atoms.
        :return: Dictionary of pdb names and their RMSD values. Units: Angstrom.
        :rtype: dict[str, float]
        '''

        sel = self._select_and_validate(selection, 2)
        rmsd = md.rmsd(self._traj, self._ref, atom_indices=sel) * 10
        return {pdb: r for pdb, r in zip(self._pdb_name, rmsd)}

    @property
    def peptide_bond_stats(self) -> dict[str, NDArray[np.float_]]:
        '''
        Get the mean and standard deviation of the peptide bondlengths
        per structure. A dictionary with the pdb names as keys.
        '''

        atom_pairs = []
        for c in self.top.chains:
            chainid = c.index
            sel_C = self._select_and_validate(f'protein and chainid {chainid} and name C', 2)[:-1]
            sel_N = self._select_and_validate(f'protein and chainid {chainid} and name N', 2)[1:]
            atom_pairs.append(np.column_stack((sel_N, sel_C)))
        atom_pairs = np.vstack(atom_pairs)

        distances = md.compute_distances(self._traj, atom_pairs=atom_pairs) * 10  # Angstrom
        mean = distances.mean(1)
        std = distances.std(1)

        return {
            pdb: r for pdb, r in
            zip(self._pdb_name, np.column_stack((mean, std)))
        }

    @property
    def minimum_nonbonded_distance(self) -> dict[str, float]:
        '''
        The minimum non-bonded distances in the structures
        generated by AF2. A dictionary with the pdb names as keys.
        '''

        from itertools import combinations
        def get_batches(input_list, batch_size):
            input_list = input_list[:]
            for i in range(0, len(input_list), batch_size):
                yield input_list[i:i + batch_size]

        traj_noH = self._select_and_validate('not element H')
        pairs = set(combinations(traj_noH, 2))

        bonded_pairs = {(b[0].index, b[1].index) for b in self._top.bonds}
        pairs.difference_update(bonded_pairs)

        nb_pairs = list(pairs)
        min_dist = None
        # Compute distances in batches, to avoid memory issues
        for batch in get_batches(nb_pairs, 100000):
            distance = md.compute_distances(self._traj,
                                            atom_pairs=batch,
                                            periodic=False
                                            ) * 10  # Angstroms
            if min_dist is not None:
                min_dist = np.minimum(min_dist, distance.min(1))
            else:
                min_dist = distance.min(1)

        return {pdb: r for pdb, r in zip(self._pdb_name, min_dist)}

    # ===== Filtering =====

    def rmsd_filter(self, selection="name CA", rmsd_cutoff: float = 10.0) -> list[str]:
        '''
        Filter structures with a RMSD cutoff.

        Filter structures that are too irrelavant by dropping those with RMSD
        larger than a cutoff (in Angstrom). This returns a list of pdb names.
        The filter can be subsequently applied by the apply_filter method.

        :param float rmsd_cutoff:
            The RMSD cutoff value. Default: 10.0 Angstrom
        :param str selection:
            The selection string to the atoms to calculate the RMSD.
            Default: "name CA"
        :return: The pdb names of the selected structures
        :rtype: list[str]
        :raises ValueError: If no structures meet the cutoff criteria.
        '''

        rmsd = self.get_rmsd(selection)
        mask = [k for k, v in rmsd.items() if v <= rmsd_cutoff]
        if len(mask) == 0:
            raise ValueError(f"No structures are below the RMSD cutoff of {rmsd_cutoff} Angstrom.")
        return mask

    def peptide_bond_filter(self, mean_cutoff=1.4, std_cutoff=0.2) -> list[str]:
        '''
        Filter structures with a peptide bond cutoff.

        Some AlphaFold2 generated structures have unrealistic backbone structures,
        often characterized with too long or too short peptide bonds.
        The mean and standard deviation of the peptide bond lengths are calculated for each structure.
        If the mean is larger than the cutoff, or the standard deviation
        is larger than the cutoff, the structure will be filtered out.

        :param float mean_cutoff:
            Maximum allowed mean peptide bond length per structure.
            Default: 1.4 Angstrom
        :param float std_cutoff:
            Maximum allowed standard deviation of peptide bond length per structure.
            Default: 0.2 Angstrom
        :return: The pdb names of the selected structures
        :rtype: list[str]
        :raises ValueError: If no structures meet the cutoff criteria.
        '''

        mask = [k for k, (m, s) in self.peptide_bond_stats.items()
                if m <= mean_cutoff and s <= std_cutoff
                ]
        if len(mask) == 0:
            raise ValueError("No structures are below the peptide bond cutoffs of "
                             f"mean={mean_cutoff} Angstrom and "
                             f"std={std_cutoff} Angstrom."
                             )
        return mask

    def steric_clash_filter(self, min_non_bonded_cutoff=1.0) -> list[str]:
        '''
        Filter structures based on non-bonded heavy atom distances.

        Some AlphaFold2-generated structures have steric clashes
        between non-bonded atoms. This method filters out structures
        where non-bonded heavy atom distances are too short, leading
        to overlap in van der Waals radii.

        :param float min_non_bonded_cutoff:
            Minimum allowed non-bonded heavy atom distance.
            Default: 1.0 Angstrom
        :return: The pdb names of the selected structures
        :rtype: list[str]
        :raises ValueError: If no structures meet the cutoff criteria.
        '''
        min_nb_dists = self.minimum_nonbonded_distance
        mask = [k for k, v in min_nb_dists.items() if v >= min_non_bonded_cutoff]
        if len(mask) == 0:
            raise ValueError("No structures are above the dist cutoff of "
                             f"{min_non_bonded_cutoff} Angstrom.")
        return mask

    def apply_filter(self, *args: list[str]) -> None:
        '''
        Apply a mask to the trajectory.
        Each mask is a list of strings which are pdb names to keep.
        Multiple masks can be applied at once.

        Example:
            .. code-block:: python

                fs.apply_filter(mask)
                fs.apply_filter(mask1, mask2)

        :param list[str] mask: The mask to apply.
        :raises ValueError: If the mask is invalid.
        '''

        mask = natsorted(set.intersection(*map(set, args)))

        # Check if the intersection is empty
        if len(mask) == 0:
            raise ValueError("No structures are selected by the filter.")

        # Check if the mask is valid
        exist = [m in self.pdb_name for m in mask]
        if not all(exist):
            non_exist = [m for m, e in zip(mask, exist) if not e]
            raise ValueError(f"Invalid mask. Some structures do not exist: {non_exist}")

        # Apply the mask
        idx = [self._pdb_name.index(m) for m in mask]
        self._pdb_name = mask
        self._traj = md.join([self._traj[i] for i in idx])

    # ===== Feature selection =====
    def _get_atom_pairs(self, selection: str | tuple[str, str]) -> NDArray[np.int_]:
        """
        Get the atom pairs from the selection string.

        - If `selection` is a string, it returns all pairs of atoms in the selection.
        - If `selection` is a tuple of two strings, it returns all pairs of atoms between the two selections.

        :param selection: A string representing a single selection or a tuple of two selection strings.
        :type selection: str | tuple[str, str]
        :return: A NumPy array of atom pairs.
        :raises ValueError: If `selection` is not a string or a tuple of two strings.
        """

        from itertools import combinations, product

        if isinstance(selection, str):
            atom_index = self._select_and_validate(selection, min_atoms=2)
            return np.array(list(combinations(atom_index, 2)), dtype=np.int_)

        if isinstance(selection, tuple) and len(selection) == 2:
            idx_a = self._select_and_validate(selection[0])
            idx_b = self._select_and_validate(selection[1])
            return np.array(list(product(idx_a, idx_b)), dtype=np.int_)

        raise ValueError("Selection must be a string or a tuple of two strings.")

    def rank_feature(self,
                     selection: str | tuple[str, str] | list[str | tuple[str, str]] = "name CA"
                     ) -> tuple[list[str], NDArray[np.float_]]:
        """
        Rank the features by the coefficient of variation (CV).
        The argument ``selection`` can be:

            - A `string`:
                Computes all pairs of atoms within the selection.
            - A `tuple` of two strings:
                Computes all pairs of atoms between the two selections.
            - A `list` of strings or tuples:
                Computes atom pairs for each selection in the list.

        :param selection: The selection string(s) used to determine atom pairs.
        :return:
            - names: A list of feature names.
            - cv: A NumPy array containing the coefficient of variation values.
        :raises ValueError: If `selection` is not a valid type.
        """

        from .utils import representation

        if isinstance(selection, (str, tuple)):
            atom_pairs = self._get_atom_pairs(selection)
        elif isinstance(selection, list):
            # Shape: (n_pairs, 2)
            atom_pairs = np.vstack([self._get_atom_pairs(s) for s in selection])
        else:
            raise ValueError("Selection must be a string, a tuple of two strings, or a list of them.")

        # Compute pairwise distances in nanometers, convert to Angstroms
        # Shape: (n_structures, n_pairs)
        pw_dist = md.compute_distances(self.traj, atom_pairs, periodic=False) * 10

        # Generate feature names
        names = [f"{representation(self._top, i)}-{representation(self._top, j)}"
                 for i, j in atom_pairs
                 ]

        # Store features
        for name, pwd, ap in zip(names, pw_dist.T, atom_pairs):
            self._features[name] = pwd
            self._atom_pairs[name] = ap

        # Compute coefficient of variation (CV)
        mean_dist = np.mean(pw_dist, axis=0)
        std_dist = np.std(pw_dist, axis=0)

        # Handle division errors safely
        with np.errstate(divide='ignore', invalid='ignore'):
            cv = np.where(mean_dist != 0, std_dist / mean_dist, np.nan)

        # Rank features by CV in descending order
        rank = np.argsort(cv)[::-1]
        names_sorted = [names[i] for i in rank]

        return names_sorted, cv[rank]

    # ===== Format conversion =====
    def get_chimera_plotscript(self,
                               feature_name: list[str],
                               add_header: bool = True
                               ) -> str:
        """
        Generate a Chimera plotscript to visualize the selected features.

        :param feature_name: A list of feature names to visualize.
        :param add_header: Whether to add the "open xxx.pdb" header to the plotscript.
        :return: The Chimera plotscript as a string.
        :raises ValueError: If `feature_name` is None or contains invalid names.
        """

        from .utils import chimera_representation, resid

        plotscript_lines = set()

        for fn in feature_name:
            if fn not in self._atom_pairs:
                raise ValueError(f"Feature name '{fn}' not found in stored atom pairs.")

            i, j = self._atom_pairs[fn]
            atom_i = chimera_representation(self.top, i)
            atom_j = chimera_representation(self.top, j)

            plotscript_lines.add(f"distance {atom_i} {atom_j}")

            if "CA" not in atom_i:
                plotscript_lines.add(f"show :{resid(self.top, i)} a")
            if "CA" not in atom_j:
                plotscript_lines.add(f"show :{resid(self.top, j)} a")

        plotscript = "\n".join(sorted(plotscript_lines)) + "\n"  # Sorting ensures deterministic output

        if add_header:
            plotscript = f"open {self.ref_pdb}\n{plotscript}"

        return plotscript

    # ===== Clustering =====

    def regular_space_clustering(self,
                                 feature_name: list[str],
                                 min_dist: float,
                                 max_centers: int = 100,
                                 batch_size: int = 100,
                                 randomseed: int = 0) -> tuple[NDArray[np.float_], NDArray[np.int_]]:
        """
        Performs regular space clustering on the selected dimensions of features.

        :param list[str] feature_name:
            List of feature names to use for clustering.
        :param float min_dist:
            Minimum distance between cluster centers. Unit: Angstrom.
        :param int max_centers:
            Maximum number of cluster centers. Default: 100.
        :param int batch_size:
            Number of points to process in each batch. Default: 100.
        :param int randomseed:
            Random seed for the permutation.
        :return: A tuple containing:

            - center (np.ndarray): Cluster center coordinates.
            - center_id (np.ndarray): Indices of the cluster centers.
        :raises ValueError: If `max_centers` is exceeded.
        """

        if not feature_name:
            raise ValueError("Feature list cannot be empty.")

        # Extract feature time series and transpose to shape (npoints, nfeatures)
        z = np.array([self._features[fn] for fn in feature_name], dtype=np.float_).T
        npoints, ndim = z.shape

        # Determine the reference index
        idx = self.pdb_name.index(self.ref_pdb) if self.ref_pdb in self.pdb_name else 0

        # Generate a random permutation while ensuring the reference index remains fixed
        rng = np.random.default_rng(seed=randomseed)
        perm = rng.permutation(npoints)
        lookup = (np.arange(npoints) + np.where(perm == idx)[0][0]) % npoints
        perm = perm[lookup]
        data = z[perm]

        # Initialize cluster centers
        center_id = np.full(max_centers, -1, dtype=np.int_)
        center_id[0] = perm[0]
        ncenter = 1
        i = 1

        while i < npoints:
            x_active = data[i:i + batch_size]
            current_centers = data[center_id[center_id != -1]]

            # Compute Euclidean distances normalized by sqrt(ndim)
            distances = np.linalg.norm(x_active[:, np.newaxis, :] - current_centers[np.newaxis, :, :], axis=2) / np.sqrt(ndim)

            # Find indices of points that are at least `min_dist` away from all cluster centers
            valid_indices = np.nonzero(np.all(distances > min_dist, axis=1))[0]

            if valid_indices.size > 0:
                center_id[ncenter] = perm[i + valid_indices[0]]
                ncenter += 1
                i += valid_indices[0] + 1
            else:
                i += batch_size

            if ncenter >= max_centers:
                raise ValueError(f"{i}/{npoints} clustered. "
                                 f"Exceeded the maximum number of cluster centers ({max_centers}). "
                                 f"Consider increasing `min_dist`.")

        center_id = center_id[center_id != -1]

        return center_id

    def pca(self, n_components: int = 2, **kwargs) -> tuple[PCA, NDArray[np.float_]]:
        """
        Perform Principal Component Analysis (PCA) on the selected features.

        :param n_components: The number of principal components to compute.
        :param kwargs: Additional keyword arguments to pass to the PCA constructor.
        :return: A tuple containing the fitted PCA object and the transformed data.
        :raises ValueError: If no features are available for PCA.
        """

        if not self.features:
            raise ValueError("No features available for PCA.")

        # Extract time series data from features
        z = np.array([self._features[fn] for fn in self.features], dtype=np.float_).T

        # Ensure there are enough features to compute the requested components
        if z.shape[1] < n_components:
            raise ValueError(f"Number of components ({n_components}) cannot exceed available features ({z.shape[1]}).")

        # Perform PCA
        pca = PCA(n_components=n_components, **kwargs)
        transformed_data = pca.fit_transform(z)

        return pca, transformed_data
