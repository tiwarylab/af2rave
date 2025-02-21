Feature selection module
========================

This module uses a set of PDB structures for the given protein to select the most relevant features.


The analysis usually starts with intializing a :class:`af2rave.feature.FeatureSelection` object. 
The object takes arguments of either a directory, a list of PDB files, or a combination of those.
An option ``ref_pdb`` can be used to specify a reference PDB file to align the structures to. 
If none is provided, the first PDB file in the list will be used as the reference.
If a directory name is provided, a natural sort is performed to the filenames in the folder.

.. code-block:: python3

    import af2rave.feature
    fs = af2rave.feature.FeatureSelection("path/to/pdb/files")

The second step is to apply filters. The set of implemented filters currently includes two chemistry-based filters and one RMSD filter. Under chemistry-based filters, the method ``peptide_bond_filter()`` ensures that the residues in a single chain follow peptide bond lengths, and the method ``steric_clash_filter()`` filters out the structures that contain steric clashes between a non-bonded pair of heavy atoms. On the other hand, the ``rmsd_filter()`` method handles misfolded and other biophysically non-relevant conformations.
All the filter methods will return a mask, which is a list of pdb names, that will remain in the selection.
The users are welcome to implement their own filters and intersect the masks.

Then ``apply_filter`` will remove all the structures that are not in the mask.

.. code-block:: python3

    mask = fs.rmsd_filter(selection="name CA", rmsd_cutoff=6) # Angstroms
    fs.apply_filter(mask)

.. Hint:: 

    Multiple filters can be applied at the same time. For example,

    .. code-block:: python3

	mask_peptide_bonds = fs.peptide_bond_filter(mean_cutoff=1.4,std_cutoff=0.2)
        mask_steric_clash = fs.steric_clash_filter(min_non_bonded_cutoff=1.0)
	mask_antigen = fs.rmsd_filter(selection="name CA and chainid 0", rmsd_cutoff=3)
        mask_antibody = fs.rmsd_filter(selection="name CA and chainid 1 2", rmsd_cutoff=6)

        fs.apply_filter(mask_peptide_bonds,mask_steric_clash,mask_antibody, mask_antigen)
    
    The first two filters will ensure the chemistry of the generated structures.
    The last two filters will ensure ``chainid 0`` and ``chainid 1 2`` are both folded, 
    but allow them to adapt to different relative positions.

Method ``get_rmsd()`` will return a dictionary of RMSD values between the reference structure and all other structures.

.. code-block:: python3

    rmsd = np.asarray([r for r in fs.get_rmsd(selection="name CA").values()])

Subsequently, the pairwise distances can be ranked by their coefficients of variation (CoV), which is variance/mean.

.. code-block:: python3

    atom_set = ["resid 52 to 81 and name CA",
                "resid 162 to 224 and name CA",
                "resid 185 and name CB CG",    # DFG-Asp
                "resid 186 and name CZ CG",    # DFG-Phe
                "resid 187 and name O",        # DFG-Gly
                "resid 73 and name CD",        # ChelE
                "resid 56 and name CB CZ NZ",  # SB-K
                "resid 171 and name N"         # SB-R
                ]     

    selection = " or ".join([f"({atom})" for atom in atom_set])
    names, cv = fs.rank_feature(selection=selection)

In our example, we manually designated a few atoms, and used them to create a selection.
All pairwise distances within this selection will be used to calculate CoV.
The ``names`` are generated according to the atoms involved, and the CoV values are returned in arrays.
These ``names`` are returned in decreasing order of CoV, so the first one has the largest CoV.

The class has an attribute called ``features`` (a dictionary). 
It stores all the distances across all structures. 
The names here are also keys for this dictionary. 
For example, this will print out the name and mean of most variable 100 features.

.. code-block:: python3

    for name in names[:100]:
        print(name, np.mean(fs.features[name]))

Another good way to visualize this is to generate a plotting script for visualization software like ChimeraX.

.. code-block:: python3

    print(fs.get_chimera_plotscript(names[:200], add_header=True))

This will give you a script that can be run in ChimeraX to visualize the most variable 200 features.
``add_header``, if set true, will add a ``open <filename>`` command to the top of the script.

Finally, regular space clustering will give a list of cluster centers.

.. code-block:: python3

    center_id = fs.regular_space_clustering(names[:200], 5)

The regular space clustering happens in a subspace of all features with a smaller dimension. 
This subspace is defined by the names of those taken features as the first argument.
In the above example, the top 200 variable features are used.
Eucliean distances can be come increasingly uninformative when the number of dimensions increase.
It is recommended the numbers is not too big so clustering becomes less meaningful.
Nor should it be too small so important features are not missed.

The second argument is the distance threshold in Angstrom. 
This parameter mostly controls how many cluster centers are identified.
The ``max_centers`` and ``batch_size`` mostly controls the performance of the code which shouldn't really be a concern.
A ``randomseed`` option is also provided for reproducibility.

These returned ``center_id`` can be used to retrieve the filenames of the cluster centers.

.. code-block:: python3

    for i in center_id:
        print(fs.pdb_name[i])

The atom indices of the selected pairwise distances can also be retrieved for subsequent MD simulation.

.. code-block :: python3
    
    for n in names:
        print(fs.atom_pairs[n])
