AMINO module
========================

AMINO is the acronym of "Automatic Mutual Information Noise Omission". 

.. tip::
    
    For the original paper, please refer to:

    Pavan Ravindra, Zachary Smith and Pratyush Tiwary, Automatic mutual information noise omission (AMINO): generating order parameters for molecular systems, *Mol. Syst. Des. Eng.*, 2020,5, 339-348, doi: `10.1039/C9ME00115H <https://doi.org/10.1039/C9ME00115H>`_

A brief overview is as follows: A set of poorly chosen collective variables (CVs) contains many redundancies. Mutual information between CVs can be used to quantify this redundancy. Our previously selected set of a few hundred CVs can be clustered based on their mutual information. The cluster centers serve as new CVs that are not redundant with each other. Information from molecular dynamics (MD) simulations enables the calculation of the mutual information matrix.  

The recommended way to run AMINO is by using the Colvar file output from MD simulations as input.  

.. code-block:: python3  

    import af2rave.amino as af2amino  
    import glob  

    colvar_files = glob.glob("/path/to/colvar/files/*.dat")  
    amino = af2amino.AMINO.from_file(colvar_files)  

The results can then be obtained with:  

.. code-block:: python3  

    print(amino.result)  

This will print a list of CV names in the format ``dist_xxx_yyy``, where ``xxx`` and ``yyy`` are the atom indices of the two atoms involved in the CV.  

The method ``AMINO.from_file`` accepts several keyword arguments. The primary parameter users may consider adjusting is ``n``, which controls the maximum number of CVs returned. The default value is 20, which is generally suitable. However, when CVs have very similar distances, AMINO may return only two CVs. In such cases, users may increase ``n`` to allow more CVs to be included in the output.  


The processed timeseries files can also be obtained easily. The class :class:`af2rave.Colvar` provides a convenient way to handle Colvar files.  

.. code-block:: python3  

    from af2rave import Colvar  

    for f in colvar_files:  
        colvar = Colvar.from_file(f)  
        colvar.choose(amino.result).write(f"{f}.selected")  

This will generate a new Colvar file containing the selected CVs. The ``choose`` method takes a list of CV names as input, and the ``write`` method saves the selected CVs to a new file.  

The next step is to run SPIB on this non-redundant subset of CVs to obtain state labels and latent variables.  
