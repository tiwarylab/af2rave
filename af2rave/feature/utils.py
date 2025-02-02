import mdtraj as md

def resid(top: md.Topology, index: int) -> int:
    """
    Get the residue ID by atom index. This residue ID starts from 1.

    :param top: The topology object.
    :param index: The index of the atom.
    :return: The residue ID.
    """
    return int(top.atom(index).residue.resSeq)

def resname(top: md.Topology, index: int) -> str:
    """
    Get the residue name by atom index. Example: "GLY".

    :param top: The topology object.
    :param index: The index of the atom.
    :return: The residue name.
    """
    return top.atom(index).residue.name

def chain(top: md.Topology, index: int) -> str:
    """
    Get the chain ID by atom index. Example: "A".

    :param top: The topology object.
    :param index: The index of the atom.
    :return: The chain ID.
    """
    return top.atom(index).residue.chain.chain_id

def atom_name(top: md.Topology, index: int) -> str:
    """
    Get the atom name by its index. Example: "CA".

    :param top: The topology object.
    :param index: The index of the atom.
    :return: The atom name.
    """
    return top.atom(index).name

def representation(top: md.Topology, index: int) -> str:
    """
    Get a formatted atom representation.

    Format example: "GLY1A-CA" for Gly-1 in chain A, CA atom.

    :param top: The topology object.
    :param index: The index of the atom.
    :return: The formatted atom representation.
    """
    atom = top.atom(index)
    residue = atom.residue
    return f"{residue.name}{residue.resSeq}{residue.chain.chain_id}-{atom.name}"

def chimera_representation(top: md.Topology, index: int) -> str:
    """
    Get the ChimeraX representation of the atom by its index.

    Format example: "/A:1@CA" for Gly-1 in chain A, CA atom.

    :param top: The topology object.
    :param index: The index of the atom.
    :return: The ChimeraX representation of the atom.
    """
    atom = top.atom(index)
    return f"/{atom.residue.chain.chain_id}:{atom.residue.resSeq}@{atom.name}"
