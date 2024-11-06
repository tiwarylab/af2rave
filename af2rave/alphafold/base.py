'''
This is the base class of the AlphaFold class. It will be inherited
by either ColabFold interface or OpenFold interface.
'''

from typing import List, Tuple
from pathlib import Path
from typing import Union
import os

class AlphaFoldBase(object):

    def __init__(self, 
                 sequence: str,
                 name: str = "prediction",
                 output_dir: str = "output"):

        self._sequence = sequence
        self._name = name
        self._output_dir = output_dir
        self._fasta_string = f">{name}\n{sequence}"
        self._msa = None

    @classmethod
    def from_sequence(cls, sequence: str, name: str = "prediction", output_dir: str = "output"):
        return cls(sequence=sequence, name=name, output_dir=output_dir)
    
    @classmethod
    def from_fasta(cls, fasta_string: Union[str, Path], name=None, output_dir: str = "output"):

        # first check if this string is a file
        if os.path.isfile(fasta_string):
            fasta_string = Path(fasta_string).read_text()
        
        sequences, descriptions = parse_fasta(fasta_string)
        if len(sequences) != 1:
            raise ValueError("Illegal FASTA format or contains more than one sequence.")
        if name is None:
            name = descriptions[0]
        return cls(sequence=sequences[0], name=name, output_dir=output_dir)
    
    @classmethod
    def from_a3m_msa(cls, a3m_msa: str, name=None, output_dir: str = "output"):

        if os.path.isfile(a3m_msa):
            a3m_msa = Path(a3m_msa).read_text()
        
        sequence, descriptions = parse_fasta(a3m_msa)
        if name is None:
            name = descriptions[0]
        fold = cls(sequence=sequence[0], name=name, output_dir=output_dir)
        fold._msa = a3m_msa
        return fold

    def set_msa(self, filename: Union[str, Path]):

        input_path = Path(filename)
        if not input_path.exists():
            raise FileNotFoundError(f"MSA file not found: {filename}")
        
        self._msa = input_path.read_text()
    
    def predict(self, **kwargs):
        raise NotImplementedError("AlphaFoldBase::predict() is a pure virtual function.")


def parse_fasta(fasta_string: str) -> Tuple[List[str], List[str]]:
    """Parses FASTA string and returns list of strings with amino-acid sequences.

    Arguments:
      fasta_string: The string contents of a FASTA file.

    Returns:
      A tuple of two lists:
      * A list of sequences.
      * A list of sequence descriptions taken from the comment lines. In the
        same order as the sequences.
    """
    sequences = []
    descriptions = []
    index = -1
    for line in fasta_string.splitlines():
        line = line.strip()
        if line.startswith("#"):
            continue
        if line.startswith(">"):
            index += 1
            descriptions.append(line[1:])  # Remove the '>' at the beginning.
            sequences.append("")
            continue
        elif not line:
            continue  # Skip blank lines.
        sequences[index] += line

    return sequences, descriptions
