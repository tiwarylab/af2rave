'''
LocalColabFold interface
'''

from .base import AlphaFoldBase
from colabfold.batch import run as colabfold_run

class ColabFold(AlphaFoldBase):

    def __init__(self, **kwargs):
        super().__init__(kwargs)

    def predict(self, seqs, output_dir, **kwargs):

        query = ("prediction", "AAAAAAAAAAA", None)

        kwargs["is_complex"] = False
        kwargs["user_agent"] = "colabfold/1.5.5"

        return colabfold_run(queries=query, result_dir=output_dir, **kwargs)