'''
Openfold interface
'''

from .base import AlphaFoldBase

class OpenFold(AlphaFoldBase):

    def __init__(self, **kwargs):
        super().__init__(kwargs)