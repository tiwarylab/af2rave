'''
AF2RAVE: AlphaFold module
'''

try:
    import openfold
except ImportError:
    openfold = None

try:
    import colabfold
except ImportError:
    colabfold = None

if openfold is not None:
    from .openfold import OpenFold as AlphaFold
elif colabfold is not None:
    from .colabfold import ColabFold as AlphaFold
else:
    raise ImportError("No AlphaFold module found. Please install either openfold or colabfold.")
