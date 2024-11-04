import typer
from typing import Annotated
from rich import print as rprint

from .wrapper import AMINO
from ..colvar import Colvar


def main(filename: Annotated[str,
            typer.Argument(help="Name of the COLVAR file containing the order parameters")],
         n: Annotated[int,
            typer.Option(help="Number of order parameters to be calculated")] = None,
         bins: Annotated[int,
            typer.Option(help="Number of bins")] = 50,
         kde_bandwidth: Annotated[float,
            typer.Option(help="Bandwidth for kernel density estimatio ")] = 0.02,
         override: Annotated[bool,
            typer.Option(help=("By default the --n parameter is capped at 20."
                               "Using --override will default --n option "
                               "to the number of OPs in the input COLVAR file"))] = False):

    colvar = Colvar(filename)

    # number of order parameters
    if n is None:
        n = colvar.shape[1]
        if n > 20 and not override:
            # Only trigger this warning if user defaulted the --n option
            raise UserWarning("Refuse to construct big dissimilarity matrix "
                              f"with n = {n} greater than 20. Specify an --n "
                              "or use --override flag to override this limit.")
    if n > 20:
        rprint("[bold red]Warning:[/bold red] "
               "Dimension of dissimilarity matrix n > 20. "
               "It is advised against such a big n.")

    # initializing objects and run the code
    a = AMINO(n=n, bins=bins, kde_bandwidth=kde_bandwidth, verbose=True)
    result = a.from_colvar(colvar)

    print(f"\n{len(result)} AMINO Order Parameters:")
    for i in result:
        print(i)


if __name__ == "__main__":
    typer.run(main)
