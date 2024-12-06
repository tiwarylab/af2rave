"""Console script for af2rave."""
from . import app as wrapper
from pathlib import Path
import json
import af2rave

import typer
from rich.console import Console
from typing import Annotated

app = typer.Typer()
console = Console()

@app.callback()
def callback():
    """
    AlphaFold2-RAVE (af2rave) is a python package that computes
    Boltzmann weights of protein alternative conformations using AlphaFold2 and RAVE.
    """
    pass

@app.command()
def simulation(input_json: str, parsed_json: str = None):
    """
    Run the MD part of af2rave.
    """

    input = wrapper.parse_input(input_json)
    sim_metadata = input.get("simulation", None)
    if sim_metadata is None:
        console.print("No simulation metadata found in the input JSON.")
        return 1
    return wrapper.app_simulation(sim_metadata, parsed_json)

@app.command()
def amino(json: Annotated[str,
              typer.Option(help=("Input JSON file containing the order parameter labels and data."
                                 "Providing this will override any other command line arguments."))] = None,
          parsed_json: Annotated[str,
              typer.Option(help="Output JSON file containing the parsed input JSON.")] = None,
          filename: Annotated[str,
              typer.Option(help="Name of the COLVAR file containing the order parameters")] = None,
          n: Annotated[int,
              typer.Option(help="Number of order parameters to be calculated")] = None,
          bins: Annotated[int,
              typer.Option(help="Number of bins")] = 50,
          kde_bandwidth: Annotated[float,
              typer.Option(help="Bandwidth for kernel density estimatio ")] = 0.02,
          verbose: Annotated[bool,
              typer.Option(help="Print progress")] = False):

    if json is None:
        amino_metadata = {
            "filename": filename,
            "n": n,
            "bins": bins,
            "kde_bandwidth": kde_bandwidth,
            "verbose": verbose
        }
    else:
        metadata = wrapper.parse_input(json)
        amino_metadata = metadata.get("amino", None)

    if amino_metadata is None:
        console.print("No AMINO metadata found in the input JSON.")
        return 1

    return wrapper.app_amino(amino_metadata, parsed_json)
