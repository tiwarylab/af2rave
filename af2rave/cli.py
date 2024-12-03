"""Console script for af2rave."""
from . import app as wrapper
from pathlib import Path
import json

import typer
from rich.console import Console

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

    if Path(input_json).exists():
        input = wrapper.parse_input(input_json)
    else:
        # in such case the input json might be a string rather than a filename
        try:
            input = json.loads(input_json)
        except json.JSONDecodeError:
            console.print(f"Illegal JSON string or input JSON file does not exist.")
            return -1

    sim_metadata = input.get("simulation", None)
    if sim_metadata is None:
        console.print("No simulation metadata found in the input JSON.")
        return 1
    return wrapper.app_simulation(sim_metadata, parsed_json)
