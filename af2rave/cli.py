"""Console script for af2rave."""
from . import app as wrapper

import typer
from rich.console import Console

app = typer.Typer()
console = Console()

@app.callback()
def callback():
    """
    af2rave is a tool to run MD simulations.
    """
    pass

@app.command()
def simulation(input_json: str):
    """
    Run the MD part of af2rave.
    """
    input = wrapper.parse_input(input_json)
    sim_metadata = input.get("simulation", None)
    if sim_metadata is None:
        console.print("No simulation metadata found in the input JSON.")
        return -1
    return wrapper.app_simulation(sim_metadata)
