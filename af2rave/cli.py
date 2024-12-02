"""Console script for af2rave."""
import af2rave
from . import af2rave as wrapper

import typer
from rich.console import Console

app = typer.Typer()
console = Console()


@app.command()
def main():
    """Console script for af2rave."""
    console.print("Replace this message by putting your code into "
               "af2rave.cli.main")
    console.print("See Typer documentation at https://typer.tiangolo.com/")

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
    return wrapper.simulation(sim_metadata)

if __name__ == "__main__":
    app()
