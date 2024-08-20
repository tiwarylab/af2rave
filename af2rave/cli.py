"""Console script for af2rave."""
import af2rave

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
    


if __name__ == "__main__":
    app()
