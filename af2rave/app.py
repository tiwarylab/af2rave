"""Main module."""

import json
from pathlib import Path

from . import simulation
from . import amino
from .colvar import Colvar

def parse_input(input_file):
    """
    Parse the input json, either a file or a json string.
    """

    if Path(input_file).exists():
        with open(input_file, 'r') as f:
            data = json.load(f)
        return json.load(f)
    else:
        # in such case the input json might be a string rather than a filename
        try:
            data = json.loads(input_file)
        except json.JSONDecodeError:
            print(f"Illegal JSON string or input JSON file does not exist.")
            return -1
        return data

def write_input(data, output_file):
    """Write the input file."""
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

def fill_missing_entries(json_data, default_values):
    """
    Recursively fills missing keys in json_data with values from default_values.

    :param json_data: dict, input JSON data
    :param default_values: dict, default values for missing keys
    :return: dict, JSON data with missing entries filled
    """
    for key, default_value in default_values.items():
        if key not in json_data:
            json_data[key] = default_value
        elif isinstance(default_value, dict) and isinstance(json_data.get(key), dict):
            fill_missing_entries(json_data[key], default_value)
    return json_data

def app_simulation(metadata: dict, output_json: str = None):
    """
    Run the MD part of af2rave.
    """

    kwargs = {}
    default_values = {
        "pdb_file": None,
        "temp": 310,
        "pressure": 1,
        "dt": 0.002,
        "cutoff": 10.0,
        "steps": 50000000,
        "append": False,
        "progress_every": 1000,
        "save_checkpoint": None,
        "save_pdb": None,
        "cv_reporter": {
            "list_of_index": None,
            "cv_file": "COLVAR.dat",
            "cv_freq": 500
        },
        "xtc_reporter": {
            "xtc_file": "traj.xtc",
            "xtc_freq": 5000
        }
    }

    metadata = fill_missing_entries(metadata, default_values)
    
    kwargs["pdb_file"] = metadata["pdb_file"]
    kwargs["temp"] = metadata["temp"]
    kwargs["pressure"] = metadata["pressure"]
    kwargs["dt"] = metadata["dt"]
    kwargs["cutoff"] = metadata["cutoff"]
    kwargs["append"] = metadata["append"]
    kwargs["progress_every"] = metadata["progress_every"]

    kwargs["list_of_index"] = metadata["cv_reporter"]["list_of_index"]
    kwargs["cv_file"] = metadata["cv_reporter"]["cv_file"]
    kwargs["cv_freq"] = metadata["cv_reporter"]["cv_freq"]

    kwargs["xtc_file"] = metadata["xtc_reporter"]["xtc_file"]
    kwargs["xtc_freq"] = metadata["xtc_reporter"]["xtc_freq"]

    if kwargs["pdb_file"] is None:
        raise ValueError("pdb_file is required.")
    
    if output_json is not None:
        write_input(metadata, output_json)

    ubs = simulation.UnbiasedSimulation(**kwargs)
    ubs.run(metadata["steps"])

    if metadata["save_checkpoint"] is not None:
        ubs.save_checkpoint(metadata["save_checkpoint"])
    if metadata["save_pdb"] is not None:
        ubs.save_pdb(metadata["save_pdb"])


def app_amino(metadata,
              output_json: str = None):
    
    default_values = {
        "filename": None,
        "n": 20,
        "bins": 50,
        "kde_bandwidth": 0.02,
        "verbose": False
    }
    metadata = fill_missing_entries(metadata, default_values)

    if metadata["filename"] is None:
        raise ValueError("An input file is required.")
    if output_json is not None:
        write_input(metadata, output_json)

    a = amino.AMINO.from_file(**metadata)
    print(f"\n{len(a.result)} AMINO Order Parameters:")
    for i in a.result:
        print(i)