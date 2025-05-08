# TCParse

TCParse is a Python parser for TeraChem output files. It extracts various molecular properties (such as energies, geometries, dipole moments and derivatives, and charge information) and stores them in a structured JSON format. The parsed data is then validated using [pydantic](https://pydantic-docs.helpmanual.io/), defined in the [`TCJobData`](src/tcparse/tcparse.py) model.

## Project Structure

- **src/tcparse/**  
  Contains the main parser logic in [`tcparse.py`](src/tcparse/tcparse.py).
- **tests/**  
  Holds test scripts and sample jobs to validate the functionality.
- **pyproject.toml**  
  Project configuration.
- **README.md**  
  This file.

## Requirements

- Python 3.9+
- [numpy](https://numpy.org/)
- [pydantic](https://pydantic-docs.helpmanual.io/)

## Installation

1. Clone the repository.
2. (Optional) Create and activate a virtual environment.
3. Install the dependencies:

```sh
pip install .
```

*Alternatively, you can use the `pyproject.toml` file directly with `pip` for editable installs (e.g. when pulling new changes from GitHub):*

```sh
pip install -e .
```

## Usage

To run the parser, execute the main script with the required arguments:

```sh
python -m tcparse -t path/to/tera.out -j path/to/output.json -x path/to/geom.xyz
```

Where:
- `-t` is your TeraChem output file.
- `-j` is the output JSON file path.
- `-x` is the (optional) coordinate file location.

To display the help screen with all available options, use:

```sh
python -m tcparse -h
```

## Using TCParse as a Python Module

You can also use `tcparse` as a Python module to parse TeraChem output programmatically. The following functions are available:

### `parse_from_file(filepath: str) -> TCJobData | list[TCJobData]`
Parses a TeraChem output file and returns the extracted data as a TCJobData object. The details of this object can be found in `src/tcparse/job_data.py`, but in short, the object is a `pydantic` object with fields corresponding to the various data values. IF you'd prefer to work with a dictionary, you can use Pydantic's `model_dump` function.

**Example:**
```python
from tcparse import parse_from_file

# import the data from a file
data = parse_from_file("path/to/terachem.out")

# convert to a python dictionary
data_dict = data.model_dump()

#convert to a JSON-serializable dictionary
data_dict_json = data.model_dump('json)

```


### `parse_from_list(lines: list[str]) -> dict`
Parses TeraChem output provided as a list of strings (e.g., lines read from a file) and returns the same aforementioned data. This is usefull if the lines were already parsed from another python function.

**Example:**
```python
from tcparse import parse_from_list

with open("path/to/terachem.out", "r") as file:
    lines = file.readlines()
data = parse_from_list(lines)
```

These functions allow you to integrate `tcparse` into larger Python workflows or scripts.

## Data Structure

The parsed data is structured to represent various molecular properties in a clear and organized manner. Electronic structure elements that are based on a particular electronic state (e.g., ground state or excited states) are stored as lists of data in the order of ($S_0$, $S_1$, $S_2$, etc.). 

Values that involve multiple states (such as transition dipoles or nonadiabatic couplings) are stored in row-major ordering but only include the "upper triangle." For example, if there are three excited states, the transition dipoles would be in the order: $\vec\mu_{01}$, $\vec\mu_{02}$, $\vec\mu_{03}$, $\vec\mu_{12}$, $\vec\mu_{13}$, $\vec\mu_{23}$, and the full dataset would be a list of lists with a size (6x3).

## Testing

Tests can be found in the [tests](tests/) folder. Run the tests using:

```sh
python -m unittest discover tests
```

## Contributing

Contributions are welcome. Please feel free to submit issues or pull requests.
