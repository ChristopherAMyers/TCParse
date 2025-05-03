# TCParse

TCParse is a Python parser for TeraChem output files. It extracts various molecular properties (such as energies, geometries, dipole moments and derivatives, and charge information) and stores them in a structured JSON format. The parsed data is then validated using [pydantic](https://pydantic-docs.helpmanual.io/), defined in the [`TCJobData`](src/tcparse/tcparse.py) model.

## Project Structure

- **src/tcparse/**  
  Contains the main parser logic in [`tcparse.py`](src/tcparse/tcparse.py).
- **tests/**  
  Holds test scripts and sample jobs to validate the functionality.
- **debug/**  
  Contains debug files produced during development.
- **pyproject.toml**  
  Project configuration.
- **README.md**  
  This file.

## Requirements

- Python 3.7+
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
python -m tcparse.tcparse -t path/to/tera.out -j path/to/output.json -x path/to/geom.xyz
```

Where:
- `-t` is your TeraChem output file.
- `-j` is the output JSON file path.
- `-x` is the (optional) coordinate file location.

To display the help screen with all available options, use:

```sh
python -m tcparse.tcparse -h
```

## Using TCParse as a Python Module

You can also use `tcparse` as a Python module to parse TeraChem output programmatically. The following functions are available:

### `parse_from_file(filepath: str) -> dict`
Parses a TeraChem output file and returns the extracted data as a dictionary.

**Example:**
```python
from tcparse import parse_from_file

data = parse_from_file("path/to/terachem.out")
print(data)
```

### `parse_from_list(lines: list[str]) -> dict`
Parses TeraChem output provided as a list of strings (e.g., lines read from a file) and returns the extracted data as a dictionary.

**Example:**
```python
from tcparse import parse_from_list

with open("path/to/terachem.out", "r") as file:
    lines = file.readlines()
data = parse_from_list(lines)
print(data)
```

These functions allow you to integrate `tcparse` into larger Python workflows or scripts.

## Testing

Tests can be found in the [tests](tests/) folder. Run the tests using:

```sh
python -m unittest discover tests
```

## Contributing

Contributions are welcome. Please feel free to submit issues or pull requests.

## License

Specify your license here.