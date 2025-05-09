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

The `-x` option is usefull when the coordinates are not included in the TeraChem output.

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

If the file that you parsed is from a numerical derivative or moolecular dynamics (MD) simulation with multiple energy calculations, then the returned data will be a list of `TCJobData`'s.

**Example:**
```python
from tcparse import parse_from_file

# import the data from a file
data = parse_from_file("path/to/terachem_md.out")

# convert to a python dictionary
data_dict = [x.model_dump('json') for x in datal]

```


### `parse_from_list(lines: list[str]) -> TCJobData | list[TCJobData]`
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

## Parsed Data Format Description

When you parse a TeraChem output for a CIS job with TCParse, you receive a JSON object with the following keys. The example shown below (from `tests/jobs/cis/tc.json`) illustrates the structure and shape of many of the fields.

### Common Fields Across Job Types

- **`atoms`**  
  - *Type:* List of strings  
  - *Description:* Each element is the atomic symbol (e.g. `"C"`, `"H"`, `"O"`) for each atom in the system.  
  - *Shape:* `(n_atoms,)`

- **`energy`**  
  - *Type:* List of floats  
  - *Description:* Energies (in atomic units) corresponding to each energy state. If only a ground state energy is calculated, then this will be a list of only one value. Else, it will be equal to the number of excited states + 1.
  - *Shape:* `(n_states,)` 

- **`geom`**  
  - *Type:* List of lists  
  - *Description:* Cartesian coordinates of the atoms. Each inner list represents `[x, y, z]` coordinates for one atom.
  - *Shape:* `(n_atoms, 3)`

- **`dipole_moment`**  
  - *Type:* Float  
  - *Description:* The ground state dipole moment magnitude of the system.
  
- **`dipole_vector`**  
  - *Type:* List of 3 floats  
  - *Description:* The ground state dipole moment expressed as a vector `[mu_x, mu_y, mu_z]`.

- **`dipole_deriv`**  
  - *Type:* Nested list (2-dimensional)  
  - *Description:* The derivative of the ground state dipole moment with respect to the nuclear coordinates. The dimensions are organized as follows:  
    - First level: The three dipole components that are differentiated `[mu_x, mu_y, mu_z]`
    - Second level: Individual atoms indices
    - Third level: The three Cartesian components of the atomic coordinates `[x, y, z]` for each entry  
  - *Shape:* `(3, n_atoms, 3)` 

- **`esp_charges`** and **`resp_charges`**  
  - *Type:* The (restrained) electrostatic potential charges for the ground state density.
  - *Description:* Represent the electrostatic potential (ESP) derived charges and RESP charges.
  - *Shape:*  `(n_atoms,)`

- **`esp_dipoles`** and **`resp_dipoles`**  
  - *Type:* 3 component list of floats.
  - *Description:* The dipole moments computed using the ground state ESP or RESP charges.
  - *Shape:* `(3,)` 

- **`gradient`**  
  - *Type:* Nested list (2-dimensional)  
  - *Description:* The final gradient components printed in the TeraChem output. The output is controlled by the target state specified in the TeraChem input. For example, if one set `cistarget = 2`, then the gradient would be nuclear gradient of the second CIS excited state energy. Each vector is given as `[dE/dx, dE/dy, dE/dz]`.
  - *Shape:* `(n_atoms, 3)`

### CIS-Specific Fields

For this section, we define the number of excited states as `n_ex_states`, and the number of states in a system (including the ground state) as `n_states = n_ex_states + 1`. The number of combinates of states (i.e. the number of possible transitions) is labeled `n_trans = n_states * (n_states + 1) / 2`.

- **`cis_states`**  
  - *Type:* Integer  
  - *Description:* Number of CIS states included.

- **`cis_ex_energies`**  
  - *Type:* List of floats  
  - *Description:* Excitation energies (in eV) for each CIS state.
  - *Shape:* `(n_ex_states,)`

- **`cis_unrelaxed_dipoles`** and **`cis_relaxed_dipoles`** 
  - *Type:* Nested list (2-dimensional)  
  - *Description:* Dipole vectors of the unrelaxed and unrelaxed densities.
  - *Shape:* `(n_ex_states, 3)`

- **`cis_transition_dipoles`** and **`cis_relaxed_transition_dipoles`**  
  - *Type:* Nested list (2-dimensional)  
  - *Description:* Dipole vectors of the relaxed CIS transition densities.
  - *Shape:* `(n_ex_states, 3)`

- **`cis_excitations`**  
  - *Type:* Nested list (typically 2-dimensional)  
  - *Description:* For each CIS state, this field lists the orbital excitations and their CI coefficients. The number of excitations listed for each state is controlled by the `cisprintthresh` TeraChem option. Each excitation is represented as a triplet:  
    - `[orbital_from, orbital_to, coefficient]`  
  - *Shape:* `(n_ex_states, n_excitations_per_state, 3)`

- **`cis_gradients`**  
  - *Type:* Nested list (2-dimensional)  
  - *Description:* Nuclear gradients for each CIS state. Each state contains an array of gradient vectors for the atoms, with each vector given as `[dE/dx, dE/dy, dE/dz]`.
  - *Shape:* `(n_ex_states, n_atoms, 3)`

- **`cis_couplings`**  
  - *Type:* Nested list  
  - *Description:* Coupling values between different states. For each state a set of coupling vectors is provided, each with three components.
  - *Shape:* Typically `(n_trans, n_atoms, 3)` or a similar structure depending on the output.

- **`cis_osc_strengths`**  
  - *Type:* List of floats  
  - *Description:* Oscillator strengths for the CIS transitions.
  - *Shape:* `(n_ex_states,)`

- **`cis_relaxed_esp_charges`** and **`cis_relaxed_resp_charges`**  
  - *Type:* List of lists of floats  
  - *Description:* Electrostatic potential (ESP) charges and RESP charges for each atom computed for each state using the relaxed CIS density.
  - *Shape:* `(n_ex_states, n_atoms)`

- **`cis_relaxed_esp_dipoles`** and **`cis_relaxed_resp_dipoles`**  
  - *Type:* List of lists (each inner list has 3 floats)  
  - *Description:* Electrostatic potential (ESP) dipoles and RESP dipoles computed for each state using their respective charges.
  - *Shape:* `(n_ex_states, 3)`

- **`cis_tr_esp_charges`** and **`cis_tr_resp_charges`**  
  - *Type:* Typically 3-dimensional nested lists, with entries for different electronic transitions.  
  - *Description:* Transition ESP and RESP charges between the CIS states.  
  - *Shape:* For example, `(n_trans, n_atoms)`.

- **`cis_tr_esp_dipoles`** and **`cis_tr_resp_dipoles`**  
  - *Type:* Nested lists with three components per transition.  
  - *Description:* Transition ESP and RESP dipoles between the CIS states using the `cis_tr_esp_charges` or `cis_tr_resp_charges`.  
  - *Shape:* For example, `(n_trans, 3)`.

- **`cis_unrelaxed_dipole_deriv`**  
  - *Type:* Nested list (4-dimensional)  
  - *Description:* These are the unrelaxed dipole derivatives resulting from the CIS calculations. The dimensions are organized as follows:  
    - First level: Different CIS states `[mu_x, mu_y, mu_z]`
    - Second level: The three dipole components that are differentiated
    - Third level: Individual atoms indices
    - Fourth level: The three Cartesian components of the atomic coordinates `[x, y, z]` for each entry  
  - *Example Shape:* `(n_ex_states, 3, n_atoms, 3)`

- **`cis_transition_dipole_deriv`**  
  - *Type:* Nested list (4-dimensional)  
  - *Description:* Derived dipole derivatives for the transitions between states. The dimensions mimic those of the unrelaxed dipole derivatives, `cis_unrelaxed_dipole_deriv`, except that there are `n_trans` of them.
  - *Shape:* `(n_trans, 3, n_atoms, 3)`.

### CAS-Specific Fields

- **`cas_states`**  
  - *Type:* Integer  
  - *Description:* Number of CAS states included.

- **`cas_dipoles`**  
  - *Type:* Nested list (2-dimensional)  
  - *Description:* Dipole vectors for each CAS state. Each vector is represented as `[mu_x, mu_y, mu_z]`. 
  - *Shape:* `(n_states, 3)`

- **`cas_transition_dipoles`**  
  - *Type:* Nested list (2-dimensional)  
  - *Description:* Transition dipole vectors between CAS states. Each vector is represented as `[dx, dy, dz]`. There are `n_trans = n_states (n_states + 1) / 2` possible transitions.
  - *Shape:* `(n_trans, 3)`.

---

*Note:* The exact shapes may vary depending on the details of your TeraChem job (e.g., how many states were computed, whether the job was a single-point energy calculation, MD simulation, or numerical derivative). This description is based on the provided example and typical outputs from CIS calculations.

This extended information should help you understand the structure and meaning of each part of the parsed JSON data.

## Testing

Tests can be found in the [tests](tests/) folder. Run the tests using:

```sh
python -m unittest discover tests
```

## Contributing

Contributions are welcome. Please feel free to submit issues or pull requests.
