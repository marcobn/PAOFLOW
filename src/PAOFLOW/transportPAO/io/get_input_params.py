import yaml
from PAOFLOW.transportPAO.io.input_parameters import ConductorData, CurrentData


def get_input_from_yaml(yaml_file: str) -> dict:
    with open(yaml_file) as f:
        content = f.read()
        return yaml.safe_load(content)


def load_conductor_data_from_yaml(yaml_path: str, comm=None) -> ConductorData:
    """
    Load and validate conductor input parameters from a YAML configuration file.

    This function parses the YAML file, extracts the `input_conductor` and
    `hamiltonian_data` sections, validates the conductor input against the
    `ConductorData` schema, and returns the result as a ConductorData object.

    Parameters
    ----------
    yaml_path : str
        Path to the YAML file containing the conductor input configuration.
    comm : optional
        MPI communicator (default: None).

    Returns
    -------
    ConductorData
        Validated ConductorData object.
    """
    full_yaml = get_input_from_yaml(yaml_path)
    merged = {}
    for section in ("input_conductor", "hamiltonian_data"):
        merged.update(full_yaml.get(section, {}))
    validated = ConductorData(filename=yaml_path, validate=True, **merged)
    return validated


def load_current_data_from_yaml(yaml_path: str) -> dict | None:
    """
    Load current input parameters from a YAML file.

    Parameters
    ----------
    yaml_path : str
        Path to the `current.yaml` file.

    Returns
    -------
    dict or None
        Parsed dictionary of input parameters if file exists, otherwise None.
    """
    validated = CurrentData(filename=yaml_path, validate=True)
    return validated.model_dump()
