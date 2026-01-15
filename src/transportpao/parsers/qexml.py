from pathlib import Path
from typing import Dict, Any
import xml.etree.ElementTree as ET
import numpy as np


def qexml_read_cell(file_path: str) -> Dict[str, Any]:
    """
    Read lattice vectors and cell parameters from a QE XML file.

    Parameters
    ----------
    `file_path` : str
        Path to the XML file, e.g., `atomic_proj.xml`.

    Returns
    -------
    `cell_data` : dict
        Dictionary containing the lattice vectors and parameters:
        - `alat` : float
        - `avec` : np.ndarray, shape (3,3)
        - `bvec` : np.ndarray, shape (3,3)
    """
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"File {file_path} not found")

    tree = ET.parse(file_path)
    root = tree.getroot()

    ns = {"q": root.tag.split("}")[0].strip("{")} if "}" in root.tag else {}

    def find_text(tag):
        el = root.find(f".//q:{tag}" if ns else f".//{tag}", namespaces=ns)
        return el.text if el is not None else None

    def find_array(tag):
        text = find_text(tag)
        return np.fromstring(text, sep=" ") if text else None

    alat = float(find_text("LATTICE_PARAMETER"))
    a1 = find_array("a1")
    a2 = find_array("a2")
    a3 = find_array("a3")
    b1 = find_array("b1")
    b2 = find_array("b2")
    b3 = find_array("b3")

    avec = np.column_stack((a1, a2, a3))
    bvec = np.column_stack((b1, b2, b3))
    bvec = bvec * 2 * np.pi / alat  # convert to reciprocal lattice vectors in bohr^-1

    return {
        "alat": alat,
        "avec": avec,
        "bvec": bvec,
    }
