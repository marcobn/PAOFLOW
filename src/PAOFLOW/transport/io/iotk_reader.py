import re
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np


class IOTKReader:
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.lines: List[str] = []
        self.pointer: int = 0

        if not self.file_path.exists():
            raise FileNotFoundError(f"IOTK file {file_path} not found")

        with open(self.file_path) as f:
            self.lines = [line.strip() for line in f if line.strip()]

    def rewind(self):
        self.pointer = 0

    def find_section(self, section: str) -> None:
        """
        Locate the start of a section, handling both @SECTION and <SECTION> XML formats.
        Sets self.pointer to the first line after the section tag.
        """
        section_upper = section.upper()

        # First try legacy @SECTION style
        pattern_at = f"@{section_upper}"
        for idx, line in enumerate(self.lines):
            if line.upper().startswith(pattern_at):
                self.pointer = idx + 1
                return

        # Then try XML-style <SECTION> block
        pattern_xml = f"<{section_upper}>"
        for idx, line in enumerate(self.lines):
            if line.strip().upper() == pattern_xml:
                self.pointer = idx + 1
                return

        raise ValueError(f"Section @{section} or <{section}> not found in file")

    def read_attr_block(self) -> Dict[str, str]:
        attrs = {}
        while self.pointer < len(self.lines):
            line = self.lines[self.pointer]
            if line.startswith("@"):  # new section starts
                break
            if "=" in line:
                key, val = line.split("=", 1)
                attrs[key.strip().lower()] = val.strip()
            self.pointer += 1
        return attrs

    def read_matrix(self, shape: Tuple[int, int]) -> np.ndarray:
        nrows, ncols = shape
        data = []
        while len(data) < nrows * ncols:
            if self.pointer >= len(self.lines):
                raise ValueError("Unexpected end of file while reading matrix")
            line = self.lines[self.pointer]
            tokens = re.split(r"[\s,]+", line.strip())
            floats = list(map(float, tokens))
            data.extend(floats)
            self.pointer += 1
        return np.array(data).reshape((nrows, ncols))

    def read_complex_matrix(self, shape: Tuple[int, int]) -> np.ndarray:
        nrows, ncols = shape
        data = []
        while len(data) < 2 * nrows * ncols:
            if self.pointer >= len(self.lines):
                raise ValueError("Unexpected end of file while reading complex matrix")
            line = self.lines[self.pointer]
            tokens = re.split(r"[\s,]+", line.strip())
            floats = list(map(float, tokens))
            data.extend(floats)
            self.pointer += 1

        complex_data = np.array(data).reshape(nrows * ncols, 2)
        result = complex_data[:, 0] + 1j * complex_data[:, 1]
        return result.reshape(nrows, ncols)

    def read_vector_array(self, shape: Tuple[int, int]) -> np.ndarray:
        nrows, ncols = shape
        data = []
        while len(data) < nrows * ncols:
            if self.pointer >= len(self.lines):
                raise ValueError("Unexpected end of file while reading vector array")
            line = self.lines[self.pointer]
            tokens = re.split(r"[\s,]+", line.strip())
            ints = list(map(int, tokens))
            data.extend(ints)
            self.pointer += 1
        return np.array(data).reshape((nrows, ncols))

    def read_block(
        self, name: str, shape: Tuple[int, int], complex_values: bool = False
    ) -> np.ndarray:
        self.rewind()
        found = False
        for idx, line in enumerate(self.lines):
            if line.upper().startswith(name.upper()):
                self.pointer = idx + 1
                found = True
                break
        if not found:
            raise ValueError(f"Block {name} not found in IOTK file")

        if complex_values:
            return self.read_complex_matrix(shape)
        else:
            return self.read_matrix(shape)

    def find_spin_section(self, ispin: int):
        tag = f"@SPIN{ispin}"
        for idx, line in enumerate(self.lines):
            if line.upper().startswith(tag):
                self.pointer = idx + 1
                return
        raise ValueError(f"Spin section {tag} not found")

    def read_header(self) -> dict:
        self.rewind()
        inside_hamiltonian = False
        collecting_data_line = False
        data_line = ""

        for line in self.lines:
            if "<HAMILTONIAN>" in line:
                inside_hamiltonian = True
                continue
            if "</HAMILTONIAN>" in line:
                break

            if inside_hamiltonian:
                if "<DATA" in line:
                    collecting_data_line = True
                    data_line = line
                    if "/>" in line:
                        collecting_data_line = False
                elif collecting_data_line:
                    data_line += " " + line
                    if "/>" in line:
                        collecting_data_line = False

                if data_line and not collecting_data_line:
                    # Extract key="value" pairs from the assembled line
                    pattern = r'(\w+)=["\']([^"\']+)["\']'
                    matches = re.findall(pattern, data_line)
                    header = {}
                    for key, val in matches:
                        val_clean = val.strip().strip("[]")
                        if key in {"nr", "nk", "shift"}:
                            header[key.lower()] = np.fromstring(val_clean, sep=" ")
                        elif key in {"have_overlap"}:
                            header[key.lower()] = val.upper() in {"T", ".TRUE."}
                        elif key in {"dimwann", "nkpnts", "nspin", "nrtot"}:
                            header[key.lower()] = int(val)
                        elif key in {"fermi_energy"}:
                            header[key.lower()] = float(val)
                        else:
                            header[key.lower()] = val
                    return header

        raise ValueError(
            "Could not find complete <DATA .../> line inside <HAMILTONIAN> section"
        )

    def read_array(self, name: str, shape: tuple[int, int], dtype=int) -> np.ndarray:
        self.rewind()
        found = False
        for idx, line in enumerate(self.lines):
            if line.upper().startswith(f"<{name.upper()}"):
                self.pointer = idx + 1
                found = True
                break
        if not found:
            raise ValueError(f"Array <{name}> not found in IOTK file")

        if dtype is int:
            return self.read_vector_array(shape)
        elif dtype is float:
            return self.read_matrix(shape)
        elif dtype is complex:
            return self.read_complex_matrix(shape)
        else:
            raise TypeError(f"Unsupported dtype {dtype} for read_array")
