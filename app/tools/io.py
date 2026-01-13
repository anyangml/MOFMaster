"""
Input/Output tools for CIF files and data management
"""

import os
from pathlib import Path


def get_data_dir() -> Path:
    """Get the data directory path, creating it if necessary"""
    data_dir = Path(os.getenv("DATA_DIR", "./data"))
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def read_cif_file(filepath: str) -> str:
    """
    Read a CIF file and return its contents as a string.

    Args:
        filepath: Path to the CIF file

    Returns:
        Contents of the CIF file as a string
    """
    with open(filepath, "r") as f:
        return f.read()


def write_cif_file(filepath: str, content: str) -> str:
    """
    Write CIF content to a file.

    Args:
        filepath: Path where to write the CIF file
        content: CIF file content as string

    Returns:
        The filepath where the file was written
    """
    with open(filepath, "w") as f:
        f.write(content)
    return filepath


def ensure_cif_in_data_dir(filepath: str) -> str:
    """
    Ensure a CIF file is in the data directory.
    If it's already there, return the path. Otherwise, copy it.

    Args:
        filepath: Path to CIF file

    Returns:
        Path to the file in the data directory
    """
    data_dir = get_data_dir()
    filepath_obj = Path(filepath)

    # If already in data dir, return as-is
    if data_dir in filepath_obj.parents or filepath_obj.parent == data_dir:
        return filepath

    # Otherwise copy to data dir
    dest = data_dir / filepath_obj.name
    content = read_cif_file(filepath)
    write_cif_file(str(dest), content)
    return str(dest)
