"""
MOF database retrieval and search tools
"""

from typing import Dict, Any
from langchain_core.tools import tool

from app.tools.io import get_data_dir


# Sample MOF database (for MVP - would be replaced with vector DB later)
SAMPLE_MOF_DB = [
    {
        "mof_name": "HKUST-1",
        "formula": "Cu3(BTC)2",
        "description": "Copper-based MOF with high surface area",
        "tags": ["copper", "high surface area", "paddle-wheel"],
        "cif_filename": "HKUST-1.cif",
        "properties": {"surface_area_m2g": 1850, "pore_volume_cm3g": 0.75},
    },
    {
        "mof_name": "MOF-5",
        "formula": "Zn4O(BDC)3",
        "description": "Zinc-based MOF, one of the first MOFs discovered",
        "tags": ["zinc", "BDC", "cubic"],
        "cif_filename": "MOF-5.cif",
        "properties": {"surface_area_m2g": 3800, "pore_volume_cm3g": 1.55},
    },
    {
        "mof_name": "UiO-66",
        "formula": "Zr6O4(OH)4(BDC)6",
        "description": "Zirconium-based MOF with exceptional stability",
        "tags": ["zirconium", "stable", "water-stable"],
        "cif_filename": "UiO-66.cif",
        "properties": {"surface_area_m2g": 1187, "pore_volume_cm3g": 0.44},
    },
    {
        "mof_name": "MIL-101",
        "formula": "Cr3F(H2O)2O[(O2C)-C6H4-(CO2)]3",
        "description": "Chromium-based MOF with very high surface area",
        "tags": ["chromium", "very high surface area", "mesoporous"],
        "cif_filename": "MIL-101.cif",
        "properties": {"surface_area_m2g": 4100, "pore_volume_cm3g": 2.15},
    },
]


@tool
def search_mof_db(query_string: str) -> Dict[str, Any]:
    """
    Search for MOF structures in the database.

    This tool searches a database of Metal-Organic Frameworks based on
    a query string. It performs a simple keyword match against MOF names,
    formulas, descriptions, and tags.

    Args:
        query_string: Search query (e.g., "copper based", "HKUST-1", "high surface area")

    Returns:
        Dictionary containing MOF information including name, formula,
        CIF filename, and properties. Returns the first match found.

    Example:
        >>> search_mof_db("copper")
        {'mof_name': 'HKUST-1', 'formula': 'Cu3(BTC)2', ...}
    """

    query_lower = query_string.lower()

    # Search through the database
    for mof in SAMPLE_MOF_DB:
        # Check if query matches name, formula, description, or tags
        if (
            query_lower in mof["mof_name"].lower()
            or query_lower in mof["formula"].lower()
            or query_lower in mof["description"].lower()
            or any(query_lower in tag.lower() for tag in mof["tags"])
        ):

            # Create a simple CIF file if it doesn't exist
            data_dir = get_data_dir()
            cif_path = data_dir / mof["cif_filename"]

            if not cif_path.exists():
                # Create a minimal CIF file (placeholder for MVP)
                # In production, this would be real crystallographic data
                cif_content = f"""data_{mof['mof_name']}
_cell_length_a    26.343
_cell_length_b    26.343
_cell_length_c    26.343
_cell_angle_alpha 90.0
_cell_angle_beta  90.0
_cell_angle_gamma 90.0
_symmetry_space_group_name_H-M 'F m -3 m'

loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Cu1 Cu 0.250 0.250 0.250
O1  O  0.200 0.200 0.200
C1  C  0.150 0.150 0.150
"""
                with open(cif_path, "w") as f:
                    f.write(cif_content)

            result = {
                "mof_name": mof["mof_name"],
                "formula": mof["formula"],
                "description": mof["description"],
                "cif_filepath": str(cif_path),
                "properties": mof["properties"],
            }

            return result

    # If no match found
    return {
        "error": f"No MOF found matching query: {query_string}",
        "suggestion": "Try queries like: 'copper', 'zinc', 'HKUST-1', 'MOF-5', 'high surface area'",
    }
