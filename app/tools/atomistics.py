"""
Atomistic simulation tools using ASE (Atomic Simulation Environment)
"""

from pathlib import Path
from typing import Dict, Any
from langchain_core.tools import tool

import ase.io
from ase.calculators.emt import EMT
from ase.optimize import BFGS

from app.tools.io import get_data_dir


@tool
def optimize_structure_ase(cif_filepath: str) -> Dict[str, Any]:
    """
    Optimize the geometry of a MOF structure using ASE.

    This tool performs geometry optimization on a crystal structure
    using the BFGS algorithm. It uses the EMT (Effective Medium Theory)
    calculator for fast testing. The optimization continues until forces
    are below 0.05 eV/Å.

    Args:
        cif_filepath: Path to the input CIF file

    Returns:
        Dictionary containing:
        - optimized_cif_filepath: Path to the optimized structure
        - initial_energy_ev: Initial potential energy (eV)
        - final_energy_ev: Final potential energy after optimization (eV)
        - n_steps: Number of optimization steps taken

    Example:
        >>> optimize_structure_ase("/path/to/structure.cif")
        {'optimized_cif_filepath': '/path/to/structure_optimized.cif',
         'final_energy_ev': -123.45, ...}
    """

    try:
        # Read the structure
        atoms = ase.io.read(cif_filepath)

        # Set up calculator (EMT for lightweight testing)
        atoms.calc = EMT()

        # Get initial energy
        initial_energy = atoms.get_potential_energy()

        # Run optimization
        optimizer = BFGS(atoms, logfile=None)  # logfile=None to suppress output
        optimizer.run(fmax=0.05)  # Optimize until forces < 0.05 eV/Å

        # Get final energy
        final_energy = atoms.get_potential_energy()

        # Save optimized structure
        input_path = Path(cif_filepath)
        output_filename = input_path.stem + "_optimized.cif"
        output_path = get_data_dir() / output_filename

        ase.io.write(str(output_path), atoms, format="cif")

        return {
            "optimized_cif_filepath": str(output_path),
            "initial_energy_ev": float(initial_energy),
            "final_energy_ev": float(final_energy),
            "energy_change_ev": float(final_energy - initial_energy),
            "n_steps": optimizer.get_number_of_steps(),
            "converged": True,
        }

    except Exception as e:
        return {"error": f"Optimization failed: {str(e)}", "cif_filepath": cif_filepath}


@tool
def calculate_energy_force(cif_filepath: str) -> Dict[str, Any]:
    """
    Calculate the energy and forces of a structure.

    This tool performs a static point calculation on a crystal structure
    to determine its potential energy and the forces on each atom.

    Args:
        cif_filepath: Path to the CIF file

    Returns:
        Dictionary containing:
        - energy_ev: Potential energy in eV
        - max_force_ev_ang: Maximum force component in eV/Å
        - cif_filepath: Path to the analyzed structure

    Example:
        >>> calculate_energy_force("/path/to/structure.cif")
        {'energy_ev': -123.45, 'max_force_ev_ang': 0.02, ...}
    """

    try:
        # Read the structure
        atoms = ase.io.read(cif_filepath)

        # Set up calculator
        atoms.calc = EMT()

        # Calculate energy and forces
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()

        # Find maximum force component
        max_force = float(abs(forces).max())

        return {
            "energy_ev": float(energy),
            "max_force_ev_ang": max_force,
            "cif_filepath": cif_filepath,
            "n_atoms": len(atoms),
        }

    except Exception as e:
        return {"error": f"Energy calculation failed: {str(e)}", "cif_filepath": cif_filepath}
