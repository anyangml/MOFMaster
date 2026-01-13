"""
Unit tests for tools
"""

from pathlib import Path
from app.tools.io import get_data_dir, write_cif_file, read_cif_file


def test_get_data_dir():
    """Test that data directory is created"""
    data_dir = get_data_dir()
    assert data_dir.exists()
    assert data_dir.is_dir()


def test_write_and_read_cif():
    """Test writing and reading CIF files"""
    data_dir = get_data_dir()
    test_file = data_dir / "test.cif"

    # Write file
    content = "data_test\n_cell_length_a 10.0\n"
    write_cif_file(str(test_file), content)

    # Read file
    read_content = read_cif_file(str(test_file))

    assert read_content == content

    # Cleanup
    test_file.unlink()


def test_search_mof_db():
    """Test MOF database search"""
    from app.tools.retrieval import search_mof_db

    # Search for copper
    result = search_mof_db.func("copper")

    assert "mof_name" in result
    assert result["mof_name"] == "HKUST-1"
    assert "cif_filepath" in result

    # Verify CIF file was created
    assert Path(result["cif_filepath"]).exists()


def test_search_mof_db_not_found():
    """Test MOF database search with no results"""
    from app.tools.retrieval import search_mof_db

    result = search_mof_db.func("nonexistent_element_xyz")

    assert "error" in result
