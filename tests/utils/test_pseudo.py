"""Test ``utils.pseudo`` module."""

import pytest
from pathlib import Path

from aiida_sssp_workflow.utils import extract_pseudo_info, parse_std_filename
from aiida_sssp_workflow.utils.pseudo import (
    compute_total_nelectrons,
    get_pseudo_O,
    get_pseudo_N,
    CurateType,
)

upf_folder = Path(__file__).parent.parent / "_statics" / "upf"
upf_files = list(upf_folder.glob("*.upf"))


@pytest.mark.parametrize(
    "file",
    upf_files,
)
def test_extract_pseudo_info(file):
    """Test the ``extract_pseudo_info`` function."""
    with open(file, "r") as fh:
        pseudo_text = fh.read()

    info_from_text = extract_pseudo_info(pseudo_text)
    info_from_filename = parse_std_filename(file.name)

    assert info_from_text == info_from_filename


@pytest.mark.parametrize(
    "curate_type, expected_filename",
    [
        (CurateType.SSSP, "O.paw.pbe.z_6.ld1.psl.v0.1.upf"),
        (CurateType.NC, "O.nc.pbe.z_6.oncvpsp3.dojo.v0.4.1-std.upf"),
    ],
)
def test_get_pseudo_O(curate_type, expected_filename):
    """Test get_pseudo_O for different curate type"""
    pseudo, _, _ = get_pseudo_O(curate_type)

    assert pseudo.filename == expected_filename


def test_get_pseudo_O_unknown_type():
    """Test raise when curate_type is unknown

    We have type annotation for the function but python does not prevent user from passing wrong type.
    """
    with pytest.raises(ValueError, match="Unknown curate_type"):
        get_pseudo_O("unknown")


def test_compute_total_nelectrons():
    """Test util function computer_total_nelectrons"""
    pseudo_O, _, _ = get_pseudo_O()
    pseudo_N, _, _ = get_pseudo_N()

    for configuration, pseudos, n_total in [
        ("XO", {"O": pseudo_O, "N": pseudo_N}, 11),
        ("XO2", {"O": pseudo_O, "N": pseudo_N}, 17),
        ("XO3", {"O": pseudo_O, "N": pseudo_N}, 23),
        ("X2O", {"O": pseudo_O, "N": pseudo_N}, 16),
        ("X2O3", {"O": pseudo_O, "N": pseudo_N}, 56),
        ("X2O5", {"O": pseudo_O, "N": pseudo_N}, 80),
    ]:
        assert compute_total_nelectrons(configuration, pseudos) == n_total
