"""Test ``utils.pseudo`` module."""

from pathlib import Path

import pytest

from aiida_sssp_workflow.utils.pseudo import extract_pseudo_info, parse_std_filename

upf_folder = Path(__file__).parent.parent / "_statics" / "upf"
upf_files = list(upf_folder.glob("*.upf"))


@pytest.mark.parametrize(
    "files",
    upf_files,
)
def test_extract_pseudo_info(files):
    """Test the ``extract_pseudo_info`` function."""
    with open(files, "r") as fh:
        pseudo_text = fh.read()

    info_from_text = extract_pseudo_info(pseudo_text)
    info_from_filename = parse_std_filename(files.name)

    assert info_from_text == info_from_filename
