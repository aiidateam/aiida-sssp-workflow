# -*- coding: utf-8 -*-
"""fixtures"""
import os

import pytest

pytest_plugins = ["aiida.manage.tests.pytest_fixtures"]

# Directory where to store outputs for known inputs (usually tests/data)
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_static")


@pytest.fixture(scope="function")
def mocked_pw67(mock_code_factory):
    """
    Create mocked "pw" code
    """
    return mock_code_factory(
        label="pw67",
        data_dir_abspath=DATA_DIR,
        entry_point="quantumespresso.pw",
        ignore_files=("_aiidasubmit.sh",),
    )


@pytest.fixture(scope="function")
def pp_silicon_sg15():
    """
    Create a aiida-pseudo pp data of sg15 silicon
    """
    from aiida import plugins

    UpfData = plugins.DataFactory("pseudo.upf")

    pp_name = "Si_ONCV_PBE-1.2.upf"
    pp_path = os.path.join(STATIC_DIR, pp_name)

    with open(pp_path, "rb") as stream:
        pseudo = UpfData(stream)

    yield pseudo
