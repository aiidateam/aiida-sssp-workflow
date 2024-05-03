# -*- coding: utf-8 -*-
"""fixtures"""

import os

import pytest
from aiida import orm
from aiida.plugins import DataFactory

pytest_plugins = ["aiida.manage.tests.pytest_fixtures"]

# Directory where to store outputs for known inputs (usually tests/data)
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_static")


@pytest.fixture(scope="function")
def psp_Si_SG15():
    """
    Create a aiida-pseudo pp data of sg15 silicon
    """
    UpfData = DataFactory("pseudo.upf")

    pp_name = "Si_ONCV_PBE-1.2.upf"
    pp_path = os.path.join(STATIC_DIR, pp_name)

    with open(pp_path, "rb") as stream:
        pseudo = UpfData(stream)

    yield pseudo


@pytest.fixture(scope="function")
def pw_code(aiida_localhost):
    aiida_localhost.set_use_double_quotes(True)
    engine_command = """singularity exec --bind $PWD:$PWD {image_name}"""
    code = orm.ContainerizedCode(
        default_calc_job_plugin="quantumespresso.pw",
        filepath_executable="pw.x",
        engine_command=engine_command,
        image_name="docker://containers4hpc/qe-mpich314:0.1.0",
        computer=aiida_localhost,
    ).store()

    yield code


@pytest.fixture(scope="function")
def ph_code(aiida_localhost):
    aiida_localhost.set_use_double_quotes(True)
    engine_command = """singularity exec --bind $PWD:$PWD {image_name}"""
    code = orm.ContainerizedCode(
        default_calc_job_plugin="quantumespresso.ph",
        filepath_executable="ph.x",
        engine_command=engine_command,
        image_name="docker://containers4hpc/qe-mpich314:0.1.0",
        computer=aiida_localhost,
    ).store()

    yield code
