# -*- coding: utf-8 -*-
"""fixtures"""

import os
import pytest
from pathlib import Path
import uuid
import hashlib

from aiida import orm
from aiida.orm.utils.managers import NodeLinksManager
from aiida.engine import ProcessBuilder
from aiida_sssp_workflow.utils import serialize_data

pytest_plugins = ["aiida.manage.tests.pytest_fixtures"]

STATICS_DIR = Path(__file__).parent / "_statics"


@pytest.fixture
def generate_uuid():
    def _generate_uuid(seed="0"):
        return str(uuid.UUID(hashlib.md5(seed.encode()).hexdigest()))

    return _generate_uuid


@pytest.fixture(scope="function")
def code_generator(aiida_localhost):
    """For quantum espresso codes generator"""

    def _code_generator(bin):
        if bin == "pw":
            exec_path = "pw.x"
            plugin = "quantumespresso.pw"
        elif bin == "ph":
            exec_path = "ph.x"
            plugin = "quantumespresso.ph"
        else:
            raise ValueError(f"bin {bin} not supported")

        aiida_localhost.set_use_double_quotes(True)
        uid = os.getuid()
        gid = os.getgid()
        engine_command = """docker run -i -v $PWD:/workdir -w /workdir -u {uid}:{gid} {{image_name}} sh -c""".format(
            uid=uid, gid=gid
        )
        code = orm.ContainerizedCode(
            label=f"{bin}-docker",
            default_calc_job_plugin=plugin,
            filepath_executable=exec_path,
            image_name="ghcr.io/cnts4sci/quantum-espresso:edge",
            wrap_cmdline_params=True,
            engine_command=engine_command,
            use_double_quotes=True,
            computer=aiida_localhost,
        ).store()

        return code

    return _code_generator


@pytest.fixture(scope="function")
def pseudo_path():
    def _pseudo_path(pseudo="Al.paw"):
        if pseudo == "Al.paw":
            path = STATICS_DIR / "upf" / "Al.paw.pbe.z_3.ld1.psl.v0.1.upf"
        elif pseudo == "O.nc":
            path = STATICS_DIR / "upf" / "O.nc.pbe.z_6.oncvpsp3.dojo.v0.4.1-std.upf"
        elif pseudo == "O.paw":
            path = STATICS_DIR / "upf" / "O.paw.pbe.z_6.atompaw.jth.v1.1-std.upf"
        else:
            raise ValueError(f"pseudo {pseudo} not found")

        return path

    return _pseudo_path


@pytest.fixture
def serialize_inputs():
    """Serialize the given process inputs into a dictionary with nodes turned into their value representation.
    (Borrowed from aiida-quantumespresso/tests/conftest.py::serialize_builder)

    :param input: the process inputs of type NodeManegerLink to serialize
    :return: dictionary
    """

    def _serialize_inputs(inputs: NodeLinksManager):
        # NodeLinksManager -> dict
        _inputs = {}
        for key in inputs._get_keys():
            _inputs[key] = inputs[key]

        return serialize_data(_inputs)

    return _serialize_inputs


@pytest.fixture
def serialize_builder():
    """Serialize the builder into a dictionary with nodes turned into their value representation.
    (Borrowed from aiida-quantumespresso/tests/conftest.py::serialize_builder)

    :param builder: the process builder to serialize
    :return: dictionary
    """

    def _serialize_builder(builder: ProcessBuilder):
        return serialize_data(builder._inputs(prune=True))

    return _serialize_builder
