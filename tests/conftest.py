# -*- coding: utf-8 -*-
"""fixtures"""

import os
import pytest
from pathlib import Path

from aiida import orm
from aiida.orm.utils.managers import NodeLinksManager
from aiida.engine import ProcessBuilder

pytest_plugins = ["aiida.manage.tests.pytest_fixtures"]

STATICS_DIR = Path(__file__).parent / "_statics"


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
            image_name="ghcr.io/containers4hpc/quantum-espresso:v2024.1001",
            wrap_cmdline_params=True,
            engine_command=engine_command,
            use_double_quotes=True,
            computer=aiida_localhost,
        ).store()

        return code

    return _code_generator


@pytest.fixture(scope="function")
def pseudo_path():
    def _pseudo_path(element="Al"):
        if element == "Al":
            path = STATICS_DIR / "upf" / "Al.paw.pbe.z_3.ld1.psl.v0.1.upf"
        else:
            raise ValueError(f"pseudo for {element} not found")

        return path

    return _pseudo_path


def _serialize_data(data):
    from aiida.orm import (
        AbstractCode,
        BaseType,
        Data,
        Dict,
        KpointsData,
        List,
        RemoteData,
        SinglefileData,
    )
    from aiida.plugins import DataFactory

    StructureData = DataFactory("core.structure")
    UpfData = DataFactory("pseudo.upf")

    if isinstance(data, dict):
        return {key: _serialize_data(value) for key, value in data.items()}

    if isinstance(data, BaseType):
        return data.value

    if isinstance(data, AbstractCode):
        return data.full_label

    if isinstance(data, Dict):
        return data.get_dict()

    if isinstance(data, List):
        return data.get_list()

    if isinstance(data, StructureData):
        return data.get_formula()

    if isinstance(data, UpfData):
        return f"{data.element}<md5={data.md5}>"

    if isinstance(data, RemoteData):
        # For `RemoteData` we compute the hash of the repository. The value returned by `Node._get_hash` is not
        # useful since it includes the hash of the absolute filepath and the computer UUID which vary between tests
        return data.base.repository.hash()

    if isinstance(data, KpointsData):
        try:
            return data.get_kpoints().tolist()
        except AttributeError:
            return data.get_kpoints_mesh()

    if isinstance(data, SinglefileData):
        return data.get_content()

    if isinstance(data, Data):
        return data.base.caching._get_hash()

    return data


@pytest.fixture
def serialize_inputs():
    """Serialize the given process inputs into a dictionary with nodes turned into their value representation.
    (Borrowed from aiida-quantumespresso/tests/conftest.py::serialize_builder)

    :param input: the process inputs of type NodeManegerLink to serialize
    :return: dictionary
    """

    def _serialize_data(data):
        from aiida.orm import (
            AbstractCode,
            BaseType,
            Data,
            Dict,
            KpointsData,
            List,
            RemoteData,
            SinglefileData,
        )
        from aiida.plugins import DataFactory

        StructureData = DataFactory("core.structure")
        UpfData = DataFactory("pseudo.upf")

        if isinstance(data, dict):
            return {key: _serialize_data(value) for key, value in data.items()}

        if isinstance(data, BaseType):
            return data.value

        if isinstance(data, AbstractCode):
            return data.full_label

        if isinstance(data, Dict):
            return data.get_dict()

        if isinstance(data, List):
            return data.get_list()

        if isinstance(data, StructureData):
            return data.get_formula()

        if isinstance(data, UpfData):
            return f"{data.element}<md5={data.md5}>"

        if isinstance(data, RemoteData):
            # For `RemoteData` we compute the hash of the repository. The value returned by `Node._get_hash` is not
            # useful since it includes the hash of the absolute filepath and the computer UUID which vary between tests
            return data.base.repository.hash()

        if isinstance(data, KpointsData):
            try:
                return data.get_kpoints().tolist()
            except AttributeError:
                return data.get_kpoints_mesh()

        if isinstance(data, SinglefileData):
            return data.get_content()

        if isinstance(data, Data):
            return data.base.caching._get_hash()

        return data

    def _serialize_inputs(inputs: NodeLinksManager):
        # NodeLinksManager -> dict
        _inputs = {}
        for key in inputs._get_keys():
            _inputs[key] = inputs[key]

        return _serialize_data(_inputs)

    return _serialize_inputs


@pytest.fixture
def serialize_builder():
    """Serialize the builder into a dictionary with nodes turned into their value representation.
    (Borrowed from aiida-quantumespresso/tests/conftest.py::serialize_builder)

    :param builder: the process builder to serialize
    :return: dictionary
    """

    def _serialize_builder(builder: ProcessBuilder):
        return _serialize_data(builder._inputs(prune=True))

    return _serialize_builder
