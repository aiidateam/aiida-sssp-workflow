# -*- coding: utf-8 -*-
"""fixtures"""

import os
import pytest
from pathlib import Path

from aiida import orm

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
            default_calc_job_plugin=plugin,
            filepath_executable=exec_path,
            image_name="container4hpc/qe-mpich314:0.1.0",
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
