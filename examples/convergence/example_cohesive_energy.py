#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Running cohesive energy convergence workflow
"""
import os

import numpy as np
from aiida import orm
from aiida.engine import run_get_node
from aiida.plugins import DataFactory, WorkflowFactory

UpfData = DataFactory("pseudo.upf")
ConvergenceCohesiveEnergy = WorkflowFactory("sssp_workflow.convergence.cohesive_energy")

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../", "_static")


def run_cohesive_cov(code, upf, dual):
    inputs = {
        "code": code,
        "pseudo": upf,
        "protocol": orm.Str("test"),
        "dual": orm.Float(4.0),
        "options": orm.Dict(
            dict={
                "resources": {"num_machines": 1},
                "max_wallclock_seconds": 1800 * 3,
                "withmpi": False,
            }
        ),
        "parallelization": orm.Dict(dict={}),
        "clean_workdir": orm.Bool(True),
    }
    res, node = run_get_node(ConvergenceCohesiveEnergy, **inputs)

    return res, node


if __name__ == "__main__":
    from aiida import load_profile
    from aiida.orm import load_code

    load_profile("sssp-dev")
    code = load_code("pw64@localhost")

    upf_sg15 = {}
    # sg15/Si_ONCV_PBE-1.2.upf
    pp_name = "Si_ONCV_PBE-1.2.upf"
    pp_path = os.path.join(STATIC_DIR, pp_name)
    with open(pp_path, "rb") as stream:
        pseudo = UpfData(stream)
        upf_sg15["si"] = pseudo

    for element, upf in upf_sg15.items():
        res, node = run_cohesive_cov(code, upf)
        node.description = f"sg15/{element}"
        print(node)
