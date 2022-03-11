#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Running cohesive energy convergence workflow
"""
import os

import numpy as np
from aiida import orm
from aiida.engine import run_get_node
from aiida.plugins import DataFactory

from aiida_sssp_workflow.utils import get_standard_cif_filename_from_element
from aiida_sssp_workflow.workflows.evaluate._cohesive_energy import (
    CohesiveEnergyWorkChain,
)

UpfData = DataFactory("pseudo.upf")

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../", "_static")

bulk_parameters = {
    "SYSTEM": {
        "degauss": 0.00735,
        "occupations": "smearing",
        "smearing": "marzari-vanderbilt",
    },
    "ELECTRONS": {
        "conv_thr": 1.0e-8,
    },
}

atom_parameters = {
    "SYSTEM": {
        "degauss": 0.00735,
        "occupations": "smearing",
        "smearing": "gaussian",
    },
    "ELECTRONS": {
        "conv_thr": 1.0e-8,
    },
}


def get_structure_from_element(element="Si"):
    cif_file = get_standard_cif_filename_from_element(element)
    structure = orm.CifData.get_or_create(cif_file)[0].get_structure(
        primitive_cell=True
    )

    return structure


def run_cohesive_eva(code, pseudos, ecutwfc=30.0, ecutrho=120.0):
    inputs = {
        "code": code,
        "pseudos": pseudos,
        "structure": get_structure_from_element("Si"),
        "bulk_parameters": orm.Dict(dict=bulk_parameters),
        "atom_parameters": orm.Dict(dict=atom_parameters),
        "ecutwfc": orm.Float(ecutwfc),
        "ecutrho": orm.Float(ecutrho),
        "kpoints_distance": orm.Float(0.5),
        "vacuum_length": orm.Float(12.0),
        "options": orm.Dict(
            dict={
                "resources": {"num_machines": 1},
                "max_wallclock_seconds": 1800 * 3,
                "withmpi": False,
            }
        ),
        "parallelization": orm.Dict(dict={}),
        "clean_workdir": orm.Bool(False),
    }
    res, node = run_get_node(CohesiveEnergyWorkChain, **inputs)

    return res, node


if __name__ == "__main__":
    from aiida import load_profile
    from aiida.orm import load_code

    code = load_code("pw-6.7@localhost")

    upf_sg15 = {}
    # sg15/Si_ONCV_PBE-1.2.upf
    pp_name = "Si_ONCV_PBE-1.2.upf"
    pp_path = os.path.join(STATIC_DIR, pp_name)
    with open(pp_path, "rb") as stream:
        pseudo = UpfData(stream)
        upf_sg15["si"] = pseudo

    pseudos = {"Si": pseudo}
    res, node = run_cohesive_eva(code, pseudos)
    node.description = f"sg15/silicon"
    print(node)

    # TODO for evaluate wf it is always inportant to add test for multi element structure
