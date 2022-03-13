#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Running cohesive energy convergence workflow
"""
import os

from aiida import orm
from aiida.engine import run_get_node
from aiida.plugins import DataFactory

from aiida_sssp_workflow.utils import get_standard_cif_filename_from_element
from aiida_sssp_workflow.workflows.evaluate._phonon_frequencies import (
    PhononFrequenciesWorkChain,
)

UpfData = DataFactory("pseudo.upf")

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../", "_static")

pw_base_parameters = {
    "SYSTEM": {
        "degauss": 0.0045,
        "occupations": "smearing",
        "smearing": "cold",
    },
    "ELECTRONS": {
        "conv_thr": 1e-10,
    },
    "CONTROL": {
        "calculation": "scf",
        "wf_collect": True,
        "tstress": True,
    },
}

ph_base_parameters = {
    "INPUTPH": {
        "tr2_ph": 1e-16,
        "epsil": False,
    }
}


def get_structure_from_element(element="Si"):
    cif_file = get_standard_cif_filename_from_element(element)
    structure = orm.CifData.get_or_create(cif_file)[0].get_structure(
        primitive_cell=True
    )

    return structure


def run_phonon_frequencies_eva(pw_code, ph_code, pseudos, ecutwfc=30.0, ecutrho=120.0):
    inputs = {
        "pw_code": pw_code,
        "ph_code": ph_code,
        "pseudos": pseudos,
        "structure": get_structure_from_element("Si"),
        "pw_base_parameters": orm.Dict(dict=pw_base_parameters),
        "ph_base_parameters": orm.Dict(dict=ph_base_parameters),
        "ecutwfc": orm.Float(ecutwfc),
        "ecutrho": orm.Float(ecutrho),
        "qpoints": orm.List(
            list=[
                [0.5, 0.5, 0.5],
            ]
        ),
        "kpoints_distance": orm.Float(0.5),
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
    res, node = run_get_node(PhononFrequenciesWorkChain, **inputs)

    return res, node


if __name__ == "__main__":
    from aiida.orm import load_code

    pw_code = load_code("pw-6.7@localhost")
    ph_code = load_code("ph-6.7@localhost")

    upf_sg15 = {}
    # sg15/Si_ONCV_PBE-1.2.upf
    pp_name = "Si_ONCV_PBE-1.2.upf"
    pp_path = os.path.join(STATIC_DIR, pp_name)
    with open(pp_path, "rb") as stream:
        pseudo = UpfData(stream)
        upf_sg15["si"] = pseudo

    pseudos = {"Si": pseudo}
    res, node = run_phonon_frequencies_eva(pw_code, ph_code, pseudos)
    node.description = f"sg15/silicon"
    print(node)

    # TODO for evaluate wf it is always inportant to add test for multi element structure
