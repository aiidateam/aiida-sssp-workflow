#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Running verification workchain example
"""
import os
import sys

from aiida import orm
from aiida.engine import run_get_node
from aiida.plugins import DataFactory, WorkflowFactory

from aiida_sssp_workflow.workflows.verifications import DEFAULT_PROPERTIES_LIST

UpfData = DataFactory("pseudo.upf")
VerificationWorkChain = WorkflowFactory("sssp_workflow.verification")

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_static")


def run_verification(
    pw_code, ph_code, upf, properties_list=DEFAULT_PROPERTIES_LIST, label=None
):
    inputs = {
        "accuracy": {
            "protocol": orm.Str("test"),
            "cutoff_control": orm.Str("test"),
        },
        "convergence": {
            "protocol": orm.Str("test"),
            "cutoff_control": orm.Str("test"),
            "criteria": orm.Str("efficiency"),
            # "preset_ecutwfc": orm.Int(60),
        },
        "pw_code": pw_code,
        "ph_code": ph_code,
        "pseudo": upf,
        "properties_list": orm.List(list=properties_list),
        "label": orm.Str(label),
        "options": orm.Dict(
            dict={
                "resources": {
                    "num_machines": 1,
                    "num_mpiprocs_per_machine": 1,
                },
                "max_wallclock_seconds": 1800 * 3,
                "withmpi": True,
            }
        ),
        "parallelization": orm.Dict(dict={}),
        "clean_workdir_level": orm.Int(1),
    }

    res, node = run_get_node(VerificationWorkChain, **inputs)
    return res, node


if __name__ == "__main__":
    from aiida.orm import load_code

    try:
        element = sys.argv[1]
    except:
        raise ("element please.")

    properties_list = []
    for property in DEFAULT_PROPERTIES_LIST:
        if property in sys.argv[2:]:
            properties_list.append(property)

    pw_code = load_code("pw-6.7@localhost")
    ph_code = load_code("ph-6.7@localhost")

    if element == "Mg":
        pp_label = "psl/Mg.pbe-spn-kjpaw_psl.1.0.0.UPF"
    elif element == "Fe":
        pp_label = "psl/Fe.pbe-spn-kjpaw_psl.0.2.1.UPF"
    elif element == "O":
        pp_label = "psl/O.pbe-n-kjpaw_psl.1.0.0.UPF"
    elif element == "Er":
        pp_label = "Wentzcovitch/Er.GGA-PBE-paw-v1.0.UPF"
    elif element == "La":
        pp_label = "Wentzcovitch/La.GGA-PBE-paw-v1.0.UPF"
    elif element == "Hf":
        pp_label = "psl/Hf.pbe-spn-rrkjus_psl.1.0.0.UPF"
    else:
        pp_label = "psl/Si.pbe-n-rrkjus_psl.1.0.0.UPF"

    pp_name = pp_label.split("/")[1]
    pp_path = os.path.join(STATIC_DIR, pp_name)
    with open(pp_path, "rb") as stream:
        pseudo = UpfData(stream)

    res, node = run_verification(pw_code, ph_code, pseudo, properties_list, pp_label)
    node.description = pp_label
    print(node)
