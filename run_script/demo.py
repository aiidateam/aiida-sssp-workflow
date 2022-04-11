#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Running verification workchain example
"""
import os
import sys

from aiida import orm
from aiida.engine import run_get_node, submit
from aiida.plugins import DataFactory, WorkflowFactory

UpfData = DataFactory("pseudo.upf")
VerificationWorkChain = WorkflowFactory("sssp_workflow.verification")

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_static")


def run_verification(
    pw_code, ph_code, upf, label, is_submit=True, clean_level=9, num_procs=32
):
    inputs = {
        "pw_code": pw_code,
        "ph_code": ph_code,
        "pseudo": upf,
        "protocol": orm.Str("demo"),
        "criteria": orm.Str("efficiency"),
        "cutoff_control": orm.Str("demo"),
        "label": orm.Str(label),
        "properties_list": orm.List(
            list=[
                # "accuracy:delta",
                # "accuracy:bands",
                # "convergence:cohesive_energy",
                # "convergence:phonon_frequencies",
                # "convergence:pressure",
                # "convergence:delta",
                "convergence:bands",
            ]
        ),
        "options": orm.Dict(
            dict={
                "resources": {
                    "num_machines": 1,
                    "num_mpiprocs_per_machine": num_procs,
                },
                "max_wallclock_seconds": 1800,
                "withmpi": True,
            }
        ),
        # 'parallelization': orm.Dict(dict={}),
        "clean_workdir_level": orm.Int(clean_level),
    }

    if is_submit:
        node = submit(VerificationWorkChain, **inputs)
        return node
    else:
        res, node = run_get_node(VerificationWorkChain, **inputs)
        return node


if __name__ == "__main__":
    from aiida.orm import load_code

    try:
        element = sys.argv[1]
    except:
        raise

    try:
        # can be the filename or with the path
        fn = sys.argv[2]
    except:
        raise

    try:
        label = sys.argv[3]
    except:
        raise

    try:
        computer = sys.argv[4]
    except:
        raise

    if computer == "localhost":
        num_procs = 2
        is_submit = False
        clean_level = 0
    else:
        num_procs = 32
        is_submit = True
        clean_level = 9

    pw_code = load_code(f"pw-6.7@{computer}")
    ph_code = load_code(f"ph-6.7@{computer}")

    pp_path = os.path.join(STATIC_DIR, element, os.path.basename(fn))
    with open(pp_path, "rb") as stream:
        pseudo = UpfData(stream)

    node = run_verification(
        pw_code,
        ph_code,
        pseudo,
        label,
        is_submit=is_submit,
        clean_level=clean_level,
        num_procs=num_procs,
    )
    node.description = label
    print(node)

# verdi run run_script/demo.py Si run_script/_static/Si/Si.pbe-n-kjpaw_psl.0.1.UPF si/paw/z=4/psl/v0.1 localhost
# verdi run run_script/demo.py Mg run_script/_static/Mg/Mg.dojo-sr-04-std.upf mg/nc/z=10/dojo/v04 localhost
