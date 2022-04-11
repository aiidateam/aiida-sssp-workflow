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


def run_verification(pw_code, ph_code, upf, label, is_submit=True, clean_level=9):
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
                "accuracy:delta",
                "accuracy:bands",
                "convergence:cohesive_energy",
                "convergence:phonon_frequencies",
                "convergence:pressure",
                "convergence:delta",
                "convergence:bands",
            ]
        ),
        "options": orm.Dict(
            dict={
                "resources": {
                    "num_machines": 1,
                    "num_mpiprocs_per_machine": 32,
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

    pw_code = load_code("pw-6.7@imxgesrv1")
    ph_code = load_code("ph-6.7@imxgesrv1")

    pp_path = os.path.join(STATIC_DIR, element, os.path.basename(fn))
    with open(pp_path, "rb") as stream:
        pseudo = UpfData(stream)

    node = run_verification(
        pw_code, ph_code, pseudo, label, is_submit=True, clean_level=9
    )
    node.description = label
    print(node)

# verdi run demo/verification.py Si demo/_static/Si/Si.pbe-n-kjpaw_psl.0.1.UPF si/paw/z=4/psl/v0.1
# verdi run demo/verification.py Si demo/_static/Si/Si_ONCV_PBE-1.2.upf si/nc/z=4/sg15/v1.2-o2
# verdi run demo/verification.py Si demo/_static/Si/Si.sg15-v1.2-oncv4.upf si/nc/z=4/sg15/v1.2-o4
# verdi run demo/verification.py Si demo/_static/Si/Si.dojo-sr-04-std.upf si/nc/z=4/dojo/v04
# verdi run demo/verification.py Mg demo/_static/Mg/Mg_ONCV_PBE-1.2.upf mg/nc/z=10/sg15/v1.2-o2
# verdi run demo/verification.py Mg demo/_static/Mg/Mg.sg15-v1.2-oncv4.upf mg/nc/z=10/sg15/v1.2-o4
# verdi run demo/verification.py Mg demo/_static/Mg/Mg.dojo-sr-04-std.upf mg/nc/z=10/dojo/v04
# verdi run demo/verification.py Mg demo/_static/Mg/Mg.pbe-spn-kjpaw_psl.1.0.0.UPF mg/paw/z=10/psl/v1.0.0
