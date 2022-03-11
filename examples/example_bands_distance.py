# -*- coding: utf-8 -*-
"""
Example of bands distance to chessboard comparision
"""
import os

from aiida import orm
from aiida.engine import run_get_node
from aiida.plugins import DataFactory, WorkflowFactory

from aiida_sssp_workflow.utils import to_valid_key

UpfData = DataFactory("pseudo.upf")
BandsDistanceWorkChain = WorkflowFactory("sssp_workflow.bands_distance")

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_static")


def run_bands_chessboard(code, d_input_upfs):
    inputs = {
        "code": code,
        "input_pseudos": d_input_upfs,
        "protocol": orm.Str("test"),
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

    res, node = run_get_node(BandsDistanceWorkChain, **inputs)
    return res, node


if __name__ == "__main__":
    from os.path import basename, splitext

    from aiida.orm import load_code

    code = load_code("pw64@localhost")

    d_input_upfs = {}
    pp_name_list = [
        "Si_ONCV_PBE-1.2.upf",
        "Si.pbe-n-kjpaw_psl.1.0.0.UPF",
        "Si.pbe-nl-rrkjus_psl.1.0.0.UPF",
    ]
    for pp_name in pp_name_list:
        pp_path = os.path.join(STATIC_DIR, pp_name)
        with open(pp_path, "rb") as stream:
            pseudo = UpfData(stream)
            name = to_valid_key(pp_name)
            d_input_upfs[name] = pseudo

    res, node = run_bands_chessboard(code, d_input_upfs)
    node.description = "Silicon"
    print(node)
