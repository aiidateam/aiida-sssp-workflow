#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from aiida import orm
from aiida.common import AttributeDict
from aiida.engine import run_get_node, submit
from aiida.plugins import WorkflowFactory

from aiida_sssp_workflow.workflows.helper import helper_get_v0_b0_b1

ConvergencePressureWorkChain = WorkflowFactory("sssp_workflow.convergence.pressure")


def run_test(code, upf, dual):
    ecutwfc = np.array([30, 35, 40, 45, 50, 55, 60, 200])
    ecutrho = ecutwfc * dual
    PARA_ECUTWFC_LIST = orm.List(list=list(ecutwfc))
    PARA_ECUTRHO_LIST = orm.List(list=list(ecutrho))

    element = upf.element
    V0, B0, B1 = helper_get_v0_b0_b1(element)
    v0_b0_b1 = {
        "V0": V0,
        "B0": B0,
        "B1": B1,
    }

    inputs = AttributeDict(
        {
            "code": code,
            "pseudo": upf,
            "parameters": {
                "ecutwfc_list": PARA_ECUTWFC_LIST,
                "ecutrho_list": PARA_ECUTRHO_LIST,
                "ref_cutoff_pair": orm.List(list=[200, 200 * dual]),
                "v0_b0_b1": orm.Dict(dict=v0_b0_b1),
            },
        }
    )
    node = submit(ConvergencePressureWorkChain, **inputs)

    return node


if __name__ == "__main__":
    from aiida.orm import load_code, load_node

    code = load_code("qe-6.6-pw@daint-mc")

    upf_sg15 = {}
    # # sg15/Au_ONCV_PBE-1.2.upf
    # upf_sg15['au'] = load_node('2c467668-2f38-4a8c-8b57-69d67a3fb2a4')
    # sg15/Si_ONCV_PBE-1.2.upf
    upf_sg15["si"] = load_node("39e55083-3fc7-4405-8b3b-54a2c940dc67")

    for element, upf in upf_sg15.items():
        dual = 4.0
        node = run_test(code, upf, dual)
        node.description = f"sg15/{element}"
        print(node)
