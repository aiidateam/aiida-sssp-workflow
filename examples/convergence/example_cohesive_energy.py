#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import os

from aiida import orm
from aiida.plugins import WorkflowFactory, DataFactory

from aiida.engine import run_get_node

UpfData = DataFactory('pseudo.upf')
ConvergenceCohesiveEnergy = WorkflowFactory(
    'sssp_workflow.convergence.cohesive_energy')

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../', '_static')


def run_test(code, upf, dual):
    ecutwfc = [30, 35, 40]
    ecutrho = list(np.array(ecutwfc) * dual)
    PARA_ECUTWFC_LIST = orm.List(list=ecutwfc)
    PARA_ECUTRHO_LIST = orm.List(list=ecutrho)

    inputs = {
        'code': code,
        'pseudo': upf,
        'parameters': {
            'ecutwfc_list': PARA_ECUTWFC_LIST,
            'ecutrho_list': PARA_ECUTRHO_LIST,
            'ref_cutoff_pair': orm.List(list=[40, 40 * dual])
        },
    }
    res, node = run_get_node(ConvergenceCohesiveEnergy, **inputs)

    return res, node


if __name__ == '__main__':
    from aiida.orm import load_code, load_node

    code = load_code('pw67@localhost')

    upf_sg15 = {}
    # sg15/Si_ONCV_PBE-1.2.upf
    pp_name = 'Si_ONCV_PBE-1.2.upf'
    pp_path = os.path.join(STATIC_DIR, pp_name)
    with open(pp_path, 'rb') as stream:
        pseudo = UpfData(stream)
        upf_sg15['si'] = pseudo

    for element, upf in upf_sg15.items():
        dual = 4.0
        res, node = run_test(code, upf, dual)
        node.description = f'sg15/{element}'
        print(node)
