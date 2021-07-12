#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Running delta factor workchain example
"""
import os

from aiida import orm

from aiida.plugins import WorkflowFactory, DataFactory
from aiida.engine import run_get_node

UpfData = DataFactory('pseudo.upf')
DeltaFactorWorkChain = WorkflowFactory('sssp_workflow.delta_factor')

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_static')

def run_delta(code, upf):
    inputs = {
        'code': code,
        'pseudo': upf,
        'protocol': orm.Str('test'),
        'options': orm.Dict(
                dict={
                    'resources': {
                        'num_machines': 1
                    },
                    'max_wallclock_seconds': 1800 * 3,
                    'withmpi': False,
                }),
    }

    res, node = run_get_node(DeltaFactorWorkChain, **inputs)
    return res, node


if __name__ == '__main__':
    from aiida.orm import load_code

    code = load_code('pw67@localhost')

    upf_sg15 = {}
    # sg15/Si_ONCV_PBE-1.2.upf
    pp_name = 'Si_ONCV_PBE-1.2.upf'
    pp_path = os.path.join(STATIC_DIR, pp_name)
    with open(pp_path, 'rb') as stream:
        pseudo = UpfData(stream)
        upf_sg15['si'] = pseudo

    for element, upf in upf_sg15.items():
        res, node = run_delta(code, upf)
        node.description = f'sg15/{element}'
        print(node)

    # # test on lanthanides
    # upf_wt = {}
    # # WT/La.GGA-PBE-paw-v1.0.UPF
    # upf_wt['La'] = load_node('b2880763-579c-4f6d-8803-2c77f4fb10e8')
    # # WT/Eu.GGA-PBE-paw-v1.0.UPF
    # upf_wt['Eu'] = load_node('220a8ebd-0ac5-44f1-a6a9-0790b24965a9')
    #
    # for element, upf in upf_wt.items():
    #     node = run_delta(code, upf, is_nc=False)
    #     node.description = f'WT/{element}-PBE'
    #     print(node)
