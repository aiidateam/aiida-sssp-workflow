#!/usr/bin/env python
"""
Running delta factor workchain example
"""
from aiida.common import AttributeDict
from aiida import orm

from aiida.plugins import WorkflowFactory
from aiida.engine import run_get_node, submit

DeltaFactorWorkChain = WorkflowFactory('sssp_workflow.delta_factor')


def run_delta(code, upf, is_nc=False):

    if is_nc:
        dual = 4
    else:
        dual = 8

    ecutwfc = 200.0
    ecutrho = ecutwfc * dual
    inputs = AttributeDict({
        'code':
        code,
        'pseudo':
        upf,
        'options':
        orm.Dict(
            dict={
                'resources': {
                    'num_machines': 1
                },
                'max_wallclock_seconds': 1800 * 3,
                'withmpi': True,
            }),
        'parameters': {
            'pw':
            orm.Dict(dict={
                'SYSTEM': {
                    'ecutwfc': ecutwfc,
                    'ecutrho': ecutrho,
                },
            })
        },
    })

    node = submit(DeltaFactorWorkChain, **inputs)
    return node


if __name__ == '__main__':
    from aiida.orm import load_code, load_node

    code = load_code('qe-6.6-pw@daint-mc')

    upf_sg15 = {}
    # # sg15/Au_ONCV_PBE-1.2.upf
    # upf_sg15['au'] = load_node('2c467668-2f38-4a8c-8b57-69d67a3fb2a4')
    # sg15/Si_ONCV_PBE-1.2.upf
    upf_sg15['si'] = load_node('39e55083-3fc7-4405-8b3b-54a2c940dc67')

    for element, upf in upf_sg15.items():
        node = run_delta(code, upf, is_nc=True)
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
