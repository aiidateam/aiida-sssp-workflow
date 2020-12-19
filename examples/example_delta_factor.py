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

    ecutwfc = 200
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

    # upf_gbrv = {}
    # # GBRV_pbe/au_pbe_v1.uspp.F.UPF
    # upf_gbrv['au'] = load_node('51f62644-fb95-4b04-9c31-ee732b6d8155')
    # # GBRV_pbe/o_pbe_v1.2.uspp.F.UPF
    # upf_gbrv['o'] = load_node('005f1eaa-6746-41f2-8b12-38f6dabe4f61')
    # # GBRV_pbe/si_pbe_v1.uspp.F.UPF
    # upf_gbrv['si'] = load_node('15f938a1-9466-4a41-9022-4c6f06cf4e20')
    #
    # for element, upf in upf_gbrv.items():
    #     node = run_delta(code, upf, is_nc=False)
    #     node.description = f'GBRV_pbe/{element}'
    #     print(node)

    upf_sg15 = {}
    # sg15/Au_ONCV_PBE-1.2.upf
    # upf_sg15['au'] = load_node('62e411c5-b0ab-4d08-875c-6fa4f74eb74e')
    # # sg15/O_ONCV_PBE-1.2.upf
    # upf_sg15['o'] = load_node('52e4e647-d6ef-43c6-b8a1-adfb54bff07e')
    # sg15/Si_ONCV_PBE-1.2.upf
    upf_sg15['si'] = load_node('98f04e42-6da8-4960-acfa-0161e0e339a5')
    # # sg15/Xe_ONCV_PBE-1.2.upf
    # upf_sg15['xe'] = load_node('3a3c2612-9cdc-4fc9-bd37-589fa87fab06')

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
