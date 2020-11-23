#!/usr/bin/env python
"""
Running delta factor workchain example
"""
from aiida.common import AttributeDict
from aiida import orm

from aiida.plugins import WorkflowFactory
from aiida.engine import run_get_node, submit

DeltaFactorWorkChain = WorkflowFactory('sssp_workflow.delta_factor')

def run_delta(code, upf, structure=None, is_nc=False):

    if is_nc:
        dual=4
    else:
        dual=8

    ecutwfc = 200
    ecutrho = ecutwfc * dual
    inputs = AttributeDict({
        'code': code,
        'pseudo': upf,
        'options': orm.Dict(dict={
            'resources': {'num_machines': 1},
            'max_wallclock_seconds': 1800,
            'withmpi': True,
        }),
        'parameters': {
            'scale_count': orm.Int(7),
            'scale_increment': orm.Float(0.02),
            'kpoints_distance': orm.Float(0.1),
            'pw': orm.Dict(dict={
                'SYSTEM': {
                    'ecutwfc': ecutwfc,
                    'ecutrho': ecutrho,
                },
            })
        },
    })

    if structure:
        inputs.update({'structure': structure})

    node = submit(DeltaFactorWorkChain, **inputs)
    return node



if __name__ == '__main__':
    from aiida.orm import load_code, load_node

    code = load_code('qe-6.6-pw@daint-mc')

    # # GBRV_pbe/au_pbe_v1.uspp.F.UPF
    # upf = load_node('5b2176f8-0d40-44c9-a902-f260c94d62ca')
    #
    # node = run_delta(code, upf, is_nc=False)
    # node.description = 'GBRV_pbe/au_pbe_v1.uspp.F.UPF'
    # print(node)
    #
    # # Au_ONCV_PBE-1.0.oncvpsp.upf
    # upf = load_node('197acb08-ff93-4e65-8de9-2242a96197b2')
    #
    # # maximum inputs
    # node = run_delta(code, upf, is_nc=True)
    # node.description = 'Au_ONCV_PBE-1.0.oncvpsp.upf'
    # print(node)

    # # GBRV_pbe/si_pbe_v1.uspp.F.UPF
    # upf = load_node('65f0427b-4cc5-4311-9f22-6a29a7771e1f')
    #
    # node = run_delta(code, upf, is_nc=False)
    # node.description = 'GBRV_pbe/si_pbe_v1.uspp.F.UPF'
    # print(node)

    # # sg15/Si_ONCV_PBE-1.2.upf
    # upf = load_node('98f04e42-6da8-4960-acfa-0161e0e339a5')
    #
    # node = run_delta(code, upf, is_nc=True)
    # node.description = 'sg15/Si_ONCV_PBE-1.2.upf'
    # print(node)

    #
    #
    # # Si.pbe-n-rrkjus_psl.1.0.0.UPF
    # upf = load_node('9d9d57fc-49e3-4e0c-8c37-4682ccc0fb51')
    #
    # node = run_delta(code, upf, is_nc=False)
    # node.description = 'Si.pbe-n-rrkjus_psl.1.0.0.UPF'
    # print(node)

    # # sg15/Au_ONCV_PBE-1.2.upf
    # upf = load_node('62e411c5-b0ab-4d08-875c-6fa4f74eb74e')
    #
    # # maximum inputs
    # node = run_delta(code, upf, is_nc=True)
    # node.description = 'sg15/Au_ONCV_PBE-1.2.upf'
    # print(node)
    #
    # # sg15/Au_ONCV_PBE-1.0.upf
    # upf = load_node('6be3e57a-bca0-4b84-aff8-7682e17eb9c9')
    #
    # # maximum inputs
    # node = run_delta(code, upf, is_nc=True)
    # node.description = 'sg15/Au_ONCV_PBE-1.0.upf'
    # print(node)
    #
    # # sg15/Au_ONCV_PBE_FR-1.0.upf
    # upf = load_node('c4594201-f325-4c0b-890c-18ad051c9c57')
    #
    # # maximum inputs
    # node = run_delta(code, upf, is_nc=True)
    # node.description = 'sg15/Au_ONCV_PBE_FR-1.0.upf'
    # print(node)

    # La.GGA-PBE-paw-v1.0.UPF
    # upf = load_node('a8302677-a72a-4775-8b6a-66b8f6283ff3')
    #
    # # maximum inputs
    # node = run_delta(code, upf, is_nc=False)
    # node.description = 'La.GGA-PBE-paw-v1.0.UPF'
    # print(node)

    # sg15/O_ONCV_PBE-1.2.upf
    # upf = load_node('804b71ea-6d9b-4d80-a343-fd38f8d59382')
    #
    # # maximum inputs
    # node = run_delta(code, upf, is_nc=True)
    # node.description = 'sg15/O_ONCV_PBE-1.2.upf'
    # print(node)

    # GBRV_pbe/o_pbe_v1.2.uspp.F.UPF
    upf = load_node('be9781ac-7578-4cef-8c53-68173b3eff9a')

    # maximum inputs
    node = run_delta(code, upf, is_nc=False)
    node.description = 'GBRV_pbe/o_pbe_v1.2.uspp.F.UPF'
    print(node)

    #
    # # sg15/Mn_ONCV_PBE-1.2.upf
    # upf = load_node('8daf2779-b72b-49d0-98b1-7bed7f86a9b9')
    #
    # # maximum inputs
    # node = run_delta(code, upf, is_nc=True)
    # node.description = 'sg15/Mn_ONCV_PBE-1.2.upf'
    # print(node)


