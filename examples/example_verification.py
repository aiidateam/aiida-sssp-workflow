#!/usr/bin/env python
import numpy as np

from aiida import orm
from aiida.common import AttributeDict
from aiida.plugins import WorkflowFactory
from aiida.engine import submit

VerificationWorkChain = WorkflowFactory('sssp_workflow.verification')


def run_test(pw_code, ph_code, upf, dual):
    ecutwfc = np.array(
        [30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 90, 100, 120, 150, 200])
    ecutrho = ecutwfc * dual
    PARA_ECUTWFC_LIST = orm.List(list=list(ecutwfc))
    PARA_ECUTRHO_LIST = orm.List(list=list(ecutrho))

    inputs = AttributeDict({
        'pw_code': pw_code,
        'ph_code': ph_code,
        'pseudo': upf,
        'parameters': {
            'ecutwfc_list': PARA_ECUTWFC_LIST,
            'ecutrho_list': PARA_ECUTRHO_LIST,
            'dual': orm.Float(dual),
        },
    })
    node = submit(VerificationWorkChain, **inputs)

    return node


if __name__ == '__main__':
    from aiida.orm import load_code, load_node

    pw_code = load_code('qe-6.6-pw@daint-mc')
    ph_code = load_code('qe-6.6-ph@daint-mc')

    # upf_gbrv = {}
    # # GBRV_pbe/si_pbe_v1.uspp.F.UPF
    # # upf_gbrv['si'] = load_node('4006d41d-5784-4e6c-898a-c4402c693522')
    # # # GBRV_pbe/fe_pbe_v1.5.uspp.F.UPF
    # # upf_gbrv['fe'] = load_node('a4493a3a-aef2-4937-b479-52e527dd68a0')
    # # # GBRV_pbe/f_pbe_v1.4.uspp.F.UPF
    # # upf_gbrv['f'] = load_node('d38ea9e2-1d81-4d43-94cd-69e700048a1b')
    # # # GBRV_pbe/au_pbe_v1.uspp.F.UPF
    # # upf_gbrv['au'] = load_node('c4c2d6e9-4523-4b67-a3f4-6159de83735b')
    #
    # for element, upf in upf_gbrv.items():
    #     if element == 'fe':
    #         dual = 12.0
    #     else:
    #         dual = 8.0
    #     node = run_test(pw_code, ph_code, upf, dual)
    #     node.description = f'GBRV_pbe/{element}'
    #     print(node)

    upf_sg15 = {}
    # # sg15/Au_ONCV_PBE-1.2.upf
    # upf_sg15['au'] = load_node('2c467668-2f38-4a8c-8b57-69d67a3fb2a4')
    # sg15/Si_ONCV_PBE-1.2.upf
    upf_sg15['si'] = load_node('39e55083-3fc7-4405-8b3b-54a2c940dc67')
    # # sg15/Xe_ONCV_PBE-1.2.upf
    # upf_sg15['xe'] = load_node('73c56fb5-c28a-4d9d-9f22-ad3b4b571069')
    # # sg15/Fe_ONCV_PBE-1.2.upf
    # upf_sg15['fe'] = load_node('ef7e8b79-13a7-4925-9f9a-fcb9f32e5d1b')
    # # sg15/F_ONCV_PBE-1.2.upf
    # upf_sg15['f'] = load_node('207fb79d-9d56-41f1-a0b6-594d62701f9e')

    for element, upf in upf_sg15.items():
        dual = 4.0
        node = run_test(pw_code, ph_code, upf, dual)
        node.description = f'sg15/{element}'
        print(node)
    #
    # # test on lanthanides
    # upf_wt = {}
    # # WT/La.GGA-PBE-paw-v1.0.UPF
    # upf_wt['La'] = load_node('b3d8c4c7-c48f-4170-a12b-ef67cf02117f')
    # # WT/Eu.GGA-PBE-paw-v1.0.UPF
    # upf_wt['Eu'] = load_node('5a621c8c-a908-4b7a-a363-db2e97d9a6fc')
    #
    # for element, upf in upf_wt.items():
    #     node = run_test(pw_code, ph_code, upf, dual=8.0)
    #     node.description = f'WT/{element}-PBE'
    #     print(node)
