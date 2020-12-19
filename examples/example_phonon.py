#!/usr/bin/env python
import numpy as np

from aiida import orm
from aiida.common import AttributeDict
from aiida.plugins import WorkflowFactory
from aiida.engine import submit

ConvergencePhononFrequencies = WorkflowFactory(
    'sssp_workflow.convergence.phonon_frequencies')


def run_test(pw_code, ph_code, upf, dual):
    ecutwfc = np.array([30, 35, 200])
    # ecutwfc = np.array(
    #     [30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 90, 100, 120, 150, 200])
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
            'ref_cutoff_pair': orm.List(list=[200, 200 * dual])
        },
    })
    node = submit(ConvergencePhononFrequencies, **inputs)

    return node


if __name__ == '__main__':
    from aiida.orm import load_code, load_node

    pw_code = load_code('qe-6.6-pw@daint-mc')
    ph_code = load_code('qe-6.6-ph@daint-mc')

    # upf_gbrv = {}
    # # GBRV_pbe/si_pbe_v1.uspp.F.UPF
    # # upf_gbrv['si'] = load_node(28953)
    # # # GBRV_pbe/fe_pbe_v1.5.uspp.F.UPF
    # # upf_gbrv['fe'] = load_node('cdd71d89-6ff8-4f54-a996-df0d37b39489')
    # # # GBRV_pbe/f_pbe_v1.4.uspp.F.UPF
    # # upf_gbrv['f'] = load_node('5963dba6-e1e9-4f8a-92c2-de84219f393f')
    # # # GBRV_pbe/au_pbe_v1.uspp.F.UPF
    # # upf_gbrv['au'] = load_node('51f62644-fb95-4b04-9c31-ee732b6d8155')
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
    upf_sg15['au'] = load_node('62e411c5-b0ab-4d08-875c-6fa4f74eb74e')
    # sg15/Si_ONCV_PBE-1.2.upf
    upf_sg15['si'] = load_node('98f04e42-6da8-4960-acfa-0161e0e339a5')
    # # sg15/Xe_ONCV_PBE-1.2.upf
    # upf_sg15['xe'] = load_node('3a3c2612-9cdc-4fc9-bd37-589fa87fab06')
    # # sg15/Fe_ONCV_PBE-1.2.upf
    # upf_sg15['fe'] = load_node('12aeff74-048d-4199-b97e-2b996e7fc2d3')
    # # sg15/F_ONCV_PBE-1.2.upf
    # upf_sg15['f'] = load_node('3ef1d267-b442-43fb-bc29-7113761ca15b')

    for element, upf in upf_sg15.items():
        dual = 4.0
        node = run_test(pw_code, ph_code, upf, dual)
        node.description = f'sg15/{element}'
        print(node)
    #
    # # test on lanthanides
    # upf_wt = {}
    # # WT/La.GGA-PBE-paw-v1.0.UPF
    # upf_wt['La'] = load_node('b2880763-579c-4f6d-8803-2c77f4fb10e8')
    # # WT/Eu.GGA-PBE-paw-v1.0.UPF
    # upf_wt['Eu'] = load_node('220a8ebd-0ac5-44f1-a6a9-0790b24965a9')
    #
    # for element, upf in upf_wt.items():
    #     node = run_test(pw_code, ph_code, upf, dual=8.0)
    #     node.description = f'WT/{element}-PBE'
    #     print(node)
