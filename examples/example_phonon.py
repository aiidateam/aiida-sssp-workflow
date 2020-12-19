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

    upf_sg15 = {}
    # # sg15/Au_ONCV_PBE-1.2.upf
    upf_sg15['au'] = load_node('62e411c5-b0ab-4d08-875c-6fa4f74eb74e')
    # sg15/Si_ONCV_PBE-1.2.upf
    upf_sg15['si'] = load_node('98f04e42-6da8-4960-acfa-0161e0e339a5')

    for element, upf in upf_sg15.items():
        dual = 4.0
        node = run_test(pw_code, ph_code, upf, dual)
        node.description = f'sg15/{element}'
        print(node)
