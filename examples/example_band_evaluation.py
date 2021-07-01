#!/usr/bin/env python
# -*- coding: utf-8 -*-

from aiida import orm
from aiida.common import AttributeDict
from aiida.engine import submit
from aiida.plugins import WorkflowFactory

from aiida_sssp_workflow.helpers import get_pw_inputs_from_pseudo

BandsWorkChain = WorkflowFactory('sssp_workflow.evaluation.bands')


def run_test(code, pseudo, dual):
    res = get_pw_inputs_from_pseudo(pseudo=pseudo)

    structure = res['structure']
    pseudos = res['pseudos']

    inputs = AttributeDict({
        'code': code,
        'pseudos': pseudos,
        'structure': structure,
        'parameters': {
            'ecutwfc': orm.Float(200),
            'ecutrho': orm.Float(200 * dual),
            'run_band_structure': orm.Bool(True),
            'nbands_factor': orm.Float(2)
        },
    })
    node = submit(BandsWorkChain, **inputs)

    return node


if __name__ == '__main__':
    from aiida.orm import load_code, load_node

    code = load_code('qe-6.5-pw@daint-mc')

    upf_sg15 = {}
    # # sg15/Au_ONCV_PBE-1.2.upf
    # upf_sg15['au'] = load_node('2c467668-2f38-4a8c-8b57-69d67a3fb2a4')
    # sg15/Si_ONCV_PBE-1.2.upf
    upf_sg15['si'] = load_node('39e55083-3fc7-4405-8b3b-54a2c940dc67')

    for element, upf in upf_sg15.items():
        dual = 4.0
        node = run_test(code, upf, dual)
        node.description = f'sg15/{element}'
        print(node)
