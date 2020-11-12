#!/usr/bin/env python
from aiida.common import AttributeDict
from aiida.plugins import WorkflowFactory
from aiida.engine import submit
from aiida import orm
from aiida.orm import load_code, load_node

from aiida_sssp_workflow.calculations.helper_functions import helper_get_primitive_structure

PressueEvaluationWorkChain = WorkflowFactory('sssp_workflow.pressure_evaluation')

def pressure_evaluate(ecutwfc, ecutrho):
    code = load_code('qe-6.6-pw@daint-mc')

    # Silicon structure and pseudopotential
    structure = load_node('0c5793b9-e52c-4708-b5a1-f53b5c4814bf')
    upf = load_node('9d9d57fc-49e3-4e0c-8c37-4682ccc0fb51')

    primitive_structure = helper_get_primitive_structure(structure)

    PW_PARAS = orm.Dict(
        dict={
            'SYSTEM': {
                'degauss': 0.01,
                'ecutrho': ecutrho,
                'ecutwfc': ecutwfc,
                'occupations': 'smearing',
                'smearing': 'marzari-vanderbilt',
            },
            'ELECTRONS': {
                'conv_thr': 1e-10,
            },
        })

    # minimal inputs, maximum wc
    inputs = AttributeDict({
        'code': code,
        'pseudo': upf,
        'structure': primitive_structure,
        'parameters': {
            'pw': PW_PARAS,
        },
    })
    node = submit(PressueEvaluationWorkChain, **inputs)

    return node

if __name__ == '__main__':
    # silicon pressure convergence test
    ecutwfc = 30
    for ecutrho in [280,290,300]:
        # ecutrho = ecutwfc * 6
        node = pressure_evaluate(ecutwfc=ecutwfc, ecutrho=ecutrho)
        node.description = f'[Si] ecutwfc={ecutwfc} ecutrho={ecutrho}'
        print(node.pk)