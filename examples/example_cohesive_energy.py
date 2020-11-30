#!/usr/bin/env python
from aiida.common import AttributeDict
from aiida.plugins import WorkflowFactory

from aiida.engine import run_get_node, submit

ConvergenceCohesiveEnergy = WorkflowFactory(
    'sssp_workflow.convergence.cohesive_energy')

if __name__ == '__main__':
    from aiida import orm
    from aiida.orm import load_code, load_node

    code = load_code('qe-6.6-pw@daint-mc')

    # # Silicon structure and pseudopotential
    structure = load_node('0c5793b9-e52c-4708-b5a1-f53b5c4814bf')
    upf = load_node('9d9d57fc-49e3-4e0c-8c37-4682ccc0fb51')

    PARA_ECUTWFC_LIST = orm.List(list=[20, 25, 200])

    inputs = AttributeDict({
        'code': code,
        'pseudo': upf,
        'structure': structure,
        'parameters': {
            'dual': orm.Int(8),
            'ecutwfc_list': PARA_ECUTWFC_LIST,
        },
    })
    node = submit(ConvergenceCohesiveEnergy, **inputs)
    print(node)
