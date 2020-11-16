#!/usr/bin/env python
from aiida.common import AttributeDict
from aiida.plugins import WorkflowFactory

from aiida.engine import run_get_node, submit

CohesiveEnergyWorkChain = WorkflowFactory('sssp_workflow.cohesive_energy')

if __name__ == '__main__':
    from aiida import orm
    from aiida.orm import load_code, load_node

    code = load_code('qe-6.6-pw@daint-mc')

    # gold structure and pseudopotential
    # structure = load_node('9c2fc420-f76f-484f-b7d9-4df55eb7fee8')
    # upf = load_node('197acb08-ff93-4e65-8de9-2242a96197b2')


    # # Silicon structure and pseudopotential
    structure = load_node('0c5793b9-e52c-4708-b5a1-f53b5c4814bf')
    upf = load_node('9d9d57fc-49e3-4e0c-8c37-4682ccc0fb51')

    for degauss in [0.002, 0.004, 0.006, 0.008, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02]:
        # degauss = 0.02
        PW_PARAS = orm.Dict(
            dict={
                'SYSTEM': {
                    'ecutrho': 1600,
                    'ecutwfc': 200,
                    'degauss': degauss,
                },
            })

        # minimal inputs, maximum wc
        inputs = AttributeDict({
            'code': code,
            'pseudo': upf,
            'structure': structure,
            'parameters': {
                'pw_atom': PW_PARAS,
                'vacuum_length': orm.Float(12.0)
            },
        })
        node = submit(CohesiveEnergyWorkChain, **inputs)
        node.description = f'[Si] degauss={degauss}, vacuum=12.'
        print(node.pk)
