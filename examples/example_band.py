#!/usr/bin/env python
from aiida import orm
from aiida.plugins import WorkflowFactory

SSSPBandsWorkChain = WorkflowFactory('sssp_workflow.bands')

if __name__ == '__main__':
    from aiida.engine import submit
    from aiida.orm import load_node, load_code

    # # Si
    structure = load_node('0c5793b9-e52c-4708-b5a1-f53b5c4814bf')
    # upf = load_node('9d9d57fc-49e3-4e0c-8c37-4682ccc0fb51') # 8 electrons
    upf = load_node('98f04e42-6da8-4960-acfa-0161e0e339a5') # 12 electrons

    # gold structure and pseudopotential
    # structure = load_node('9c2fc420-f76f-484f-b7d9-4df55eb7fee8')
    # upf = load_node('197acb08-ff93-4e65-8de9-2242a96197b2')

    ecutrho = 240
    ecutwfc = 30
    PW_PARAS = orm.Dict(dict={
        'SYSTEM': {
            'ecutrho': ecutrho,
            'ecutwfc': ecutwfc,
        },
    })

    inputs = {
        'code': load_code('qe-6.6-pw@daint-mc'),
        'structure': structure,
        'pseudo': upf,
        'parameters': {
            'scf_kpoints_distance': orm.Float(0.1),
            'bands_kpoints_distance': orm.Float(0.2),
            'pw': PW_PARAS,
        },
    }
    node = submit(SSSPBandsWorkChain, **inputs)
    node.description = '[Si]'
    print(node)