#!/usr/bin/env python
from aiida.common import AttributeDict
from aiida.plugins import WorkflowFactory

from aiida.engine import run_get_node, submit

ConvergencePressureWorkChain = WorkflowFactory(
    'sssp_workflow.convergence.pressure')

if __name__ == '__main__':
    from aiida import orm
    from aiida.orm import load_code, load_node

    code = load_code('qe-6.6-pw@daint-mc')

    # # Silicon structure and pseudopotential
    # upf = load_node('9d9d57fc-49e3-4e0c-8c37-4682ccc0fb51')

    # Fluorine pseude
    # upf = load_node('a1b60d81-d5a3-4ddb-a31a-d5b608c1ba52')

    # gold structure and pseudopotential
    # upf = load_node('197acb08-ff93-4e65-8de9-2242a96197b2') # NC
    # upf = load_node('61830c2b-fa08-4e92-9571-1141ded79612')  # PAW

    # Lanthanum
    upf = load_node('b2880763-579c-4f6d-8803-2c77f4fb10e8')

    PARA_ECUTWFC_LIST = orm.List(list=[20, 25, 200])
    PARA_ECUTRHO_LIST = orm.List(list=[160, 200, 1600])

    inputs = AttributeDict({
        'code': code,
        'pseudo': upf,
        'parameters': {
            'ecutwfc_list': PARA_ECUTWFC_LIST,
            'ecutrho_list': PARA_ECUTRHO_LIST,
        },
    })
    node = submit(ConvergencePressureWorkChain, **inputs)
    print(node)
