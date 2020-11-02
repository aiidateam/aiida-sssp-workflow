#!/usr/bin/env python
"""
Running delta factor workchain example
"""
from aiida.common import AttributeDict
from aiida import orm

from aiida.plugins import WorkflowFactory
from aiida.engine import run_get_node

DeltaFactorWorkChain = WorkflowFactory('sssp_workflow.delta_factor')

def run_delta_with_maximum_inputs(code, structure, upf):

    inputs = AttributeDict({
        'code': code,
        'pseudo': upf,
        'structure': structure,
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
                    'degauss': 0.02,
                    'ecutrho': 800,
                    'ecutwfc': 200,
                    'occupations': 'smearing',
                    'smearing': 'marzari-vanderbilt',
                },
                'ELECTRONS': {
                    'conv_thr': 1e-10,
                },
            })
        },
    })
    res, node = run_get_node(DeltaFactorWorkChain, **inputs)
    return res, node



if __name__ == '__main__':
    from aiida.orm import load_code, load_node

    code = load_code('qe-6.6-pw@daint-mc')

    # Gold structure and pseudopotential
    upf = load_node('197acb08-ff93-4e65-8de9-2242a96197b2')

    # minimal inputs
    inputs = AttributeDict({
        'code': code,
        'pseudo': upf,
    })
    res, node = run_get_node(DeltaFactorWorkChain, **inputs)
    print(res, node)

    # Silicon structure and pseudopotential
    structure = load_node('0c5793b9-e52c-4708-b5a1-f53b5c4814bf')   # structure of silicon in ground state
    upf = load_node('9d9d57fc-49e3-4e0c-8c37-4682ccc0fb51')     # upf node of silicon

    # maximum inputs
    res, node = run_delta_with_maximum_inputs(code, structure, upf)
    print(res, node)