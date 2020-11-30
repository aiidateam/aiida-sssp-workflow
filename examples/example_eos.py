#!/usr/bin/env python
"""
Running EquationOfStateWorkChain example
You can import the necessary nodes of examples-node-archive
(and config pw-6.6 code) to run this example.
"""
from aiida.common import AttributeDict
from aiida import orm
from aiida.engine import run_get_node

from aiida.plugins import WorkflowFactory

EOSWorkChain = WorkflowFactory('sssp_workflow.eos')


def run_eos(code, structure, upf):
    """eos run for silicon"""
    builder = EOSWorkChain.get_builder()
    inputs = AttributeDict({
        'structure': structure,
        'scale_count': orm.Int(7),
        'scf': {
            'pw': {
                'code':
                code,
                'pseudos': {
                    upf.element: upf
                },
                'parameters':
                orm.Dict(
                    dict={
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
                    }),
                'metadata': {
                    'options': {
                        'resources': {
                            'num_machines': 1
                        },
                        'max_wallclock_seconds': 1800,
                        'withmpi': True,
                    },
                },
            },
            'kpoints_distance': orm.Float(0.1),
        }
    })
    builder.update(**inputs)
    res, node = run_get_node(builder)

    return res, node


if __name__ == '__main__':
    from aiida.orm import load_code, load_node

    code = load_code('qe-6.6-pw@daint-mc')

    # Silicon eos
    structure = load_node('0c5793b9-e52c-4708-b5a1-f53b5c4814bf'
                          )  # structure of silicon in ground state
    upf = load_node(
        '9d9d57fc-49e3-4e0c-8c37-4682ccc0fb51')  # upf node of silicon

    res, node = run_eos(code, structure, upf)
    print(res)  # node.outputs
    print(node)

    # Print eos result
    print(node.outputs.output_parameters.get_dict())
