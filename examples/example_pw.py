#!/usr/bin/env python
from aiida import orm
from aiida.plugins import WorkflowFactory

PwBaseWorkflow = WorkflowFactory('quantumespresso.pw.base')

def pw_builder(ecutwfc, ecutrho):
    # Si
    # structure = orm.load_node('0c5793b9-e52c-4708-b5a1-f53b5c4814bf')
    # upf = orm.load_node('9d9d57fc-49e3-4e0c-8c37-4682ccc0fb51')

    # Fe
    structure = orm.load_node('9d4cb500-7d38-4017-b9f4-8660e934df3e')
    upf = orm.load_node('76c53fef-b054-4490-aa89-c30b306909c9')

    builder = PwBaseWorkflow.get_builder()
    builder.pw.code = orm.load_code('qe-6.6-pw@daint-mc')
    builder.pw.structure = structure
    builder.pw.parameters = orm.Dict(dict={
        'CONTROL': {'calculation': 'scf'},
        'SYSTEM': {
            'ecutrho': ecutrho,
            'ecutwfc': ecutwfc,
            'occupations': 'smearing',
            'smearing': 'mv',
            'degauss':0.02,
        },
        'ELECTRONS': {
            'conv_thr': 1e-10,
        },
    })
    builder.pw.pseudos = {upf.element: upf}
    builder.pw.metadata = {
        'options': {
            'resources': {'num_machines': 1},
            'max_wallclock_seconds': 1800,
            'withmpi': True,
        },
    }
    builder.pw.settings = orm.Dict(dict={'CMDLINE': ['-ndiag', '1']})
    builder.kpoints_distance = orm.Float(0.15)
    return builder

if __name__ == '__main__':
    from aiida.engine import submit

    ecutwfc = 90
    # ecutrho = 1080
    for ecutrho in [720,810,900,990,1080,1170,1260]:
        # ecutrho = ecutwfc * 4.
        builder = pw_builder(ecutwfc, ecutrho)
        node = submit(builder)
        node.description = f'[Fe] ecutwfc={ecutwfc}, ecutrho={ecutrho}'
        print(node)