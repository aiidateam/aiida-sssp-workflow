#!/usr/bin/env python
from aiida import orm
from aiida.plugins import WorkflowFactory

from aiida_sssp_workflow.calculations.helper_functions import helper_get_primitive_structure

PwBaseWorkflow = WorkflowFactory('quantumespresso.pw.base')
PhBaseWorkflow = WorkflowFactory('quantumespresso.ph.base')

# gold
structure = orm.load_node('9c2fc420-f76f-484f-b7d9-4df55eb7fee8')
upf = orm.load_node('197acb08-ff93-4e65-8de9-2242a96197b2')
primitive_structure = helper_get_primitive_structure(structure)


def pw_builder(ecutwfc, ecutrho):
    # Si
    # structure = orm.load_node('0c5793b9-e52c-4708-b5a1-f53b5c4814bf')
    # upf = orm.load_node('9d9d57fc-49e3-4e0c-8c37-4682ccc0fb51')

    # Fe
    # structure = orm.load_node('9d4cb500-7d38-4017-b9f4-8660e934df3e')
    # upf = orm.load_node('76c53fef-b054-4490-aa89-c30b306909c9')

    builder = PwBaseWorkflow.get_builder()
    builder.pw.code = orm.load_code('qe-6.6-pw@daint-mc')
    builder.pw.structure = primitive_structure
    builder.pw.parameters = orm.Dict(
        dict={
            'CONTROL': {
                'calculation': 'scf',
                'wf_collect': True,
            },
            'SYSTEM': {
                'ecutrho': ecutrho,
                'ecutwfc': ecutwfc,
                "noinv": True,
                "nosym": True,
                'occupations': 'smearing',
                'smearing': 'mv',
                'degauss': 0.02,
            },
            'ELECTRONS': {
                'conv_thr': 1e-10,
            },
        })
    builder.pw.pseudos = {upf.element: upf}
    builder.pw.metadata = {
        'options': {
            'resources': {
                'num_machines': 1
            },
            'max_wallclock_seconds': 1800,
            'withmpi': True,
        },
    }
    builder.pw.settings = orm.Dict(
        dict={'CMDLINE': ['-ndiag', '1', '-nk', '4']})
    kpoints = orm.KpointsData()
    kpoints.set_cell_from_structure(primitive_structure)
    kpoints.set_kpoints_mesh([2, 2, 2])
    builder.kpoints = kpoints
    return builder


def ph_builder(calculation, kpoints):
    # Si
    builder = PhBaseWorkflow.get_builder()
    builder.ph.code = orm.load_code('qe-6.6-ph@daint-mc')
    builder.ph.qpoints = kpoints
    builder.ph.parent_folder = calculation.outputs.remote_folder
    builder.ph.parameters = orm.Dict(
        dict={'INPUTPH': {
            'tr2_ph': 1e-16,
            'epsil': False,
        }})
    builder.ph.settings = orm.Dict(dict={'CMDLINE': ['-nk', '4']})
    builder.ph.metadata = {
        'options': {
            'resources': {
                'num_machines': 1
            },
            'max_wallclock_seconds': 1800,
            'withmpi': True,
        },
    }

    return builder


if __name__ == '__main__':
    from aiida.engine import submit

    ecutwfc = 50
    ecutrho = 400
    # for ecutrho in [720,810,900,990,1080,1170,1260]:
    # ecutrho = ecutwfc * 4.
    builder = pw_builder(ecutwfc, ecutrho)
    node = submit(builder)
    node.description = f'[Si] ecutwfc={ecutwfc}, ecutrho={ecutrho}'
    print(node)

    # structure = orm.load_node('0c5793b9-e52c-4708-b5a1-f53b5c4814bf')
    # kpoints = orm.KpointsData()
    # kpoints.set_cell_from_structure(primitive_structure)
    # kpoints.set_kpoints([[0.5, 0.5, 0.5]])
    # calculation = orm.load_node(14621)
    # builder = ph_builder(calculation, kpoints)
    # node = submit(builder)
    # print(node)
