"""
Run a convergence study for kinetic energy cutoff on Si using QuantumEspresso pw.x
"""
from aiida import orm
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from aiida.plugins import WorkflowFactory
from aiida_optimize.engines import Convergence

OptimizationWorkChain = WorkflowFactory('optimize.optimize')

ECUT_VALUES = [20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 110, 120]

structure = orm.load_node('0c5793b9-e52c-4708-b5a1-f53b5c4814bf')
upf = orm.load_node('9d9d57fc-49e3-4e0c-8c37-4682ccc0fb51')

parameters_dict = {
    'kpoints_distance': orm.Float(0.15),
    'pw': {
        'code':
        orm.load_code('qe-6.6-pw@daint-mc'),
        'structure':
        structure,
        'pseudos': {
            upf.element: upf
        },
        'parameters':
        orm.Dict(
            dict={
                'CONTROL': {
                    'calculation': 'scf',
                },
                'SYSTEM': {
                    'degauss': 0.02,
                    'smearing': 'mv',
                },
                'ELECTRONS': {
                    'conv_thr': 1e-8,
                },
            }),
        'settings':
        orm.Dict(dict={'CMDLINE': ['-ndiag', '1']}),
        'metadata': {
            'options': {
                'resources': {
                    'num_machines': 1
                },
                'max_wallclock_seconds': 1800,
                'withmpi': True,
            }
        }
    }
}

if __name__ == '__main__':
    from aiida.engine import submit
    inputs = {
        'engine':
        Convergence,
        'engine_kwargs':
        orm.Dict(
            dict={
                'input_values': ECUT_VALUES,
                'tol': 0.001,
                'input_key': 'pw.parameters:SYSTEM.ecutwfc',
                'result_key': 'output_parameters:energy',
                'convergence_window': 3,
            }),
        'evaluate_process':
        PwBaseWorkChain,
        'evaluate':
        parameters_dict,
    }

    node = submit(OptimizationWorkChain, **inputs)
    print(node)
