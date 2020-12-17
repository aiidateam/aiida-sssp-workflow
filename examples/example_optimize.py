"""
Run a convergence study for kinetic energy cutoff on Si using QuantumEspresso pw.x
"""
from aiida import orm
from aiida.plugins import WorkflowFactory

from aiida_sssp_workflow.workflows.workfunctions import echo_workfunction
from aiida_sssp_workflow.workflows.convergence.engine import ConvergenceEngine

OptimizationWorkChain = WorkflowFactory('optimize.optimize')

if __name__ == '__main__':
    from aiida.engine import submit, run

    # Condition 1:
    # conv_thr satisfied first and convergence depend on tol
    # result: 0.102
    inputs = {
        'engine':
        ConvergenceEngine,
        'engine_kwargs':
        orm.Dict(
            dict={
                'input_values':
                [3, 2, 1.3, 1.2, 0.102, 0.0993, 0.0994, 0.0992],
                'tol': 1e-2,
                'conv_thr': 2,
                'input_key': 'x',
                'result_key': 'result',
                'convergence_window': 2
            }),
        'evaluate_process':
        echo_workfunction,
    }

    node = run(OptimizationWorkChain, **inputs)
    print(node)

    # Condition 2:
    # tol satisfied first and convergence depend on conv_thr < 1e-1
    # result: 0.0993
    inputs = {
        'engine':
        ConvergenceEngine,
        'engine_kwargs':
        orm.Dict(
            dict={
                'input_values':
                [3, 2, 1.3, 1.2, 0.102, 0.0993, 0.0994, 0.0992],
                'tol': 1e-2,
                'conv_thr': 1e-1,
                'input_key': 'x',
                'result_key': 'result',
                'convergence_window': 2
            }),
        'evaluate_process':
        echo_workfunction,
    }

    node = run(OptimizationWorkChain, **inputs)
    print(node)
