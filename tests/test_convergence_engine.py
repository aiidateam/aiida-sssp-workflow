from aiida import orm
from aiida.plugins import WorkflowFactory

from .sample_processes import echo_workfunction
from aiida_sssp_workflow.workflows.convergence.engine import TwoFactorConvergence

from aiida.engine import run

OptimizationWorkChain = WorkflowFactory('optimize.optimize')


def test_condition_1():
    """
    Condition 1:
    conv_thr satisfied first and convergence depend on tol < 1e-2
    result: 0.102
    """
    inputs = {
        'engine':
        TwoFactorConvergence,
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

    res = run(OptimizationWorkChain, **inputs)
    assert res['optimal_process_output'] == 0.102


def test_condition_2():
    """
    Condition 2:
    tol satisfied first and convergence depend on conv_thr < 1e-1
    result: 0.0993
    """
    inputs = {
        'engine':
        TwoFactorConvergence,
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

    res = run(OptimizationWorkChain, **inputs)
    assert res['optimal_process_output'] == 0.0993
