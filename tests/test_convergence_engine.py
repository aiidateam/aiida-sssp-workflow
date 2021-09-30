# -*- coding: utf-8 -*-
"""tests of customize optimize engine"""
from aiida import orm
from aiida.plugins import WorkflowFactory
from aiida.engine import run
from aiida_sssp_workflow.workflows.convergence.engine import TwoInputsTwoFactorsConvergence

from .sample_processes import echo_calcfunction

OptimizationWorkChain = WorkflowFactory('optimize.optimize')

_INPUT_VALUES = [(1, 2), (1, 1), (1.0, 0.3), (1.0, 0.2), (0.002, 0.100),
                 (0.0793, 0.02), (0.0794, 0.02), (0.0792, 0.02)]


def test_condition_1():
    """
    Condition 1:
    conv_thr satisfied first and convergence depend on tol < 1e-2
    result: 0.102
    """
    inputs = {
        'engine':
        TwoInputsTwoFactorsConvergence,
        'engine_kwargs':
        orm.Dict(
            dict={
                'input_values': _INPUT_VALUES,
                'tol': 1e-2,
                'conv_thr': 2,
                'input_key': 'x',
                'extra_input_key': 'y',
                'result_key': 'result',
                'convergence_window': 2
            }),
        'evaluate_process':
        echo_calcfunction
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
        TwoInputsTwoFactorsConvergence,
        'engine_kwargs':
        orm.Dict(
            dict={
                'input_values': _INPUT_VALUES,
                'tol': 1e-2,
                'conv_thr': 1e-1,
                'input_key': 'x',
                'extra_input_key': 'y',
                'result_key': 'result',
                'convergence_window': 2
            }),
        'evaluate_process':
        echo_calcfunction
    }

    res = run(OptimizationWorkChain, **inputs)
    assert res['optimal_process_output'] == 0.0993
