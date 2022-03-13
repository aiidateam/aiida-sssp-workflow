# -*- coding: utf-8 -*-
"""Test workflow use aiida-testing"""
from aiida import orm
from aiida.engine import run_get_node
from aiida.plugins import WorkflowFactory

DeltaFactorWorkChain = WorkflowFactory("sssp_workflow.delta_measure")


def test_delta_measure_default(mocked_pw67, pp_silicon_sg15):
    """test delta factor workflow the default"""
    inputs = {
        "code": mocked_pw67,
        "pseudo": pp_silicon_sg15,
        "protocol": orm.Str("test"),
    }

    _, node = run_get_node(DeltaFactorWorkChain, **inputs)
    assert node.is_finished_ok
