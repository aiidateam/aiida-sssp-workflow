import pytest

from aiida.engine import ProcessBuilder, run_get_node
from aiida.plugins import WorkflowFactory


@pytest.mark.parametrize(
    'pseudo_', [
        'Al.paw',
        'O.nc',
        'O.paw',
    ]
)
def test_default_builder(pseudo_, code_generator, pseudo_path, data_regression):
    """Check the builder is created from inputs"""
    _WorkChain = WorkflowFactory('sssp_workflow.verification')
    
    builder: ProcessBuilder = _WorkChain.get_builder(
        pw_code=code_generator('pw'),
        ph_code=code_generator('ph'),
        pseudo=pseudo_path(pseudo_),
        protocol='test',
        curate_type='sssp',
        dry_run=True,
    )

    result, _ = run_get_node(builder)

    data_regression.check(result['builders'])

# TODO: test using nc Oxygen when curate_type is 'nc'

@pytest.mark.slow
def test_default_verification(code_generator, pseudo_path, data_regression):
    """Check the builder is created from inputs"""
    _WorkChain = WorkflowFactory('sssp_workflow.verification')
    
    builder: ProcessBuilder = _WorkChain.get_builder(
        pw_code=code_generator('pw'),
        ph_code=code_generator('ph'),
        pseudo=pseudo_path('Al.paw'),
        protocol='test',
        curate_type='sssp',
        dry_run=False,
    )

    result, node = run_get_node(builder)





