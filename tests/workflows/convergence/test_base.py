from aiida.plugins import DataFactory, WorkflowFactory
from aiida.engine import ProcessBuilder, run_get_node

from aiida_sssp_workflow.workflows.convergence.caching import (
    _CachingConvergenceWorkChain,
)
from aiida_sssp_workflow.workflows.convergence._base import ConvergenceReport

UpfData = DataFactory("pseudo.upf")
CachingConvergenceWorkChain: _CachingConvergenceWorkChain = WorkflowFactory(
    "sssp_workflow.convergence.caching"
)


def test_run_caching_default(pseudo_path, code_generator):
    """Test running the caching convergence workflow.
    Used to test basic things of _base convergence workchain such as the
    output ports are correct and the report is correct in the format.
    """

    builder: ProcessBuilder = CachingConvergenceWorkChain.get_builder(
        pseudo=pseudo_path("Al"),
        protocol="test",
        cutoff_list=[(20, 80), (30, 120)],
        configuration="DC",
        code=code_generator("pw"),
        clean_workdir=True,
    )

    # set up the inputs
    builder.metadata.label = "test"
    builder.metadata.description = "test"

    # run the workchain
    result, node = run_get_node(builder)

    assert node.is_finished_ok

    validated_report = ConvergenceReport.construct(**result["convergence_report"])

    assert validated_report.reference == validated_report.convergence_list[-1]

    assert validated_report.reference.wavefunction_cutoff == 30
    assert validated_report.reference.charge_density_cutoff == 120
    assert validated_report.reference.exit_status == 0

    assert validated_report.convergence_list[0].wavefunction_cutoff == 20
    assert validated_report.convergence_list[0].charge_density_cutoff == 80
    assert validated_report.convergence_list[0].exit_status == 0
