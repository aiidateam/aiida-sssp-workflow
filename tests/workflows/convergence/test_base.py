import pytest

from aiida import orm
from aiida.plugins import DataFactory, WorkflowFactory
from aiida.engine import ProcessBuilder, run_get_node

from aiida_sssp_workflow.workflows.convergence.report import ConvergenceReport

UpfData = DataFactory("pseudo.upf")


@pytest.mark.slow
@pytest.mark.parametrize(
    "entry_point",
    [
        "sssp_workflow.convergence.caching",
        "sssp_workflow.convergence.eos",
        "sssp_workflow.convergence.cohesive_energy",
        "sssp_workflow.convergence.pressure",
        "sssp_workflow.convergence.bands",
        "sssp_workflow.convergence.phonon_frequencies",
    ],
)
def test_run_default(
    entry_point, pseudo_path, code_generator, serialize_inputs, data_regression
):
    """Test running the caching convergence workflow.
    Used to test basic things of _base convergence workchain such as the
    output ports are correct and the report is correct in the format.
    """
    _ConvergencWorkChain = WorkflowFactory(entry_point)

    if "phonon_frequencies" in entry_point:
        # require passing pw_code and ph_code
        builder: ProcessBuilder = _ConvergencWorkChain.get_builder(
            pseudo=pseudo_path("Al"),
            protocol="test",
            cutoff_list=[(20, 80), (30, 120)],
            configuration="DC",
            pw_code=code_generator("pw"),
            ph_code=code_generator("ph"),
            clean_workdir=True,
        )
    else:
        builder: ProcessBuilder = _ConvergencWorkChain.get_builder(
            pseudo=pseudo_path("Al"),
            protocol="test",
            cutoff_list=[(20, 80), (30, 120)],
            configuration="DC",
            code=code_generator("pw"),
            clean_workdir=True,
        )

    # run the workchain
    result, node = run_get_node(builder)

    assert node.is_finished_ok
    assert node.label == "Al.paw.pbe.z_3.ld1.psl.v0.1.upf"
    assert "DC" in node.description and "test" in node.description

    assert result["success_rate"].value == 1.0

    validated_report = ConvergenceReport.construct(**result["report"])

    assert validated_report.reference == validated_report.convergence_list[-1]

    assert validated_report.reference.wavefunction_cutoff == 30
    assert validated_report.reference.charge_density_cutoff == 120
    assert validated_report.reference.exit_status == 0

    assert validated_report.convergence_list[0].wavefunction_cutoff == 20
    assert validated_report.convergence_list[0].charge_density_cutoff == 80
    assert validated_report.convergence_list[0].exit_status == 0

    # From the report can get the uuid of the evaluate workchain (PwBaseWorkChain in caching convergenc)
    # test that the pw inputs are from the protocol (convergence/base)
    ref_evaluate_node = orm.load_node(validated_report.reference.uuid)

    data_regression.check(serialize_inputs(ref_evaluate_node.inputs))


@pytest.mark.parametrize(
    "entry_point,clean_workdir",
    [
        ("sssp_workflow.convergence.eos", True),
        ("sssp_workflow.convergence.eos", False),
    ],
)
def test_builder_pseudo_as_upfdata(
    entry_point,
    clean_workdir,
    pseudo_path,
    code_generator,
    serialize_builder,
    data_regression,
):
    _ConvergencWorkChain = WorkflowFactory(entry_point)
    pseudo = UpfData.get_or_create(pseudo_path("Al"))

    builder: ProcessBuilder = _ConvergencWorkChain.get_builder(
        pseudo=pseudo,
        protocol="test",
        cutoff_list=[(20, 80), (30, 120)],
        configuration="DC",
        code=code_generator("pw"),
        clean_workdir=clean_workdir,
    )

    data_regression.check(serialize_builder(builder))


# TODO: test not clean workdir
# TODO: test validator of _base convergence workchain working as expected
