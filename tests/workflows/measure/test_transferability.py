import pytest

from aiida import orm
from aiida.plugins import DataFactory, WorkflowFactory
from aiida.engine import ProcessBuilder, run_get_node

from aiida_sssp_workflow.workflows.measure.report import TransferabilityReport

UpfData = DataFactory("pseudo.upf")


@pytest.mark.slow
def test_run_default_check_inner_eos_inputs(
    pseudo_path, code_generator, serialize_inputs, data_regression
):
    """Test running the caching convergence workflow.
    Used to test basic things of _base convergence workchain such as the
    output ports are correct and the report is correct in the format.
    """
    _WorkChain = WorkflowFactory("sssp_workflow.measure.transferability")

    builder: ProcessBuilder = _WorkChain.get_builder(
        pseudo=pseudo_path("Al"),
        protocol="test",
        configurations=["SC", "XO"],
        wavefunction_cutoff=25,
        charge_density_cutoff=100,
        oxygen_pseudo=pseudo_path("O_nc"),
        oxygen_ecutwfc=30,
        oxygen_ecutrho=120,
        code=code_generator("pw"),
        clean_workdir=True,
    )

    # run the workchain
    result, node = run_get_node(builder)

    assert node.is_finished_ok
    assert node.label == "Al.paw.pbe.z_3.ld1.psl.v0.1.upf"
    assert "SC" in node.description and "XO" in node.description
    assert "(25, 100)" in node.description
    assert "test" in node.description

    # Check the first EOS (SC) use (25, 100) cutoffs

    outgoing: orm.LinkManager = node.base.links.get_outgoing()
    pw_parameters_SC = outgoing.get_node_by_label("SC").inputs.eos.pw.parameters
    assert isinstance(pw_parameters_SC["SYSTEM"]["ecutwfc"], int)
    assert pw_parameters_SC["SYSTEM"]["ecutwfc"] == 25
    assert pw_parameters_SC["SYSTEM"]["ecutrho"] == 100

    # Check the first EOS (XO) use (30, 120) cutoffs from Oxygen
    pw_parameters_XO = outgoing.get_node_by_label("XO").inputs.eos.pw.parameters
    assert isinstance(pw_parameters_XO["SYSTEM"]["ecutwfc"], int)
    assert pw_parameters_XO["SYSTEM"]["ecutwfc"] == 30
    assert pw_parameters_XO["SYSTEM"]["ecutrho"] == 120

    assert "SC" in result
    assert "XO" in result
    assert "report" in result

    validated_report = TransferabilityReport.construct(**result["report"])

    assert {"SC", "XO"} == set(validated_report.eos_dict.keys())

    assert validated_report.eos_dict["SC"].exit_status == 0
    assert validated_report.eos_dict["XO"].exit_status == 0

    # From the report can get the uuid of the evaluate workchain
    # test that the pw inputs are from the protocol (convergence/base)
    xo_evaluate_node = orm.load_node(validated_report.eos_dict["XO"].uuid)

    data_regression.check(serialize_inputs(xo_evaluate_node.inputs))


@pytest.mark.parametrize(
    "curate_type,clean_workdir",
    [
        ("SSSP", True),
        ("NC", False),
    ],
)
def test_builder_default_args_passing(
    curate_type,
    clean_workdir,
    pseudo_path,
    code_generator,
    serialize_builder,
    data_regression,
):
    """Test transferability workflow builder is correctly created with default args"""
    _WorkChain = WorkflowFactory("sssp_workflow.measure.transferability")

    builder: ProcessBuilder = _WorkChain.get_builder(
        pseudo=pseudo_path("Al"),
        protocol="test",
        wavefunction_cutoff=25,
        charge_density_cutoff=100,
        code=code_generator("pw"),
        curate_type=curate_type,
        clean_workdir=clean_workdir,
    )

    data_regression.check(serialize_builder(builder))
