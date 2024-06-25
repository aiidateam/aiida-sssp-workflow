import pytest

from aiida import orm
from aiida.plugins import DataFactory, WorkflowFactory
from aiida.engine import ProcessBuilder, run_get_node

from aiida_sssp_workflow.workflows.transferability.report import BandsReport

UpfData = DataFactory("pseudo.upf")


@pytest.mark.slow
def test_run_default(pseudo_path, code_generator, serialize_inputs, data_regression):
    """Test band structure verification workflow"""
    _WorkChain = WorkflowFactory("sssp_workflow.transferability.bands")

    builder: ProcessBuilder = _WorkChain.get_builder(
        pseudo=pseudo_path(),
        protocol="test",
        configuration="SC",
        cutoffs=(25, 100),
        code=code_generator("pw"),
        clean_workdir=True,
    )

    # run the workchain
    result, node = run_get_node(builder)

    assert node.is_finished_ok
    assert node.label == "Al.paw.pbe.z_3.ld1.psl.v0.1.upf"
    assert "SC" in node.description
    assert "(25, 100)" in node.description
    assert "test" in node.description

    assert "report" in result
    assert "bands" in result
    assert "band_structure" in result

    report = BandsReport.construct(result["report"].get_dict())
    bands_node = orm.load_node(report.bands.uuid)
    band_structure_node = orm.load_node(report.band_structure.uuid)

    pw_parameters_bands = bands_node.called[1].called[0].inputs.parameters
    assert isinstance(pw_parameters_bands["SYSTEM"]["ecutwfc"], int)
    assert pw_parameters_bands["SYSTEM"]["ecutwfc"] == 25
    assert pw_parameters_bands["SYSTEM"]["ecutrho"] == 100

    pw_parameters_bs = band_structure_node.called[2].called[0].inputs.parameters
    assert isinstance(pw_parameters_bs["SYSTEM"]["ecutwfc"], int)
    assert pw_parameters_bs["SYSTEM"]["ecutwfc"] == 25
    assert pw_parameters_bs["SYSTEM"]["ecutrho"] == 100

    data_regression.check(serialize_inputs(band_structure_node.inputs))


@pytest.mark.parametrize("clean_workdir", [True, False])
def test_builder_default_args_passing(
    clean_workdir,
    pseudo_path,
    code_generator,
    serialize_builder,
    data_regression,
):
    """Test transferability workflow builder is correctly created with default args"""
    _WorkChain = WorkflowFactory("sssp_workflow.transferability.bands")

    builder: ProcessBuilder = _WorkChain.get_builder(
        pseudo=pseudo_path(),
        protocol="test",
        cutoffs=(25, 100),
        code=code_generator("pw"),
        clean_workdir=clean_workdir,
    )

    data_regression.check(serialize_builder(builder))
