import pytest
from aiida import orm
from aiida.engine import launch
from aiida.plugins import WorkflowFactory

VerificationWorkChain = WorkflowFactory("sssp_workflow.verification")


@pytest.fixture(scope="function")
def generate_test_inputs(pw_code, ph_code, psp_Si_SG15):
    def _generate_inputs(properties_list):
        inputs = {
            "accuracy": {
                "protocol": orm.Str("test"),
                "cutoff_control": orm.Str("test"),
            },
            "convergence": {
                "protocol": orm.Str("test"),
                "cutoff_control": orm.Str("test"),
                "criteria": orm.Str("efficiency"),
            },
            "pw_code": pw_code,
            "ph_code": ph_code,
            "pseudo": psp_Si_SG15,
            "properties_list": orm.List(list=properties_list),
            "label": orm.Str("sg15/Si_ONCV_PBE-1.2.upf"),
            "options": orm.Dict(
                dict={
                    "resources": {
                        "num_machines": 1,
                        "num_mpiprocs_per_machine": 2,
                    },
                    "max_wallclock_seconds": 1800 * 3,
                    "withmpi": False,
                }
            ),
            "parallelization": orm.Dict(dict={"npool": 1}),
            "clean_workchain": orm.Bool(False),
        }

        return inputs

    return _generate_inputs


@pytest.mark.usefixtures("aiida_profile_clean")
def test_verification_accuracy_workflow(generate_test_inputs, data_regression):
    """Test nightly."""

    from aiida import orm

    properties_list = [
        "accuracy.delta",
    ]

    inputs = generate_test_inputs(properties_list)
    _, node = launch.run_get_node(VerificationWorkChain, **inputs)

    data_regression.check(node.outputs.accuracy.delta.output_parameters.get_dict())

    qb = orm.QueryBuilder()
    qb.append(
        cls=orm.CalcJobNode,
        filters={
            "attributes.process_state": "finished",
            "attributes.exit_status": 305,
        },
    )
    fnode = qb.all()[0][0]

    fnode.outputs.remote_folder.getfile("_scheduler-stderr.txt", "/tmp/err.txt")
    with open("/tmp/err.txt", "r") as fh:
        print(fh.read())


@pytest.mark.usefixtures("aiida_profile_clean")
def test_verification_accuracy_bands_workflow(generate_test_inputs, data_regression):
    """Test nightly."""

    properties_list = [
        "accuracy.bands",
    ]

    inputs = generate_test_inputs(properties_list)
    _, node = launch.run_get_node(VerificationWorkChain, **inputs)

    assert "bands" in node.outputs.accuracy


@pytest.mark.usefixtures("aiida_profile_clean")
def test_verification_convergence_workflow(generate_test_inputs, data_regression):
    """Test nightly."""
    properties_list = [
        "convergence.cohesive_energy",
        "convergence.phonon_frequencies",
        "convergence.pressure",
        "convergence.delta",
        "convergence.bands",
    ]

    inputs = generate_test_inputs(properties_list)

    _, node = launch.run_get_node(VerificationWorkChain, **inputs)

    data_regression.check(node.outputs.pseudo_info.get_dict())

    assert "cohesive_energy" in node.outputs.convergence
    assert "phonon_frequencies" in node.outputs.convergence
    assert "pressure" in node.outputs.convergence
    assert "delta" in node.outputs.convergence
    assert "bands" in node.outputs.convergence
