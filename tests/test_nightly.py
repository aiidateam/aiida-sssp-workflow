import pytest
from aiida import orm
from aiida.engine import launch
from aiida.plugins import WorkflowFactory

VerificationWorkChain = WorkflowFactory("sssp_workflow.verification")


@pytest.mark.usefixtures("aiida_profile_clean")
def test_dummy(aiida_localhost):
    """Test nightly."""
    aiida_localhost.set_use_double_quotes(True)
    engine_command = """singularity exec --bind $PWD:$PWD {image_name}"""
    containerized_code = orm.ContainerizedCode(
        default_calc_job_plugin="core.arithmetic.add",
        filepath_executable="/bin/sh",
        engine_command=engine_command,
        image_name="docker://alpine:3",
        computer=aiida_localhost,
    ).store()
    builder = containerized_code.get_builder()
    builder.x = orm.Int(4)
    builder.y = orm.Int(6)
    builder.metadata.options.resources = {
        "num_machines": 1,
        "num_mpiprocs_per_machine": 1,
    }

    results, node = launch.run_get_node(builder)

    print(results, node)


@pytest.mark.usefixtures("aiida_profile_clean")
def test_verification_accuracy_workflow(psp_Si_SG15, pw_code, ph_code, data_regression):
    """Test nightly."""

    properties_list = [
        "accuracy.delta",
        "accuracy.bands",
    ]

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
        "clean_workchain": orm.Bool(True),
    }

    _, node = launch.run_get_node(VerificationWorkChain, **inputs)

    assert "bands" in node.outputs.accuracy

    data_regression.check(node.outputs.accuracy.delta.output_parameters.get_dict())


# @pytest.mark.usefixtures("aiida_profile_clean")
# def test_verification_convergence_workflow(
#     psp_Si_SG15, pw_code, ph_code, data_regression
# ):
#     """Test nightly."""
#     properties_list = [
#         # "convergence.cohesive_energy",
#         "convergence.phonon_frequencies",
#         "convergence.pressure",
#         "convergence.delta",
#         "convergence.bands",
#     ]

#     inputs = {
#         "accuracy": {
#             "protocol": orm.Str("test"),
#             "cutoff_control": orm.Str("test"),
#         },
#         "convergence": {
#             "protocol": orm.Str("test"),
#             "cutoff_control": orm.Str("test"),
#             "criteria": orm.Str("efficiency"),
#         },
#         "pw_code": pw_code,
#         "ph_code": ph_code,
#         "pseudo": psp_Si_SG15,
#         "properties_list": orm.List(list=properties_list),
#         "label": orm.Str("sg15/Si_ONCV_PBE-1.2.upf"),
#         "options": orm.Dict(
#             dict={
#                 "resources": {
#                     "num_machines": 1,
#                     "num_mpiprocs_per_machine": 2,
#                 },
#                 "max_wallclock_seconds": 1800 * 3,
#                 "withmpi": False,
#             }
#         ),
#         "parallelization": orm.Dict(dict={"npool": 1}),
#         "clean_workchain": orm.Bool(True),
#     }

#     _, node = launch.run_get_node(VerificationWorkChain, **inputs)

#     data_regression.check(node.outputs.pseudo_info.get_dict())
#     # data_regression.check(node.outputs.accuracy.delta.output_parameters.get_dict())

#     assert "cohesive_energy" in node.outputs.convergence
#     assert "phonon_frequencies" in node.outputs.convergence
#     assert "pressure" in node.outputs.convergence
#     assert "delta" in node.outputs.convergence
#     assert "bands" in node.outputs.convergence
