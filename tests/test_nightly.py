import pytest
from aiida import orm
from aiida.engine import launch

@pytest.mark.requires_rmq
@pytest.mark.usefixtures('aiida_profile_clean', 'chdir_tmp_path')
def test_verification_workflow(aiida_localhost):
    """Test nightly."""
    aiida_localhost.set_use_double_quotes(True)
    engine_command = """singularity exec --bind $PWD:$PWD {image_name}"""
    containerized_code = orm.ContainerizedCode(
        default_calc_job_plugin='core.arithmetic.add',
        filepath_executable='/bin/bash',
        engine_command=engine_command,
        image_name='ubuntu',
        computer=aiida_localhost,
    ).store()
    builder = containerized_code.get_builder()
    builder.x = orm.Int(4)
    builder.y = orm.Int(6)
    builder.metadata.options.resources = {'num_machines': 1, 'num_mpiprocs_per_machine': 1}

    results, node = launch.run_get_node(builder)

    print(results, node)