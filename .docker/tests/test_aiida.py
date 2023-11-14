# -*- coding: utf-8 -*-
# pylint: disable=missing-docstring


def test_verdi_status(aiida_exec, container_user):
    output = aiida_exec("verdi status", user=container_user).decode().strip()
    assert "Connected to RabbitMQ" in output
    assert "Daemon is running" in output

    # check that we have suppressed the warnings
    assert "Warning" not in output


def test_computer_setup_success(aiida_exec, container_user):
    output = (
        aiida_exec("verdi computer test localhost", user=container_user)
        .decode()
        .strip()
    )

    assert "Success" in output
    assert "Failed" not in output


# def test_run_real_sssp_measure_precision_verification(aiida_exec, container_user):
#    cmd = "aiida-sssp-workflow launch --property measure.precision --pw-code pw-7.1@localhost --ecutwfc 30 --ecutrho 240 --protocol test --configuration BCC --withmpi True --num-mpiprocs 1 --npool 1 --no-daemon -- /opt/examples/Si.paw.z_4.ld1.psl.v1.0.0-high.upf"
#
#    output = aiida_exec(cmd, user=container_user).decode().strip()
#
#    print(output)
