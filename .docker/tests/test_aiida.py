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


def test_run_real_sssp_computation(aiida_exec, container_user):
    pass
