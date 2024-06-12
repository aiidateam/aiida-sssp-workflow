from aiida_sssp_workflow.workflows.convergence.report import ConvergenceReport


def test_point_run_report_entry(generate_uuid):
    """Test convergence report model"""
    reference = {
        "uuid": generate_uuid("0"),
        "wavefunction_cutoff": 100,
        "charge_density_cutoff": 200,
        "exit_status": 0,
    }

    report1 = {
        "uuid": generate_uuid("1"),
        "wavefunction_cutoff": 80,
        "charge_density_cutoff": 200,
        "exit_status": 0,
    }

    report2 = {
        "uuid": generate_uuid("2"),
        "wavefunction_cutoff": 100,
        "charge_density_cutoff": 200,
        "exit_status": 0,
    }

    expected_report = ConvergenceReport.construct(reference, [report1, report2])

    # regression test
    got_report = ConvergenceReport.construct(**expected_report.model_dump())

    assert got_report == expected_report
