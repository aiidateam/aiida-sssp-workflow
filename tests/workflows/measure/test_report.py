import pytest

from aiida_sssp_workflow.workflows.measure.report import TransferabilityReport


def test_construct_report_entry(generate_uuid):
    """Test convergence report model"""
    report1 = {
        "uuid": generate_uuid("1"),
        "exit_status": 0,
    }

    report2 = {
        "uuid": generate_uuid("2"),
        "exit_status": 0,
    }

    expected_report = TransferabilityReport.construct({"XO": report1, "SC": report2})

    # regression test
    got_report = TransferabilityReport.construct(**expected_report.model_dump())

    assert got_report == expected_report


def test_raise_when_configuration_not_valid(generate_uuid):
    report1 = {
        "uuid": generate_uuid("1"),
        "exit_status": 0,
    }

    report2 = {
        "uuid": generate_uuid("2"),
        "exit_status": 0,
    }

    with pytest.raises(ValueError, match="should be one of"):
        TransferabilityReport.construct({"XO": report1, "WRONG": report2})
