import pytest
from pydantic import ValidationError

from aiida_sssp_workflow.workflows.transferability.report import (
    BandsReport,
    EOSReport,
)


def test_construct_transferibility_report_entry(generate_uuid):
    """Test EOSReport report model"""
    report1 = {
        "uuid": generate_uuid("1"),
        "exit_status": 0,
    }

    report2 = {
        "uuid": generate_uuid("2"),
        "exit_status": 0,
    }

    expected_report = EOSReport.construct({"XO": report1, "SC": report2})

    # regression test
    got_report = EOSReport.construct(**expected_report.model_dump())

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
        EOSReport.construct({"XO": report1, "WRONG": report2})


def test_construct_band_report_entry(generate_uuid):
    """Test band report model"""
    uuid1 = generate_uuid("1")
    uuid2 = generate_uuid("2")

    bands = {
        "uuid": uuid1,
        "exit_status": 0,
    }

    band_structure = {
        "uuid": uuid2,
        "exit_status": 0,
    }

    expected_report = BandsReport.construct(
        {"bands": bands, "band_structure": band_structure}
    )

    # regression test
    got_report = BandsReport.construct(expected_report.model_dump())

    assert got_report == expected_report

    # Check the field can be reached with desired ref
    assert expected_report.bands.uuid == uuid1
    assert expected_report.band_structure.uuid == uuid2


def test_raise_when_key_of_band_report_not_valid(generate_uuid):
    report1 = {
        "uuid": generate_uuid("1"),
        "exit_status": 0,
    }
    report2 = {
        "uuid": generate_uuid("2"),
        "exit_status": 0,
    }

    with pytest.raises(ValidationError, match="1 validation error for BandsReport"):
        BandsReport.construct({"bands": report1, "WRONG": report2})
