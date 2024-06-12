from pydantic import BaseModel, field_validator

from aiida_sssp_workflow.utils.structure import VALID_CONFIGURATIONS


class SingleEOSEntry(BaseModel):
    uuid: str
    exit_status: int


class TransferabilityReport(BaseModel):
    eos_dict: dict[str, SingleEOSEntry]

    @classmethod
    def construct(cls, eos_dict: dict[str, dict]):
        """Construct the TransferabilityReport from dict data."""

        return cls(eos_dict={k: SingleEOSEntry(**v) for (k, v) in eos_dict.items()})

    @field_validator("eos_dict")
    def validate_eos_dict(cls, d):
        if not all(k in VALID_CONFIGURATIONS for k in d.keys()):
            raise ValueError(f"configuration should be one of {VALID_CONFIGURATIONS}")
        return d
