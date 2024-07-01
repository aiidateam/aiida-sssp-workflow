from pydantic import BaseModel


class PointRunReportEntry(BaseModel):
    uuid: str
    wavefunction_cutoff: int
    charge_density_cutoff: int
    exit_status: int


class ConvergenceReport(BaseModel):
    reference: PointRunReportEntry
    convergence_list: list[PointRunReportEntry]

    @classmethod
    def construct(cls, reference: dict, convergence_list: list[dict]):
        """Construct the ConvergenceReport from dict data."""

        return cls(
            reference=PointRunReportEntry(**reference),
            convergence_list=[
                PointRunReportEntry(**entry) for entry in convergence_list
            ],
        )
