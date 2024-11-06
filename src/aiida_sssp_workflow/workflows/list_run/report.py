from pydantic import BaseModel


class PointRunReportEntry(BaseModel):
    uuid: str
    x: float
    exit_status: int


class ListReport(BaseModel):
    report_list: list[PointRunReportEntry]

    @classmethod
    def build(cls, list_: list[dict]):
        """Construct the list of calcs from dict data."""

        return cls(
            report_list=[
                PointRunReportEntry(**entry) for entry in list_
            ],
        )
