from aiida_sssp_workflow.workflows.convergence.report import ConvergenceReport


def analyze_convergence(report: ConvergenceReport, extractor) -> dict:
    # Extract into list
    # If the calculation is not success, do not include it to results
    # The uuid is used to get the extracted result
    ps = []
    for p in report.convergence_list:
        if p.exit_status != 0:
            continue

        _p = p.model_dump()
        _uuid = _p.pop("uuid")

        _p["value"] = extractor(_uuid)
        ps.append(_p)

    # The convergence list are in ascendance order, reverser to start from the lagest
    # cutoff, until the convergence criteria matched for all points.
    # for p in reversed(ps):
    #    if p['value']
