"""This is a jb script run to generate file `statics/structures/mapping.json`."""

import json
from pathlib import Path

from aiida_sssp_workflow.utils import (
    MAGNETIC_ELEMENTS,
    ACTINIDE_ELEMENTS,
    LANTHANIDE_ELEMENTS,
    NO_GS_CONF_ELEMENTS,
    ALL_ELEMENTS,
    UNSUPPORTED_ELEMENTS,
)

MAPPING_FILE_PATH = Path(__file__).parent / "structures" / "mapping.json"


def run():
    d = {}
    d["_commont"] = (
        "The mapping from element to configuration for convergence and bands verification."
    )

    # We use DC (Diamond Cubic) for the convergence test for all elements.
    # Because it usually give the lagrest cutoff energy, which is the most strict test.
    for e in ALL_ELEMENTS:
        if e in UNSUPPORTED_ELEMENTS:
            band_configuration = "N/A"
            convergence_configuration = "N/A"
        elif e in NO_GS_CONF_ELEMENTS:
            # we don't have At in typical
            band_configuration = "DC"
            convergence_configuration = "DC"
        elif e in ACTINIDE_ELEMENTS:
            # we don't have Actinides in typical
            # The paper "Dissertation, Philipps-Universität Marburg, 2022. DOI: 10.13140/RG.2.2.28627.25121"
            # use FCC for all actinides.
            band_configuration = "DC"
            convergence_configuration = "DC"
        elif e in LANTHANIDE_ELEMENTS:
            band_configuration = "LAN"
            convergence_configuration = "DC"
        elif e in MAGNETIC_ELEMENTS:
            band_configuration = "GS"
            convergence_configuration = "DC"
        else:
            band_configuration = "GS"
            convergence_configuration = "DC"

        d[e] = {
            "band": band_configuration,
            "convergence": convergence_configuration,
        }

    with open(MAPPING_FILE_PATH, "w") as f:
        json.dump(d, f, indent=4)


if __name__ == "__main__":
    run()
