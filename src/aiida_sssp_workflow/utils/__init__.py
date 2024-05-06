# ruff: noqa: F403, F405
from .structure import *
from .pseudo import *
from .element import *
from .protocol import *

__all__ = [
    "get_default_configuration",
    "extract_pseudo_info",
    "parse_std_filename",
    "LANTHANIDE_ELEMENTS",
    "ACTINIDE_ELEMENTS",
    "MAGNETIC_ELEMENTS",
    "NON_METALLIC_ELEMENTS",
    "NO_GS_CONF_ELEMENTS",
    "HIGH_DUAL_ELEMENTS",
    "OXIDE_CONFIGURATIONS",
    "UNARIE_CONFIGURATIONS",
    "ACWF_CONFIGURATIONS",
    "get_protocol",
]


def get_default_mpi_options(
    max_num_machines=1, max_wallclock_seconds=1800, with_mpi=False
):
    """Return an instance of the options dictionary with the minimally required parameters for a `CalcJob`.

    :param max_num_machines: set the number of nodes, default=1
    :param max_wallclock_seconds: set the maximum number of wallclock seconds, default=1800
    :param with_mpi: whether to run the calculation with MPI enabled
    """
    return {
        "resources": {"num_machines": int(max_num_machines)},
        "max_wallclock_seconds": int(max_wallclock_seconds),
        "withmpi": with_mpi,
    }
