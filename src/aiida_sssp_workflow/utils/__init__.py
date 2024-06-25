# ruff: noqa: F403, F405
from .structure import *
from .pseudo import *
from .element import *
from .protocol import *

__all__ = [
    "get_default_configuration",
    "extract_pseudo_info",
    "get_default_dual",
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

def serialize_data(data):
    from aiida.orm import (
        AbstractCode,
        BaseType,
        Data,
        Dict,
        KpointsData,
        List,
        RemoteData,
        SinglefileData,
    )
    from aiida.plugins import DataFactory

    StructureData = DataFactory("core.structure")
    UpfData = DataFactory("pseudo.upf")

    if isinstance(data, dict):
        return {key: serialize_data(value) for key, value in data.items()}

    if isinstance(data, BaseType):
        return data.value

    if isinstance(data, AbstractCode):
        return data.full_label

    if isinstance(data, Dict):
        return data.get_dict()

    if isinstance(data, List):
        return data.get_list()

    if isinstance(data, StructureData):
        return data.get_formula()

    if isinstance(data, UpfData):
        return f"{data.element}<md5={data.md5}>"

    if isinstance(data, RemoteData):
        # For `RemoteData` we compute the hash of the repository. The value returned by `Node._get_hash` is not
        # useful since it includes the hash of the absolute filepath and the computer UUID which vary between tests
        return data.base.repository.hash()

    if isinstance(data, KpointsData):
        try:
            return data.get_kpoints().tolist()
        except AttributeError:
            return data.get_kpoints_mesh()

    if isinstance(data, SinglefileData):
        return data.get_content()

    if isinstance(data, Data):
        return data.base.caching._get_hash()

    return data
