import json
from pathlib import Path
from importlib import resources
from ase import Atoms, io

from aiida import orm
from aiida.engine import calcfunction
from aiida.tools.data.array.kpoints import get_kpoints_path

from .element import MAGNETIC_ELEMENTS

OXIDE_CONFIGURATIONS = ["XO", "XO2", "XO3", "X2O", "X2O3", "X2O5"]
UNARIE_CONFIGURATIONS = ["BCC", "FCC", "SC", "DC"]
ACWF_CONFIGURATIONS = OXIDE_CONFIGURATIONS + UNARIE_CONFIGURATIONS


@calcfunction
def get_default_configuration(element: orm.Str, property: orm.Str) -> orm.Str:
    """get default configuration from mapping.json"""
    # use the one from mapping.json as default
    # after some back-forward, most elements only use typical structure
    # for bands calculation.
    # For maintainance convinience, the configuration mapping is stored at mapping.json.

    return orm.Str(_get_default_configuration(element.value, property.value))


def _get_default_configuration(element: str, property: str) -> str:
    import_path = resources.path(
        "aiida_sssp_workflow.statics.structures", "mapping.json"
    )

    with import_path as path, open(path, "r") as handle:
        mapping = json.load(handle)

    configuration = mapping[element][property]

    return configuration


@calcfunction
def get_standard_structure(
    element: orm.Str, configuration=orm.Str
) -> orm.StructureData:
    try:
        ase_structure = _get_standard_structure(element.value, configuration.value)
    except FileNotFoundError:
        raise

    structure = orm.StructureData(ase=ase_structure)

    # No functionality for primitive cell in ase
    # To make the structure consistent with the structure of nat.rev.phys.2024 paper
    # For magnetic elements, use the cell of the file, do not convert to primitive cell.
    if element.value not in MAGNETIC_ELEMENTS:
        res = get_kpoints_path(structure, method="seekpath")
        structure = res["primitive_structure"]

    return structure


def _get_standard_structure(element: str, configuration=str) -> Atoms:
    """
    Create an ASE structure from property and configuration and element.

    For bands measure, first try to use the configurations from sci2016 paper.
    Because those are the groud state structures exist in real wolrd.

    For lanthanides using the nitride structure from Wenzicowitch paper.

    For elements that not tested in sci2016, using DC (Diamond Cubit) configuration from nat.rev.phys.2024

    The structure is convert to primitive cell.
    (TBC) For magnetic elements, use the conventional cell???

    If property is eos must provide specific configuration name.

    Args:
        element (str): element
        configuration (str): BCC, FCC, SC, DC, GS, XO, XO2, XO3, X2O, X2O3, X2O5, RE

    Returns:
        orm.StructureData: return a orm.StructureData
    """

    # If for delta measure workflow
    base_structure_module = "aiida_sssp_workflow.statics.structures"

    # uppercase configuration
    if configuration == "LAN":
        # use LAN-nitrides of Wenzovich paper
        # from typical (gs) cif folder
        p_ctx = resources.path(f"{base_structure_module}.gs", f"{element}N.cif")
    elif configuration == "GS":
        p_ctx = resources.path(f"{base_structure_module}.gs", f"{element}.cif")
    # For elements that are verified in nat.rev.phys.2024 paper, use the XSF files.
    # https://github.com/aiidateam/acwf-verification-scripts/tree/main/0-preliminary-do-not-run
    elif configuration in OXIDE_CONFIGURATIONS:
        p_ctx = resources.path(
            f"{base_structure_module}.oxides", f"{element}-{configuration}.xsf"
        )
    elif configuration in UNARIE_CONFIGURATIONS:
        # The xsf file named as E-Diamond.xsf
        if configuration == "DC":
            configuration = "Diamond"
        p_ctx = resources.path(
            f"{base_structure_module}.unaries", f"{element}-{configuration}.xsf"
        )
    else:
        raise ValueError(f"Unknown configuration {configuration}")

    with p_ctx as path_h:
        filepath = Path(path_h)

        if filepath.suffix not in [".cif", ".xsf"]:
            raise ValueError(f"Unknown file type {filepath.suffix}")
        else:
            ase_structure = io.read(filepath)

    return ase_structure
