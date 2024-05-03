import json
from importlib import resources

from aiida.engine import calcfunction
from aiida import orm


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


# def get_standard_structure(
#    element: str, prop: str, configuration=None
# ) -> orm.StructureData:
#    """
#    get cif abspath from element for bands measure and convergence
#    property can be `delta`, `bands` or `convergence`
#
#    The principles are for bands measure, using the configurations from Cottiner's paper since they are the groud state structures exist in real wolrd.
#    And for lanthanides using the Nitrides from Wenzowech paper.
#    For elements that don't have configuration from Cottiner's paper, using the uniaries/diamond configurations for convergence verification.
#
#    The structure is convert to primitive cell except for those magnetic elements
#    configurations (typical ones of Cottiner's paper) for bands measure, since we need to
#    set the starting magnetizations for sites.
#
#    If prop is delta must provide specific configuration name.
#
#    Args:
#        element (str): element
#        configuration (str): BCC, FCC, SC, Diamond, XO, XO2, XO3, X2O, X2O3, X2O5, RE
#
#    Returns:
#        orm.StructureData: return a orm.StructureData
#    """
#    from pathlib import Path
#
#    from aiida.tools.data.array.kpoints import get_kpoints_path
#    from ase import io
#
#    # If for delta measure workflow
#    base_structure_module = "aiida_sssp_workflow.statics.structures"
#
#    if configuration is None:
#        if prop == "delta":
#            raise ValueError("Must provide configuration name for delta measure")
#        configuration = get_default_configuration(element, prop)
#
#    # uppercase configuration
#    if configuration != "Diamond":
#        configuration = configuration.upper()
#
#    if configuration == "RE":
#        assert element in LANTHANIDE_ELEMENTS
#
#        # use RE-nitrides of Wenzovich paper
#        # from typical (gs) cif folder
#        res_path = importlib.resources.path(
#            f"{base_structure_module}.gs", f"{element}N.cif"
#        )
#
#    elif configuration == "GS":
#        res_path = importlib.resources.path(
#            f"{base_structure_module}.gs", f"{element}.cif"
#        )
#
#    # For elements that are verified in ACWF paper, use the XSF files.
#    # https://github.com/aiidateam/acwf-verification-scripts/tree/main/0-preliminary-do-not-run
#    elif configuration in OXIDE_CONFIGURATIONS:
#        res_path = importlib.resources.path(
#            f"{base_structure_module}.oxides", f"{element}-{configuration}.xsf"
#        )
#
#    elif configuration in UNARIE_CONFIGURATIONS:
#        res_path = importlib.resources.path(
#            f"{base_structure_module}.unaries", f"{element}-{configuration}.xsf"
#        )
#
#    else:
#        raise ValueError(f"Unknown configuration {configuration}")
#
#    with res_path as path:
#        if Path(path).suffix == ".cif":
#            # For magnetic elements, use the conventional cell
#            primitive_cell = True
#
#            if element in MAGNETIC_ELEMENTS:
#                primitive_cell = False
#
#            if prop == "delta":
#                primitive_cell = False
#
#            structure = orm.CifData.get_or_create(str(path), use_first=True)[
#                0
#            ].get_structure(primitive_cell=primitive_cell)
#        elif Path(path).suffix == ".xsf":
#            ase_structure = io.read(str(path))
#            structure = orm.StructureData(ase=ase_structure)
#            # No functionality for primitive cell in ase
#
#            # To make the structure consistent with the structure from acwf
#            res = get_kpoints_path(structure, method="seekpath")
#            structure = res["primitive_structure"]
#        else:
#            raise ValueError(f"Unknown file type {Path(path).suffix}")
#
#    return structure
#
