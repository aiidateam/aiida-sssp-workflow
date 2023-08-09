# -*- coding: utf-8 -*-
"""utils.py"""

import collections.abc
import importlib
import json

import yaml
from aiida import orm
from aiida.engine import calcfunction
from aiida.plugins import DataFactory

UpfData = DataFactory("pseudo.upf")


def get_protocol(category, name=None):
    """Load and read protocol from faml file to a verbose dict
    if name not set, return whole protocol."""
    import_path = importlib.resources.path(
        "aiida_sssp_workflow.protocol", f"{category}.yml"
    )
    with import_path as pp_path, open(pp_path, "rb") as handle:
        protocol_dict = yaml.safe_load(
            handle
        )  # pylint: disable=attribute-defined-outside-init

    if name:
        return protocol_dict[name]
    else:
        return protocol_dict


LANTHANIDE_ELEMENTS = [
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
]

ACTINIDE_ELEMENTS = [
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
]

MAGNETIC_ELEMENTS = ["Mn", "O", "Cr", "Fe", "Co", "Ni"]

NONMETAL_ELEMENTS = [
    "H",
    "He",
    "B",
    "N",
    "O",
    "F",
    "Ne",
    "Si",
    "P",
    "Cl",
    "Ar",
    "Se",
    "Br",
    "Kr",
    "Te",
    "I",
    "Xe",
    "Rn",
]

# These elements don't have typical configurations from Cottiner's paper
NO_GS_CONF_ELEMENTS = ["At", "Fr", "Ra"]

HIGH_DUAL_ELEMENTS = ["O", "Fe", "Mn", "Hf", "Co", "Ni", "Cr"]

OXIDE_CONFIGURATIONS = ["XO", "XO2", "XO3", "X2O", "X2O3", "X2O5"]
UNARIE_CONFIGURATIONS = ["BCC", "FCC", "SC", "Diamond"]


def update_dict(d, u):
    """update dict by7 another dict, update include all hierarchy
    return the update dict instead of change the input dict"""
    # pylint: disable=invalid-name
    import copy

    ret = copy.deepcopy(d)
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            ret[k] = update_dict(ret.get(k, {}), v)
        else:
            ret[k] = v
    return ret


def get_default_configuration(element: str, prop: str) -> str:
    """get default configuration from mapping.json"""
    # use the one from mapping.json as default
    # after some back-forward, most elements only use typical structure
    # for bands and for convergence. But with the primitived structure.
    # But for future maintainance, I keep the mapping.json for configuration
    # mapping. Only At, Fr, Ra and actinides using unaries FCC since it is not in typical.
    import_path = importlib.resources.path(
        "aiida_sssp_workflow.statics.structures", f"mapping.json"
    )

    with import_path as path, open(path, "r") as handle:
        mapping = json.load(handle)

    configuration = mapping[element][prop]

    return configuration


def get_standard_structure(
    element: str, prop: str, configuration=None
) -> orm.StructureData:
    """
    get cif abspath from element for bands measure and convergence
    property can be `delta`, `bands` or `convergence`

    The principles are for bands measure, using the configurations from Cottiner's paper since they are the groud state structures exist in real wolrd.
    And for lanthanides using the Nitrides from Wenzowech paper.
    For elements that don't have configuration from Cottiner's paper, using the uniaries/diamond configurations for convergence verification.

    The structure is convert to primitive cell except for those magnetic elements
    configurations (typical ones of Cottiner's paper) for bands measure, since we need to
    set the starting magnetizations for sites.

    If prop is delta must provide specific configuration name.

    Args:
        element (str): element
        configuration (str): BCC, FCC, SC, Diamond, XO, XO2, XO3, X2O, X2O3, X2O5, RE

    Returns:
        orm.StructureData: return a orm.StructureData
    """
    from pathlib import Path

    from aiida.tools.data.array.kpoints import get_kpoints_path
    from ase import io

    # If for delta measure workflow
    base_structure_module = "aiida_sssp_workflow.statics.structures"

    if configuration is None:
        if prop == "delta":
            raise ValueError("Must provide configuration name for delta measure")
        configuration = get_default_configuration(element, prop)

    # uppercase configuration
    if configuration != "Diamond":
        configuration = configuration.upper()

    if configuration == "RE":
        assert element in LANTHANIDE_ELEMENTS

        # use RE-nitrides of Wenzovich paper
        # from typical (gs) cif folder
        res_path = importlib.resources.path(
            f"{base_structure_module}.gs", f"{element}N.cif"
        )

    elif configuration == "GS":
        res_path = importlib.resources.path(
            f"{base_structure_module}.gs", f"{element}.cif"
        )

    # For elements that are verified in ACWF paper, use the XSF files.
    # https://github.com/aiidateam/acwf-verification-scripts/tree/main/0-preliminary-do-not-run
    elif configuration in OXIDE_CONFIGURATIONS:
        res_path = importlib.resources.path(
            f"{base_structure_module}.oxides", f"{element}-{configuration}.xsf"
        )

    elif configuration in UNARIE_CONFIGURATIONS:
        res_path = importlib.resources.path(
            f"{base_structure_module}.unaries", f"{element}-{configuration}.xsf"
        )

    else:
        raise ValueError(f"Unknown configuration {configuration}")

    with res_path as path:
        if Path(path).suffix == ".cif":
            # For magnetic elements, use the conventional cell
            primitive_cell = True

            if element in MAGNETIC_ELEMENTS:
                primitive_cell = False

            if prop == "delta":
                primitive_cell = False

            structure = orm.CifData.get_or_create(str(path), use_first=True)[
                0
            ].get_structure(primitive_cell=primitive_cell)
        elif Path(path).suffix == ".xsf":
            ase_structure = io.read(str(path))
            structure = orm.StructureData(ase=ase_structure)
            # No functionality for primitive cell in ase

            # To make the structure consistent with the structure from acwf
            res = get_kpoints_path(structure, method="seekpath")
            structure = res["primitive_structure"]
        else:
            raise ValueError(f"Unknown file type {Path(path).suffix}")

    return structure


def parse_upf(upf_content: str) -> dict:
    """

    :param upf_content:
    :param check: if check the integrity of the pp file
    :return:
    """
    from upf_to_json import upf_to_json

    upf_dict = upf_to_json(upf_content, None)

    return upf_dict["pseudo_potential"]


def helper_parse_upf(upf: UpfData) -> dict:
    """parser upf"""
    header = parse_upf(upf.get_content())["header"]

    return header


def get_default_options(max_num_machines=1, max_wallclock_seconds=1800, with_mpi=False):
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


def to_valid_key(name):
    """
    convert name into a valid key name which contain only alphanumeric and underscores
    """
    import re

    valid_name = re.sub(r"[^\w\s]", "_", name)

    return valid_name


@calcfunction
def convergence_analysis(xy: orm.List, criteria: orm.Dict):
    """
    xy is a list of xy tuple [(x1, y1), (x2, y2), ...] and
    criteria is a dict of {'mode': 'a', 'bounds': (0.0, 0.2)}
    """
    # sort xy
    sorted_xy = sorted(xy.get_list(), key=lambda k: k[0], reverse=True)
    criteria = criteria.get_dict()
    mode = criteria["mode"]

    cutoff, value = sorted_xy[0]
    if mode == 0:
        bounds = criteria["bounds"]
        eps = criteria["eps"]
        # from max cutoff, after some x all y is out of bound
        for x, y in sorted_xy:
            if bounds[0] - eps < y < bounds[1] + eps:
                cutoff, value = x, y
            else:
                break

    return {
        "cutoff": orm.Float(cutoff),
        "value": orm.Float(value),
    }


def reset_pseudos_for_magnetic(pseudo, structure):
    """
    override pseudos setting
    required for O, Mn, Cr typical configuration and
    diamond configuration of all magnetic elements where
    the kind names varies for sites
    """
    pseudos = {}
    for kind_name in structure.get_kind_names():
        pseudos[kind_name] = pseudo

    return pseudos


def get_magnetic_inputs(structure: orm.StructureData):
    """
    To set initial magnet to the magnetic system, need to set magnetic order to
    every magnetic element site, with certain pw starting_mainetization parameters.

    ! Only for typical configurations of magnetic elements.
    """
    MAG_INIT_Mn = {
        "Mn1": 0.5,
        "Mn2": -0.3,
        "Mn3": 0.5,
        "Mn4": -0.3,
    }  # pylint: disable=invalid-name
    MAG_INIT_O = {
        "O1": 0.5,
        "O2": 0.5,
        "O3": -0.5,
        "O4": -0.5,
    }  # pylint: disable=invalid-name
    MAG_INIT_Cr = {"Cr1": 0.5, "Cr2": -0.5}  # pylint: disable=invalid-name

    mag_structure = orm.StructureData(cell=structure.cell, pbc=structure.pbc)
    kind_name = structure.get_kind_names()[0]

    # ferromagnetic
    if kind_name in ["Fe", "Co", "Ni"]:
        for i, site in enumerate(structure.sites):
            mag_structure.append_atom(position=site.position, symbols=kind_name)

        parameters = {
            "SYSTEM": {
                "nspin": 2,
                "starting_magnetization": {kind_name: 0.2},
            },
        }

    #
    if kind_name in ["Mn", "O", "Cr"]:
        for i, site in enumerate(structure.sites):
            mag_structure.append_atom(
                position=site.position, symbols=kind_name, name=f"{kind_name}{i+1}"
            )

        if kind_name == "Mn":
            parameters = {
                "SYSTEM": {
                    "nspin": 2,
                    "starting_magnetization": MAG_INIT_Mn,
                },
            }

        if kind_name == "O":
            parameters = {
                "SYSTEM": {
                    "nspin": 2,
                    "starting_magnetization": MAG_INIT_O,
                },
            }

        if kind_name == "Cr":
            parameters = {
                "SYSTEM": {
                    "nspin": 2,
                    "starting_magnetization": MAG_INIT_Cr,
                },
            }

    return mag_structure, parameters
