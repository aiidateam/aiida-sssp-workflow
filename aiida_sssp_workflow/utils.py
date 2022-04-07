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


def get_protocol(category, name):
    """Load and read protocol from faml file to a verbose dict"""
    import_path = importlib.resources.path(
        "aiida_sssp_workflow.protocol", f"{category}.yml"
    )
    with import_path as pp_path, open(pp_path, "rb") as handle:
        protocol_dict = yaml.safe_load(
            handle
        )  # pylint: disable=attribute-defined-outside-init

    return protocol_dict[name]


RARE_EARTH_ELEMENTS = [
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


def get_standard_structure(
    element: str, prop: str, configuration=None
) -> orm.StructureData:
    """
    get cif abspath from element for bands measure and convergence
    property can be `delta`, `bands` or `convergence`

    The principles are for bands measure, using the configurations from Cottiner's paper since they are the groud state structures exist in real wolrd.
    And for lanthanides using the Nitrides from Wenzowech paper.
    Using the uniaries/diamond configurations for convergence verification.

    The structure is convert to primitive cell except for those magnetic elements
    configurations (typical ones of Cottiner's paper) for bands measure, since we need to
    set the starting magnetizations for sites.

    If prop is delta must provide specific configuration name.

    Args:
        element (str): element
        configuration (str): BCC, FCC, SC, Diamond, XO, XO2, XO3, X2O, X2O3, X2O5

    Returns:
        orm.StructureData: return a orm.StructureData
    """
    # If for delta measure workflow
    if prop == "delta":
        if "O" in configuration:
            data_module = "aiida_sssp_workflow.statics.cif.oxides"
        else:
            data_module = "aiida_sssp_workflow.statics.cif.unaries"

        res_path = importlib.resources.path(
            data_module, f"{element}_{configuration}.cif"
        )

    else:
        # If for bands measure or convergence workflow
        import_path = importlib.resources.path(
            "aiida_sssp_workflow.statics.cif", f"mapping.json"
        )

        with import_path as path, open(path, "r") as handle:
            mapping = json.load(handle)

        dir, fn = mapping[element][prop].split("/")
        res_path = importlib.resources.path(
            f"aiida_sssp_workflow.statics.cif.{dir}", fn
        )

    if element in MAGNETIC_ELEMENTS and prop == "bands":
        primitive_cell = False
    else:
        primitive_cell = True

    with res_path as path:
        structure = orm.CifData.get_or_create(str(path), use_first=True)[
            0
        ].get_structure(primitive_cell=primitive_cell)

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
