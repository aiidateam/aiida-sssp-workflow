# -*- coding: utf-8 -*-
"""utils.py"""

import collections.abc
import importlib

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
        protocol_dict = yaml.safe_load(handle)  # pylint: disable=attribute-defined-outside-init

    if name:
        return protocol_dict[name]
    else:
        return protocol_dict


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


def parse_label(label):
    """parse standard pseudo label to dict of pseudo info"""
    element, type, z, tool, family, *version = label.split(".")
    version = ".".join(version)

    if type == "nc":
        full_type = "NC"
    if type == "us":
        full_type = "Ultrasoft"
    if type == "paw":
        full_type = "PAW"

    return {
        "element": element,
        "type": type,
        "z": z,
        "tool": tool,
        "family": family,
        "version": version,
        "representive_label": f"{z}|{full_type}|{family}|{tool}|{version}",
        "concise_label": f"{z}|{type}|{family}|{version}",
        "full_label": f"{element}|{z}|{full_type}|{family}|{tool}|{version}",
    }


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
