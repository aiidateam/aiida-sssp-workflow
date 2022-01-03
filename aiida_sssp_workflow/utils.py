# -*- coding: utf-8 -*-
"""utils.py"""

import collections.abc
import importlib_resources

from aiida.plugins import DataFactory
from aiida import orm

UpfData = DataFactory('pseudo.upf')

RARE_EARTH_ELEMENTS = [
    'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er',
    'Tm', 'Yb', 'Lu'
]

MAGNETIC_ELEMENTS = ['Mn', 'O', 'Cr', 'Fe', 'Co', 'Ni']

NONMETAL_ELEMENTS = [
    'H', 'He', 'B', 'N', 'O', 'F', 'Ne', 'Si', 'P', 'Cl', 'Ar', 'Se', 'Br',
    'Kr', 'Te', 'I', 'Xe', 'Rn'
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


def get_standard_cif_filename_from_element(element: str) -> str:
    """
    get cif filename from element

    NOTICE!!: that for convergence verification the SiF4 structure is used, the
    name of file is `SiF4.cif` which can be get by `element=SiF4`.
    """
    if element in RARE_EARTH_ELEMENTS:
        fpath = importlib_resources.path('aiida_sssp_workflow.REF.CIFs_REN',
                                         f'{element}N.cif')
    else:
        fpath = importlib_resources.path('aiida_sssp_workflow.REF.CIFs',
                                         f'{element}.cif')
    with fpath as path:
        filename = str(path)

    return filename


def parse_upf(upf_content: str) -> dict:
    """

    :param upf_content:
    :param check: if check the integrity of the pp file
    :return:
    """
    from upf_to_json import upf_to_json

    upf_dict = upf_to_json(upf_content, None)

    return upf_dict['pseudo_potential']


def helper_parse_upf(upf: UpfData) -> dict:
    """parser upf"""
    header = parse_upf(upf.get_content())['header']

    return header


def get_default_options(max_num_machines=1,
                        max_wallclock_seconds=1800,
                        with_mpi=False):
    """Return an instance of the options dictionary with the minimally required parameters for a `CalcJob`.

    :param max_num_machines: set the number of nodes, default=1
    :param max_wallclock_seconds: set the maximum number of wallclock seconds, default=1800
    :param with_mpi: whether to run the calculation with MPI enabled
    """
    return {
        'resources': {
            'num_machines': int(max_num_machines)
        },
        'max_wallclock_seconds': int(max_wallclock_seconds),
        'withmpi': with_mpi,
    }


def to_valid_key(name):
    """
    convert name into a valid key name which contain only alphanumeric and underscores
    """
    import re

    valid_name = re.sub(r'[^\w\s]', '_', name)

    return valid_name


def helper_get_magnetic_inputs(structure: orm.StructureData):
    """
    To set initial magnet to the magnetic system, need to set magnetic order to
    every magnetic element site, with certain pw starting_mainetization parameters.
    """
    MAG_INIT_Mn = {'Mn1': 0.5, 'Mn2': -0.3, 'Mn3': 0.5, 'Mn4': -0.3}  # pylint: disable=invalid-name
    MAG_INIT_O = {'O1': 0.5, 'O2': 0.5, 'O3': -0.5, 'O4': -0.5}  # pylint: disable=invalid-name
    MAG_INIT_Cr = {'Cr1': 0.5, 'Cr2': -0.5}  # pylint: disable=invalid-name

    mag_structure = orm.StructureData(cell=structure.cell, pbc=structure.pbc)
    kind_name = structure.get_kind_names()[0]

    # ferromagnetic
    if kind_name in ['Fe', 'Co', 'Ni']:
        for i, site in enumerate(structure.sites):
            mag_structure.append_atom(position=site.position,
                                      symbols=kind_name)

        parameters = {
            'SYSTEM': {
                'nspin': 2,
                'starting_magnetization': {
                    kind_name: 0.2
                },
            },
        }

    #
    if kind_name in ['Mn', 'O', 'Cr']:
        for i, site in enumerate(structure.sites):
            mag_structure.append_atom(position=site.position,
                                      symbols=kind_name,
                                      name=f'{kind_name}{i+1}')

        if kind_name == 'Mn':
            parameters = {
                'SYSTEM': {
                    'nspin': 2,
                    'starting_magnetization': MAG_INIT_Mn,
                },
            }

        if kind_name == 'O':
            parameters = {
                'SYSTEM': {
                    'nspin': 2,
                    'starting_magnetization': MAG_INIT_O,
                },
            }

        if kind_name == 'Cr':
            parameters = {
                'SYSTEM': {
                    'nspin': 2,
                    'starting_magnetization': MAG_INIT_Cr,
                },
            }

    return mag_structure, parameters
