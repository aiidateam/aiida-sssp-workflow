# -*- coding: utf-8 -*-
"""utils.py"""

import collections.abc
import importlib_resources

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
    if element in RARE_EARTH_ELEMENTS:
        fpath = importlib_resources.path('aiida_sssp_workflow.REF.CIFs_REN',
                                         f'{element}N.cif')
    else:
        fpath = importlib_resources.path('aiida_sssp_workflow.REF.CIFs',
                                         f'{element}.cif')
    with fpath as path:
        filename = str(path)

    return filename
