# -*- coding: utf-8 -*-
"""
Module comtain helper functions for workflows
"""
import importlib_resources

from aiida import orm
from aiida.engine import calcfunction
from aiida.plugins import DataFactory
from aiida.tools.data.structure import spglib_tuple_to_structure, structure_to_spglib_tuple
import seekpath

from aiida_sssp_workflow.utils import RARE_EARTH_ELEMENTS, \
    get_standard_cif_filename_from_element

UpfData = DataFactory('pseudo.upf')


@calcfunction
def helper_get_primitive_structure(structure,
                                   **parameters) -> orm.StructureData:
    """
    :param structure: The AiiDA StructureData for which we want to obtain
        the primitive structure.

    :param parameters: A dictionary whose key-value pairs are passed as
        additional kwargs to the ``seekpath.get_explicit_k_path`` function.

    """
    structure_tuple, kind_info, kinds = structure_to_spglib_tuple(structure)

    rawdict = seekpath.get_explicit_k_path(structure=structure_tuple,
                                           **parameters)

    # Replace primitive structure with AiiDA StructureData
    primitive_lattice = rawdict.pop('primitive_lattice')
    primitive_positions = rawdict.pop('primitive_positions')
    primitive_types = rawdict.pop('primitive_types')
    primitive_tuple = (primitive_lattice, primitive_positions, primitive_types)
    primitive_structure = spglib_tuple_to_structure(primitive_tuple, kind_info,
                                                    kinds)

    return primitive_structure


def helper_get_base_inputs(pseudo: UpfData, primitive_cell=True):
    """
    helper method used to generate base pw inputs(structure, pseudos, pw_parameters).
    lanthanides elements are supported with Rare-Nithides.
    """
    element = pseudo.element

    pseudos = {element: pseudo}

    if element == 'F':
        # set element to 'SiF4' to use SiF4 structure for fluorine
        element = orm.Str('SiF4')

        fpath = importlib_resources.path('aiida_sssp_workflow.REF.UPFs',
                                         'Si.pbe-n-rrkjus_psl.1.0.0.UPF')
        with fpath as path:
            filename = str(path)
            upf_silicon = UpfData.get_or_create(filename)[0]
            pseudos['Si'] = upf_silicon

    cif_file = get_standard_cif_filename_from_element(element)

    cif_data = orm.CifData.get_or_create(cif_file)[0]

    return {
        'structure': cif_data.get_structure(primitive_cell=primitive_cell),
        'pseudos': pseudos,
        'base_pw_parameters': pw_parameters,
    }


def helper_get_v0_b0_b1(element: str):
    """get eos reference of element"""
    import re
    from aiida_sssp_workflow.calculations.wien2k_ref import WIEN2K_REF, WIEN2K_REN_REF

    if element == 'F':
        # Use SiF4 as reference of fluorine(F)
        return 19.3583, 74.0411, 4.1599

    if element in RARE_EARTH_ELEMENTS:
        element_str = f'{element}N'
    else:
        element_str = element

    regex = re.compile(
        rf"""{element_str}\s*
                        (?P<V0>\d*.\d*)\s*
                        (?P<B0>\d*.\d*)\s*
                        (?P<B1>\d*.\d*)""", re.VERBOSE)
    if element not in RARE_EARTH_ELEMENTS:
        match = regex.search(WIEN2K_REF)
        V0 = match.group('V0')
        B0 = match.group('B0')
        B1 = match.group('B1')
    else:
        match = regex.search(WIEN2K_REN_REF)
        V0 = match.group('V0')
        B0 = match.group('B0')
        B1 = match.group('B1')

    return float(V0), float(B0), float(B1)
