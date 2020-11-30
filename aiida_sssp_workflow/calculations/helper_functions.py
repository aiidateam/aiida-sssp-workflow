# -*- coding: utf-8 -*-
"""Collections of helper process functions"""
from aiida import orm
from aiida.engine import calcfunction
from aiida.tools.data.structure import spglib_tuple_to_structure, structure_to_spglib_tuple
import seekpath


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
