"""
Module comtain helper functions for workflows
"""
import importlib_resources

from aiida import orm

from aiida_sssp_workflow.utils import helper_parse_upf, \
    RARE_EARTH_ELEMENTS, \
    get_standard_cif_filename_from_element


def get_pw_inputs_from_pseudo(pseudo, primitive_cell=True):
    """
    helper method used to generate base pw inputs(structure, pseudos, pw_parameters)
    lanthanides elements are supported with Rare-Nithides.
    """
    upf_info = helper_parse_upf(pseudo)
    element = orm.Str(upf_info['element'])

    pseudos = {element.value: pseudo}

    if element.value == 'F':
        element = orm.Str('SiF4')

        fpath = importlib_resources.path('aiida_sssp_workflow.REF.UPFs',
                                         'Si.pbe-n-rrkjus_psl.1.0.0.UPF')
        with fpath as path:
            filename = str(path)
            upf_silicon = orm.UpfData.get_or_create(filename)[0]
            pseudos['Si'] = upf_silicon

    pw_parameters = {}
    if element.value in RARE_EARTH_ELEMENTS:
        fpath = importlib_resources.path('aiida_sssp_workflow.REF.UPFs',
                                         'N.pbe-n-radius_5.UPF')
        with fpath as path:
            filename = str(path)
            upf_nitrogen = orm.UpfData.get_or_create(filename)[0]
            pseudos['N'] = upf_nitrogen

        z_valence_RE = upf_info['z_valence']  # pylint: disable=invalid-name
        z_valence_N = helper_parse_upf(upf_nitrogen)['z_valence']  # pylint: disable=invalid-name
        nbands = (z_valence_N + z_valence_RE) // 2
        nbands_factor = 2
        pw_parameters = {
            'SYSTEM': {
                'nbnd': int(nbands * nbands_factor),
            },
        }

    filename = get_standard_cif_filename_from_element(element.value)

    cif_data = orm.CifData.get_or_create(filename)[0]

    return {
        'structure': cif_data.get_structure(primitive_cell=primitive_cell),
        'pseudos': pseudos,
        'base_pw_parameters': pw_parameters,
    }
