# -*- coding: utf-8 -*-
"""
calculate bands distance
"""
import numpy as np

from aiida import orm
from aiida.engine import calcfunction


@calcfunction
def calculate_bands_distance(bands_structure_a: orm.BandsData,
                             bands_parameters_a: orm.Dict,
                             bands_structure_b: orm.BandsData,
                             bands_parameters_b: orm.Dict, smearing: orm.Float,
                             is_metal: orm.Bool):
    """doc"""
    res = get_bands_distance(bands_structure_a, bands_structure_b,
                             bands_parameters_a, bands_parameters_b,
                             smearing.value, is_metal.value)

    return orm.Dict(
        dict={
            'eta_v': res.get('eta_v', None),
            'shift_v': res.get('shift_v', None),
            'max_diff_v': res.get('max_diff_v', None),
            'eta_10': res.get('eta_10', None),
            'shift_10': res.get('shift_10', None),
            'max_diff_10': res.get('max_diff_10', None),
            'bands_unit': 'eV',
        })


def get_homo(bands, num_electrons: int):
    """
    This function only work for insulator,
    therefore the num_electrons is even number.
    """
    assert num_electrons % 2 == 0, f'There are {num_electrons} electrons, must be metal'
    # get homo band
    band = bands[:, num_electrons // 2 - 1]
    return max(band)


def fermi_dirac(band_energy, fermi_energy, smearing):
    """
    The first argument can be an array
    """
    old_settings = np.seterr(over='raise', divide='raise')
    try:
        res = 1.0 / (np.exp((band_energy - fermi_energy) / smearing) + 1.0)
    except FloatingPointError:
        res = np.heaviside(fermi_energy - band_energy, 1.0)
    np.seterr(**old_settings)

    return res


def retrieve_bands(bandsdata: orm.BandsData, start_band, num_electrons, efermi,
                   smearing, is_metal):
    """
    docstring
    """
    from efermi import pyefermi

    bands = bandsdata.get_bands()
    bands = bands - efermi  # shift all bands to fermi energy 0
    bands = bands[:, start_band:]
    output_bands = orm.ArrayData()
    output_bands.set_array('kpoints', bandsdata.get_kpoints())
    output_bands.set_array('bands', bands)

    if is_metal:
        nelectrons = num_electrons
        nkpoints = np.shape(bands)[0]
        weights = np.ones(nkpoints) / nkpoints
        bands = np.asfortranarray(bands)
        meth = 2  # firmi-dirac smearing

        output_efermi = pyefermi(bands, weights, nelectrons, smearing,
                                 nkpoints, meth)

    else:
        homo_energy = get_homo(bands, num_electrons)
        output_efermi = homo_energy

    return {
        'bands': output_bands,
        'efermi': output_efermi,
    }


def calculate_eta_and_max_diff(bands_a: orm.ArrayData, bands_b: orm.ArrayData,
                               efermi_a, efermi_b, fermi_shift, smearing):
    """
    docstring
    """
    from functools import partial
    from scipy.optimize import minimize

    bands_a = bands_a.get_array('bands')
    bands_b = bands_b.get_array('bands')
    num_bands = min(np.shape(bands_a)[1], np.shape(bands_b)[1])

    assert np.shape(bands_a)[0] == np.shape(
        bands_b)[0], 'have different kpoints'

    # truncate the bands to same size
    bands_a = bands_a[:, :num_bands - 1]
    bands_b = bands_b[:, :num_bands - 1]

    occ_a = fermi_dirac(bands_a, efermi_a + fermi_shift, smearing)
    occ_b = fermi_dirac(bands_b, efermi_b + fermi_shift, smearing)
    occ = np.sqrt(occ_a * occ_b)

    bands_diff = bands_a - bands_b

    def fun_shift(occ, bands_diff, shift):
        return np.sqrt(np.sum(occ * (bands_diff + shift)**2) / np.sum(occ))

    # Compute eta
    eta_val = partial(fun_shift, occ, bands_diff)
    results = minimize(eta_val, np.array([0.0]), method='Nelder-Mead')
    eta = results.get('fun')
    shift = results.get('x')[0]

    # Compute max_diff:
    # then find from abs_diff the max one of which the occ > 0.5
    # first make occ > 0.5 to be 1 and occ <= 0.5 to be 0, then element-wise the
    # new occ matrix with abs_diff, and find the max diff
    abs_diff = np.abs(bands_diff + shift)
    new_occ = np.heaviside(occ - 0.5, 0.0)
    new_diff = np.multiply(abs_diff, new_occ)
    max_diff = np.amax(new_diff)

    return {
        'eta': eta,
        'shift': shift,
        'max_diff': max_diff,
    }


def get_bands_distance(bands_a: orm.BandsData, bands_b: orm.BandsData,
                       band_parameters_a: orm.Dict,
                       band_parameters_b: orm.Dict, smearing, is_metal):
    """
    TODO docstring
    """
    num_electrons_a = band_parameters_a['number_of_electrons']
    num_electrons_b = band_parameters_b['number_of_electrons']
    efermi_a = band_parameters_a['fermi_energy']
    efermi_b = band_parameters_b['fermi_energy']

    if num_electrons_a <= num_electrons_b:
        num_electrons = int(num_electrons_a)
        res = retrieve_bands(bands_a, 0, num_electrons, efermi_a, smearing,
                             is_metal)
        bands_a = res['bands']
        efermi_a = res['efermi']

        start_band = int(num_electrons_b - num_electrons_a) // 2
        res = retrieve_bands(bands_b, start_band, num_electrons, efermi_b,
                             smearing, is_metal)
        bands_b = res['bands']
        efermi_b = res['efermi']
    else:
        # num_electrons_b < num_electrons_a:
        num_electrons = int(num_electrons_b)
        start_band = int(num_electrons_a - num_electrons_b) // 2
        res = retrieve_bands(bands_a, start_band, num_electrons, efermi_a,
                             smearing, is_metal)
        bands_a = res['bands']
        efermi_a = res['efermi']

        res = retrieve_bands(bands_b, 0, num_electrons, efermi_b, smearing,
                             is_metal)
        bands_b = res['bands']
        efermi_b = res['efermi']

    # eta_v
    fermi_shift = 0.0
    if is_metal:
        smearing_v = smearing
    else:
        smearing_v = 0

    outputs = calculate_eta_and_max_diff(bands_a, bands_b, efermi_a, efermi_b,
                                         fermi_shift, smearing_v)
    eta_v = outputs.get('eta')
    shift_v = outputs.get('shift')
    max_diff_v = outputs.get('max_diff')

    # eta_10
    fermi_shift = 10.0
    # if not metal
    smearing_10 = smearing
    outputs = calculate_eta_and_max_diff(bands_a, bands_b, efermi_a, efermi_b,
                                         fermi_shift, smearing_10)

    eta_10 = outputs.get('eta')
    shift_10 = outputs.get('shift')
    max_diff_10 = outputs.get('max_diff')

    return {
        'eta_v': eta_v,
        'shift_v': shift_v,
        'max_diff_v': max_diff_v,
        'eta_10': eta_10,
        'shift_10': shift_10,
        'max_diff_10': max_diff_10,
    }
