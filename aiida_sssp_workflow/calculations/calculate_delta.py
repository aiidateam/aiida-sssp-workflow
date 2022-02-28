# -*- coding: utf-8 -*-
"""
Refactor from calcDelta.py v3.1 write by Kurt Lejaeghere
Copyright (C) 2012 Kurt Lejaeghere <Kurt.Lejaeghere@UGent.be>, Center for
Molecular Modeling (CMM), Ghent University, Ghent, Belgium
"""
import importlib_resources
import json

import numpy as np
from aiida.engine import calcfunction
from aiida import orm

from aiida_sssp_workflow.calculations.wien2k_ref import WIEN2K_REF, WIEN2K_REN_REF

# pylint: disable=invalid-name

RARE_EARTH_ELEMENTS = [
    'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er',
    'Tm', 'Yb', 'Lu'
]

def helper_get_v0_b0_b1(element: str, structure: str):
    """get eos reference of element"""
    import re

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
    
    echarge = 1.60217733e-19
    return float(V0), float(B0) / (echarge * 1.0e21), float(B1)


@calcfunction
def delta_analyze(element, structure, V0, B0, B1) -> orm.Dict:
    """
    The calcfunction calculate the delta factor.
    return delta factor with unit (eV/atom)
    """
    if 'O' in structure.value:
        # oxides
        import_path = importlib_resources.path('aiida_sssp_workflow.REF.AE_EOS',
                                           'WIEN2K_OXIDES.json')
        with import_path as path, open(path, 'rb') as handle:
            data = json.load(handle)

        BM_fit = data['BM_fit_data'][f'{element.value}-{structure.value}']
        ref_V0, ref_B0, ref_B1 = BM_fit['min_volume'], BM_fit['bulk_modulus_ev_ang3'], BM_fit['bulk_deriv']
    else:
        # unitary structures    
        ref_V0, ref_B0, ref_B1 = helper_get_v0_b0_b1(element.value, structure.value)
    
    # Delta computation
    Delta, Deltarel, Delta1 = _calcDelta(ref_V0, ref_B0, ref_B1, V0.value, B0.value, B1.value)

    nicola_measure = rel_errors_vec_length(ref_V0, ref_B0, ref_B1, V0.value, B0.value, B1.value, config_string=None, prefact=400, weight_b0=1/8, weight_b1=1/64)

    return orm.Dict(
        dict={
            'delta': Delta,
            'delta1': Delta1,
            'delta_unit': 'meV/atom',
            'delta_relative': Deltarel,
            'delta_relative_unit': '%',
            'birch_murnaghan_results': [V0, B0, B1],
            'reference_wien2k_V0_B0_B1': [ref_V0, ref_B0, ref_B1],
            'V0_B0_B1_units_info': 'all atoms for oxide / per atom for unitary + eV A^3',
            'rel_errors_vec_length': nicola_measure,
        })

def rel_errors_vec_length(v0w, b0w, b1w, v0f, b0f, b1f, config_string, prefact, weight_b0, weight_b1):
    """
    Returns the length of the vector formed by the relative error of V0, B0, B1
    THE SIGNATURE OF THIS FUNCTION HAS BEEN CHOSEN TO MATCH THE ONE OF ALL THE OTHER FUNCTIONS
    RETURNING A QUANTITY THAT IS USEFUL FOR COMPARISON, THIS SIMPLIFIES THE CODE LATER.
    Even though config_string is not usd
    """
    V0err =  2*(v0w-v0f)/(v0w+v0f)
    B0err =  2*(b0w-b0f)/(b0w+b0f)
    B1err =  2*(b1w-b1f)/(b1w+b1f)
    leng = np.sqrt(V0err**2+(weight_b0*B0err)**2+(weight_b1*B1err)**2)
    return leng*prefact

def _calcDelta(v0w, b0w, b1w, v0f, b0f, b1f, useasymm=False):
    """
    Calculate the Delta value, function copied from the official DeltaTest repository.
    I don't understand what it does, but it works.
    THE SIGNATURE OF THIS FUNCTION HAS BEEN CHOSEN TO MATCH THE ONE OF ALL THE OTHER FUNCTIONS
    RETURNING A QUANTITY THAT IS USEFUL FOR COMPARISON, THIS SIMPLIFIES THE CODE LATER.
    Even though 'config_string' is useless here.
    """
    # pylint: disable=too-many-statements, consider-using-enumerate

    vref = 30.
    bref = 100. * 10.**9. / 1.602176565e-19 / 10.**30.

    if useasymm:
        Vi = 0.94 * v0w
        Vf = 1.06 * v0w
    else:
        Vi = 0.94 * (v0w + v0f) / 2.
        Vf = 1.06 * (v0w + v0f) / 2.

    a3f = 9. * v0f**3. * b0f / 16. * (b1f - 4.)
    a2f = 9. * v0f**(7. / 3.) * b0f / 16. * (14. - 3. * b1f)
    a1f = 9. * v0f**(5. / 3.) * b0f / 16. * (3. * b1f - 16.)
    a0f = 9. * v0f * b0f / 16. * (6. - b1f)

    a3w = 9. * v0w**3. * b0w / 16. * (b1w - 4.)
    a2w = 9. * v0w**(7. / 3.) * b0w / 16. * (14. - 3. * b1w)
    a1w = 9. * v0w**(5. / 3.) * b0w / 16. * (3. * b1w - 16.)
    a0w = 9. * v0w * b0w / 16. * (6. - b1w)

    x = [0, 0, 0, 0, 0, 0, 0]

    x[0] = (a0f - a0w)**2
    x[1] = 6. * (a1f - a1w) * (a0f - a0w)
    x[2] = -3. * (2. * (a2f - a2w) * (a0f - a0w) + (a1f - a1w)**2.)
    x[3] = -2. * (a3f - a3w) * (a0f - a0w) - 2. * (a2f - a2w) * (a1f - a1w)
    x[4] = -3. / 5. * (2. * (a3f - a3w) * (a1f - a1w) + (a2f - a2w)**2.)
    x[5] = -6. / 7. * (a3f - a3w) * (a2f - a2w)
    x[6] = -1. / 3. * (a3f - a3w)**2.

    y = [0, 0, 0, 0, 0, 0, 0]

    y[0] = (a0f + a0w)**2 / 4.
    y[1] = 3. * (a1f + a1w) * (a0f + a0w) / 2.
    y[2] = -3. * (2. * (a2f + a2w) * (a0f + a0w) + (a1f + a1w)**2.) / 4.
    y[3] = -(a3f + a3w) * (a0f + a0w) / 2. - (a2f + a2w) * (a1f + a1w) / 2.
    y[4] = -3. / 20. * (2. * (a3f + a3w) * (a1f + a1w) + (a2f + a2w)**2.)
    y[5] = -3. / 14. * (a3f + a3w) * (a2f + a2w)
    y[6] = -1. / 12. * (a3f + a3w)**2.

    Fi = np.zeros_like(Vi)
    Ff = np.zeros_like(Vf)

    Gi = np.zeros_like(Vi)
    Gf = np.zeros_like(Vf)

    for n in range(7):
        Fi = Fi + x[n] * Vi**(-(2. * n - 3.) / 3.)
        Ff = Ff + x[n] * Vf**(-(2. * n - 3.) / 3.)

        Gi = Gi + y[n] * Vi**(-(2. * n - 3.) / 3.)
        Gf = Gf + y[n] * Vf**(-(2. * n - 3.) / 3.)

    Delta = 1000. * np.sqrt((Ff - Fi) / (Vf - Vi))
    Deltarel = 100. * np.sqrt((Ff - Fi) / (Gf - Gi))
    if useasymm:
        Delta1 = 1000. * np.sqrt((Ff - Fi) / (Vf - Vi)) \
                 / v0w / b0w * vref * bref
    else:
        Delta1 = 1000. * np.sqrt((Ff - Fi) / (Vf - Vi)) \
                 / (v0w + v0f) / (b0w + b0f) * 4. * vref * bref

    return Delta, Deltarel, Delta1
