# -*- coding: utf-8 -*-
"""
Refactor from calcDelta.py v3.1 write by Kurt Lejaeghere
Copyright (C) 2012 Kurt Lejaeghere <Kurt.Lejaeghere@UGent.be>, Center for
Molecular Modeling (CMM), Ghent University, Ghent, Belgium
"""
from io import StringIO

import numpy as np
from aiida.engine import calcfunction
from aiida import orm

from aiida_sssp_workflow.calculations.wien2k_ref import WIEN2K_REF, WIEN2K_REN_REF
from aiida_sssp_workflow.helpers import helper_get_v0_b0_b1

# pylint: disable=invalid-name

RARE_EARTH_ELEMENTS = [
    'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er',
    'Tm', 'Yb', 'Lu'
]


@calcfunction
def calculate_delta(element, V0, B0, B1) -> orm.Dict:
    """
    The calcfunction calculate the delta factor.
    return delta factor with unit (eV/atom)
    """
    if element.value in RARE_EARTH_ELEMENTS:
        wien2k_ref = StringIO(WIEN2K_REN_REF)
    else:
        # not Lanthanides elements
        wien2k_ref = StringIO(WIEN2K_REF)

    data_ref = np.loadtxt(wien2k_ref,
                          dtype={
                              'names': ('element', 'V0', 'B0', 'B1'),
                              'formats': ('U2', np.float, np.float, np.float)
                          })
    ref_V0, ref_B0, ref_B1 = helper_get_v0_b0_b1(element.value)

    # Here use dtype U2 to truncate REN to RE
    data_tested = np.array(
        [(element.value, V0.value, B0.value, B1.value)],
        dtype={
            'names': ('element', 'V0', 'B0', 'B1'),
            'formats': ('U2', np.float, np.float, np.float),
        })

    eloverlap = [element.value]
    if not eloverlap:
        # TODO ExitCode
        raise ValueError(
            f'Element {element} is not present in the reference set')
    eloverlap = [element.value]
    # Delta computation
    Delta, Deltarel, Delta1 = _calcDelta(data_tested, data_ref, eloverlap)

    return orm.Dict(
        dict={
            'delta': Delta[0],
            'delta1': Delta1[0],
            'delta_unit': 'meV/atom',
            'delta_relative': Deltarel[0],
            'delta_relative_unit': '%',
            'birch_murnaghan_results': [V0, B0, B1],
            'reference_wien2k_V0_B0_B1': [ref_V0, ref_B0, ref_B1]
        })


def _calcDelta(data_f, data_w, eloverlap, useasymm=False):
    """
    Calculate the Delta using the data in data_f, data_w on
    element in eloverlap
    data_w is all-electrons result as ref
    """
    # pylint: disable=too-many-statements, consider-using-enumerate
    v0w = np.zeros(len(eloverlap))
    b0w = np.zeros(len(eloverlap))
    b1w = np.zeros(len(eloverlap))

    v0f = np.zeros(len(eloverlap))
    b0f = np.zeros(len(eloverlap))
    b1f = np.zeros(len(eloverlap))

    elw = list(data_w['element'])
    elf = list(data_f['element'])

    for i in range(len(eloverlap)):
        searchnr = elw.index(eloverlap[i])
        v0w[i] = data_w['V0'][searchnr]
        b0w[i] = data_w['B0'][searchnr] * 10.**9. / 1.602176565e-19 / 10.**30.
        b1w[i] = data_w['B1'][searchnr]

        searchnr = elf.index(eloverlap[i])
        v0f[i] = data_f['V0'][searchnr]
        b0f[i] = data_f['B0'][searchnr] * 10.**9. / 1.602176565e-19 / 10.**30.
        b1f[i] = data_f['B1'][searchnr]

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
