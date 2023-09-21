# -*- coding: utf-8 -*-
"""
Refactor from calcDelta.py v3.1 write by Kurt Lejaeghere
Copyright (C) 2012 Kurt Lejaeghere <Kurt.Lejaeghere@UGent.be>, Center for
Molecular Modeling (CMM), Ghent University, Ghent, Belgium
"""
import importlib
import json

import numpy as np
from aiida import orm
from aiida.engine import calcfunction

from aiida_sssp_workflow.calculations.wien2k_ref import WIEN2K_REF, WIEN2K_REN_REF
from aiida_sssp_workflow.utils import (
    LANTHANIDE_ELEMENTS,
    OXIDE_CONFIGURATIONS,
    UNARIE_CONFIGURATIONS,
)

# pylint: disable=invalid-name


def helper_get_v0_b0_b1(element: str, structure: str):
    """get eos reference of element"""
    import re

    if element in LANTHANIDE_ELEMENTS:
        element_str = f"{element}N"
    else:
        element_str = element

    regex = re.compile(
        rf"""{element_str}\s*
                        (?P<V0>\d*.\d*)\s*
                        (?P<B0>\d*.\d*)\s*
                        (?P<B1>\d*.\d*)""",
        re.VERBOSE,
    )
    if element not in LANTHANIDE_ELEMENTS:
        match = regex.search(WIEN2K_REF)
        V0 = match.group("V0")
        B0 = match.group("B0")
        B1 = match.group("B1")
    else:
        match = regex.search(WIEN2K_REN_REF)
        V0 = match.group("V0")
        B0 = match.group("B0")
        B1 = match.group("B1")

    echarge = 1.60217733e-19
    return float(V0), float(B0) / (echarge * 1.0e21), float(B1)


@calcfunction
def metric_analyze(element, configuration, V0, B0, B1, natoms) -> orm.Dict:
    """
    The calcfunction calculate the metric factor.
    return delta factor with unit (eV/atom)

    The configuration can be one of:
    - RE: for Rare-earth element
    - GS: for BM fit results from Corttiner's paper
    - One of OXIDES:
        - XO
        - XO2
        - XO3
        - X2O
        - X2O3
        - X2O5
    - One of UNARIES:
        - BCC
        - FCC
        - SC
        - Diamond
    - For actinides and Ar, Fr, Ra: using FCC from ACWF dataset

    conf_key is key in json file for configurations of every element.
    """
    element = element.value
    configuration = configuration.value
    V0 = V0.value
    B0 = B0.value
    B1 = B1.value
    natoms = natoms.value
    if configuration == "RE":
        assert element in LANTHANIDE_ELEMENTS

        ref_json = "WIEN2K_LANN.json"
        conf_key = f"{element}N"

    if configuration == "GS":
        ref_json = "WIEN2K_GS.json"
        conf_key = f"{element}"

    if configuration in UNARIE_CONFIGURATIONS:
        ref_json = "AE-average-unaries.json"
        conf_key = f"{element}-X/{configuration}"

    if configuration in OXIDE_CONFIGURATIONS:
        ref_json = "AE-average-oxides.json"
        conf_key = f"{element}-{configuration}"

    import_path = importlib.resources.path(
        "aiida_sssp_workflow.statics.AE_EOS", ref_json
    )
    with import_path as path, open(path, "rb") as handle:
        data = json.load(handle)

    BM_fit = data["BM_fit_data"][conf_key]
    ref_V0, ref_B0, ref_B1 = (
        BM_fit["min_volume"],
        BM_fit["bulk_modulus_ev_ang3"],
        BM_fit["bulk_deriv"],
    )

    results = {
        "birch_murnaghan_results": [V0, B0, B1],
        "reference_ae_V0_B0_B1": [ref_V0, ref_B0, ref_B1],
        "V0_B0_B1_units_info": "eV/A^3 for B0",
    }
    # Delta computation
    try:
        delta, deltarel, delta1 = _calcDelta(ref_V0, ref_B0, ref_B1, V0, B0, B1)
    except Exception:
        pass
    else:
        results.update(
            {
                "delta": delta,
                "delta1": delta1,
                "delta_unit": "meV/atom",
                "delta_relative": deltarel,
                "delta_relative_unit": "%",
                "natoms": natoms,
                "delta/natoms": delta / natoms,
            }
        )

    # The nu_measure is a measure of the relative error of the fit parameters
    try:
        nu_measure = rel_errors_vec_length(
            ref_V0,
            ref_B0,
            ref_B1,
            V0,
            B0,
            B1,
        )
    except Exception:
        pass
    else:
        results.update({"rel_errors_vec_length": nu_measure})

    return orm.Dict(
        dict=results,
    )


def rel_errors_vec_length(
    v0w,
    b0w,
    b1w,
    v0f,
    b0f,
    b1f,
    prefact=100,
    weight_b0=1 / 20,
    weight_b1=1 / 400,
):
    """
    Returns the length of the vector formed by the relative error of V0, B0, B1
    THE SIGNATURE OF THIS FUNCTION HAS BEEN CHOSEN TO MATCH THE ONE OF ALL THE OTHER FUNCTIONS
    RETURNING A QUANTITY THAT IS USEFUL FOR COMPARISON, THIS SIMPLIFIES THE CODE LATER.
    Even though config_string is not usd
    """
    V0err = 2 * (v0w - v0f) / (v0w + v0f)
    B0err = 2 * (b0w - b0f) / (b0w + b0f)
    B1err = 2 * (b1w - b1f) / (b1w + b1f)
    leng = np.sqrt(V0err**2 + (weight_b0 * B0err) ** 2 + (weight_b1 * B1err) ** 2)
    return leng * prefact


def _calcDelta(v0w, b0w, b1w, v0f, b0f, b1f, useasymm=False):
    """
    Calculate the Delta value, function copied from the official DeltaTest repository.
    I don't understand what it does, but it works.
    THE SIGNATURE OF THIS FUNCTION HAS BEEN CHOSEN TO MATCH THE ONE OF ALL THE OTHER FUNCTIONS
    RETURNING A QUANTITY THAT IS USEFUL FOR COMPARISON, THIS SIMPLIFIES THE CODE LATER.
    Even though 'config_string' is useless here.
    """
    # pylint: disable=too-many-statements, consider-using-enumerate

    vref = 30.0
    bref = 100.0 * 10.0**9.0 / 1.602176565e-19 / 10.0**30.0

    if useasymm:
        Vi = 0.94 * v0w
        Vf = 1.06 * v0w
    else:
        Vi = 0.94 * (v0w + v0f) / 2.0
        Vf = 1.06 * (v0w + v0f) / 2.0

    a3f = 9.0 * v0f**3.0 * b0f / 16.0 * (b1f - 4.0)
    a2f = 9.0 * v0f ** (7.0 / 3.0) * b0f / 16.0 * (14.0 - 3.0 * b1f)
    a1f = 9.0 * v0f ** (5.0 / 3.0) * b0f / 16.0 * (3.0 * b1f - 16.0)
    a0f = 9.0 * v0f * b0f / 16.0 * (6.0 - b1f)

    a3w = 9.0 * v0w**3.0 * b0w / 16.0 * (b1w - 4.0)
    a2w = 9.0 * v0w ** (7.0 / 3.0) * b0w / 16.0 * (14.0 - 3.0 * b1w)
    a1w = 9.0 * v0w ** (5.0 / 3.0) * b0w / 16.0 * (3.0 * b1w - 16.0)
    a0w = 9.0 * v0w * b0w / 16.0 * (6.0 - b1w)

    x = [0, 0, 0, 0, 0, 0, 0]

    x[0] = (a0f - a0w) ** 2
    x[1] = 6.0 * (a1f - a1w) * (a0f - a0w)
    x[2] = -3.0 * (2.0 * (a2f - a2w) * (a0f - a0w) + (a1f - a1w) ** 2.0)
    x[3] = -2.0 * (a3f - a3w) * (a0f - a0w) - 2.0 * (a2f - a2w) * (a1f - a1w)
    x[4] = -3.0 / 5.0 * (2.0 * (a3f - a3w) * (a1f - a1w) + (a2f - a2w) ** 2.0)
    x[5] = -6.0 / 7.0 * (a3f - a3w) * (a2f - a2w)
    x[6] = -1.0 / 3.0 * (a3f - a3w) ** 2.0

    y = [0, 0, 0, 0, 0, 0, 0]

    y[0] = (a0f + a0w) ** 2 / 4.0
    y[1] = 3.0 * (a1f + a1w) * (a0f + a0w) / 2.0
    y[2] = -3.0 * (2.0 * (a2f + a2w) * (a0f + a0w) + (a1f + a1w) ** 2.0) / 4.0
    y[3] = -(a3f + a3w) * (a0f + a0w) / 2.0 - (a2f + a2w) * (a1f + a1w) / 2.0
    y[4] = -3.0 / 20.0 * (2.0 * (a3f + a3w) * (a1f + a1w) + (a2f + a2w) ** 2.0)
    y[5] = -3.0 / 14.0 * (a3f + a3w) * (a2f + a2w)
    y[6] = -1.0 / 12.0 * (a3f + a3w) ** 2.0

    Fi = np.zeros_like(Vi)
    Ff = np.zeros_like(Vf)

    Gi = np.zeros_like(Vi)
    Gf = np.zeros_like(Vf)

    for n in range(7):
        Fi = Fi + x[n] * Vi ** (-(2.0 * n - 3.0) / 3.0)
        Ff = Ff + x[n] * Vf ** (-(2.0 * n - 3.0) / 3.0)

        Gi = Gi + y[n] * Vi ** (-(2.0 * n - 3.0) / 3.0)
        Gf = Gf + y[n] * Vf ** (-(2.0 * n - 3.0) / 3.0)

    Delta = 1000.0 * np.sqrt((Ff - Fi) / (Vf - Vi))
    Deltarel = 100.0 * np.sqrt((Ff - Fi) / (Gf - Gi))
    if useasymm:
        Delta1 = 1000.0 * np.sqrt((Ff - Fi) / (Vf - Vi)) / v0w / b0w * vref * bref
    else:
        Delta1 = (
            1000.0
            * np.sqrt((Ff - Fi) / (Vf - Vi))
            / (v0w + v0f)
            / (b0w + b0f)
            * 4.0
            * vref
            * bref
        )

    return Delta, Deltarel, Delta1
