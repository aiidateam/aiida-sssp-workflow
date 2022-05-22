# -*- coding: utf-8 -*-
"""
Birch-Murnaghan fit as calcfunction refactor from eosfit.py v3.1 write by Kurt Lejaeghere
Copyright (C) 2012 Kurt Lejaeghere <Kurt.Lejaeghere@UGent.be>, Center for
Molecular Modeling (CMM), Ghent University, Ghent, Belgium
"""
from aiida import orm
from aiida.engine import ExitCode, calcfunction


@calcfunction
def birch_murnaghan_fit(volume_energy: orm.Dict):
    """
    doc
    """
    # pylint: disable=invalid-name
    import numpy as np

    volumes = np.array(volume_energy["volumes"])
    energies = np.array(volume_energy["energies"])
    num_of_atoms = volume_energy["num_of_atoms"]
    fitdata = np.polyfit(volumes ** (-2.0 / 3.0), energies, 3, full=True)

    ssr = fitdata[1]
    sst = np.sum((energies - np.average(energies)) ** 2.0)
    residuals0 = ssr / sst
    deriv0 = np.poly1d(fitdata[0])
    deriv1 = np.polyder(deriv0, 1)
    deriv2 = np.polyder(deriv1, 1)
    deriv3 = np.polyder(deriv2, 1)

    volume0 = 0
    x = 0
    for x in np.roots(deriv1):
        if x > 0 and deriv2(x) > 0:
            volume0 = x ** (-3.0 / 2.0)
            break

    if volume0 == 0:
        return ExitCode(100, f"get spurious volume0={volume0}.")

    if not isinstance(volume0, float):
        # In case where the fitting failed
        return ExitCode(101, f"get spurious volume0={volume0}")

    derivV2 = 4.0 / 9.0 * x**5.0 * deriv2(x)
    derivV3 = -20.0 / 9.0 * x ** (13.0 / 2.0) * deriv2(x) - 8.0 / 27.0 * x ** (
        15.0 / 2.0
    ) * deriv3(x)
    bulk_modulus0 = derivV2 / x ** (3.0 / 2.0)
    bulk_deriv0 = -1 - x ** (-3.0 / 2.0) * derivV3 / derivV2
    energy0 = deriv0(volume0 ** (-2.0 / 3.0))

    echarge = 1.60217733e-19
    return orm.Dict(
        dict={
            "volume0": round(volume0, 7),
            "energy0": round(energy0, 7),
            "num_of_atoms": int(num_of_atoms),
            "bulk_modulus0": round(bulk_modulus0, 7),
            "bulk_modulus0_GPa": round(bulk_modulus0 * echarge * 1.0e21, 7),
            "bulk_deriv0": round(bulk_deriv0, 7),
            "residuals0": round(residuals0[0]),
            "volume0_unit": "A^3",
            "bulk_modulus0_unit": "eV A^3",
            "bulk_modulus0_GPa_unit": "GPa",
        }
    )
