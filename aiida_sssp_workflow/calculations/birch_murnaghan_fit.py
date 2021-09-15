# -*- coding: utf-8 -*-
"""
Birch-Murnaghan fit as calcfunction refactor from eosfit.py v3.1 write by Kurt Lejaeghere
Copyright (C) 2012 Kurt Lejaeghere <Kurt.Lejaeghere@UGent.be>, Center for
Molecular Modeling (CMM), Ghent University, Ghent, Belgium
"""
from aiida.engine import calcfunction, ExitCode
from aiida import orm


@calcfunction
def birch_murnaghan_fit(volume_energy: orm.Dict):
    """
    doc
    """
    # pylint: disable=invalid-name
    import numpy as np

    volumes = np.array(list(volume_energy['volumes'].values()))
    energies = np.array(list(volume_energy['energies'].values()))
    fitdata = np.polyfit(volumes**(-2. / 3.), energies, 3, full=True)

    ssr = fitdata[1]
    sst = np.sum((energies - np.average(energies))**2.)
    residuals0 = ssr / sst
    deriv0 = np.poly1d(fitdata[0])
    deriv1 = np.polyder(deriv0, 1)
    deriv2 = np.polyder(deriv1, 1)
    deriv3 = np.polyder(deriv2, 1)

    volume0 = 0
    x = 0
    for x in np.roots(deriv1):
        if x > 0 and deriv2(x) > 0:
            volume0 = x**(-3. / 2.)
            break

    if volume0 == 0:
        return ExitCode(100, f'get spurious volume0={volume0}.')

    if not isinstance(volume0, float):
        # In case where the fitting failed
        return ExitCode(101, f'get spurious volume0={volume0}')

    derivV2 = 4. / 9. * x**5. * deriv2(x)
    derivV3 = (-20. / 9. * x**(13. / 2.) * deriv2(x) -
               8. / 27. * x**(15. / 2.) * deriv3(x))
    bulk_modulus0 = derivV2 / x**(3. / 2.)
    bulk_deriv0 = -1 - x**(-3. / 2.) * derivV3 / derivV2
    energy0 = deriv0(volume0**(-2. / 3.))

    echarge = 1.60217733e-19
    return orm.Dict(
        dict={
            'volume0': round(volume0, 7),
            'energy0': round(energy0, 7),
            'bulk_modulus0': round(bulk_modulus0 * echarge * 1.0e21, 7),
            'bulk_deriv0': round(bulk_deriv0, 7),
            'residuals0': round(residuals0[0]),
            'volume0_unit': 'A^3/atom',
            'bulk_modulus0_unit': 'GPa',
        })
