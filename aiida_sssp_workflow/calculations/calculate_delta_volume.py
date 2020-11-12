# -*- coding: utf-8 -*-
"""
calculate the convergence behaviour of pressure as a unit free value.
δV_press = (V′ −V0)/V0, where V0 is the reference all-electron equilibrium volume
and the V′ is solved from δP of Birch–Murnaghan equation of state for the pressure
"""
import numpy as np
from aiida import orm
from aiida.engine import calcfunction


def get_volume_from_pressure_birch_murnaghan(P, V0, B0, B1):
    """
    Knowing the pressure P and the Birch-Murnaghan equation of state
    parameters, gets the volume the closest to V0 (relatively) that is
    such that P_BirchMurnaghan(V)=P
    """

    # coefficients of the polynomial in x=(V0/V)^(1/3) (aside from the
    # constant multiplicative factor 3B0/2)
    polynomial = [3. / 4. * (B1 - 4.), 0, 1. - 3. / 2. * (B1 - 4.), 0, 3. / 4. * (B1 - 4.) - 1., 0,
                  0, 0, 0, -2 * P / (3. * B0)]
    V = min([V0 / (x.real ** 3) for x in np.roots(polynomial)
             if abs(x.imag) < 1e-8 * abs(x.real)], key=lambda V: abs(V - V0) / float(V0))
    return V

@calcfunction
def calculate_delta_volume(pressures: orm.List,
                           equilibrium_refs: orm.Dict,
                           pressure_reference: orm.Float) -> orm.List:
    """
    calculate the convergence behaviour of pressure as a unit free value.
    """
    V0 = equilibrium_refs['V0']
    B0 = equilibrium_refs['B0']
    BP = equilibrium_refs['BP']
    v_ref = get_volume_from_pressure_birch_murnaghan(pressure_reference.value, V0, B0, BP)
    volumes = [get_volume_from_pressure_birch_murnaghan(p,V0,B0,BP) for p in pressures.get_list()]
    delta_volume = [(v-v_ref) / v_ref for v in volumes]

    return orm.List(list=list(delta_volume))