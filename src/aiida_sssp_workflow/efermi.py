# -*- coding: utf-8 -*-
"""
Reuse the pure python translate from Nicola's F77 code
Adapt from Austin's repository:
https://github.com/zooks97/bieFermi/tree/main/bieFermi/py
"""

import numpy as np
from scipy.special import erfc

FDCUT = 30.0  # Fermi-Dirac cutoff
HMCUT = 10.0  # Hermite cutoff
POSHMA = -0.5634  # Positive Hermite (cold I) `a`


def find_efermi(
    bands,
    weights,
    nelec: int,
    swidth: float,
    stype: int,
    xacc: float = 1.0e-6,
    jmax: int = 10000,
    nmax: int = 100000,
) -> float:
    """Find the Fermi energy using bisection."""
    # Get min, max eigenvalue and set as initial bounds
    x1 = np.min(bands)
    x2 = np.max(bands)
    x0 = (x1 + x2) / 2

    # Calculate initial f, fmid
    f = smear(bands, weights, x1, nelec, swidth, stype)
    fmid = smear(bands, weights, x2, nelec, swidth, stype)

    # Find bounds which bracket the Fermi energy
    for n in range(1, nmax):
        if f * fmid >= 0:
            x1 = x0 - n * swidth
            x2 = x0 + (n - 0.5) * swidth
            f = smear(bands, weights, x1, nelec, swidth, stype)
            fmid = smear(bands, weights, x2, nelec, swidth, stype)
        else:
            break
    if f * fmid >= 0:
        raise Exception("Could not bracket Fermi energy. Smearing too small?")

    # Set initial fermi energy guess
    if f < 0.0:
        dx = x2 - x1
        rtb = x1
    else:
        dx = x1 - x2
        rtb = x2

    for _ in range(jmax):
        if np.abs(dx) <= xacc or fmid == 0:
            return rtb
        dx = dx * 0.5
        xmid = rtb + dx
        fmid = smear(bands, weights, xmid, nelec, swidth, stype)
        if fmid <= 0:
            rtb = xmid
    raise Exception("Reached maximum number of bisections.")


def smear(bands, weights, xe: float, nelec: int, swidth: float, stype: int) -> float:
    """Calculate smeared value used for bisection."""
    sfuncs = [gaussian, fermid, delthm, spline, poshm, poshm2]

    nkpt, nbnd = bands.shape

    z = 0.0
    for i in range(nkpt):
        for j in range(nbnd):
            x = (xe - bands[i, j]) / swidth
            z += weights[i] * sfuncs[stype - 1](x)

    return z - nelec


def gaussian(x: float) -> float:
    """Gaussian."""
    return 2.0 - erfc(x)


def fermid(x: float) -> float:
    """Fermi-Dirac."""
    x = -x
    if x > FDCUT:
        return 0.0
    elif x < -FDCUT:
        return 2.0
    else:
        return 2.0 / (1.0 + np.exp(x))


def delthm(x: float) -> float:
    """Hermite delta expansion."""
    if x > HMCUT:
        return 2.0
    elif x < -HMCUT:
        return 0.0
    else:
        return (2.0 - erfc(x)) + x * np.exp(-(x**2)) / np.sqrt(np.pi)


def spline(x: float) -> float:
    """Gaussian spline."""
    x = -x
    if x > 0.0:
        fx = np.sqrt(np.e) / 2 * np.exp(-((x + np.sqrt(2.0) / 2.0) ** 2))
    else:
        fx = 1.0 - np.sqrt(np.e) / 2 * np.exp(-((x - np.sqrt(2.0 / 2.0)) ** 2))
    return 2.0 * fx


def poshm(x: float) -> float:
    """Positive Hermite (cold I)."""
    if x > HMCUT:
        return 2.0
    elif x < -HMCUT:
        return 0.0
    else:
        return (2.0 - erfc(x)) + (-2.0 * POSHMA * x * x + 2.0 * x + POSHMA) * np.exp(
            -x * x
        ) / np.sqrt(np.pi) / 2.0


def poshm2(x: float) -> float:
    """Positive Hermite (cold II)."""
    if x > HMCUT:
        return 2.0
    elif x < -HMCUT:
        return 0.0
    else:
        return (2.0 - erfc(x - 1.0 / np.sqrt(2.0))) + np.sqrt(2.0) * np.exp(
            -(x**2) + np.sqrt(2.0) * x - 0.5
        ) / np.sqrt(np.pi)
