# -*- coding: utf-8 -*-
"""validate"""


def validate_smearing(parameters, smearing=None):
    """Validate smearing parameters and update the parameters input node accordingly.
    :param parameters: the Dict node that will be used in the inputs
    :param smearing: a tuple of a string and float corresponding to type of smearing and the degauss value
    :raises ValueError: if the input is invalid
    """
    if not any(smearing):
        return

    valid_smearing_types = {
        "gaussian": ["gaussian", "gauss"],
        "methfessel-paxton": ["methfessel-paxton", "m-p", "mp"],
        "cold": ["marzari-vanderbilt", "cold", "m-v", "mv"],
        "fermi-dirac": ["fermi-dirac", "f-d", "fd"],
    }

    for _, options in valid_smearing_types.items():
        if smearing[0] in options:
            break
    else:
        raise ValueError(
            f"the smearing type \"{smearing[0]}\" is invalid, choose from {', '.join(list(valid_smearing_types.keys()))}"
        )

    if not isinstance(smearing[1], float):
        raise ValueError("the smearing value should be a float")

    parameters["SYSTEM"]["occupations"] = "smearing"
    parameters["SYSTEM"]["smearing"] = smearing[0]
    parameters["SYSTEM"]["degauss"] = smearing[1]
