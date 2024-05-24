LANTHANIDE_ELEMENTS = [
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
]

ACTINIDE_ELEMENTS = [
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
]

# The elements that usually have magnetic configurations.
MAGNETIC_ELEMENTS = ["Mn", "O", "Cr", "Fe", "Co", "Ni"]

# The elements that usually not metallic in the ground state.
NON_METALLIC_ELEMENTS = [
    "H",
    "He",
    "B",
    "N",
    "O",
    "F",
    "Ne",
    "Si",
    "P",
    "Cl",
    "Ar",
    "Se",
    "Br",
    "Kr",
    "Te",
    "I",
    "Xe",
    "Rn",
]

# The elements don't have typical configurations from sci2016 paper
NO_GS_CONF_ELEMENTS = ["At", "Fr", "Ra"]

# The elements that from our knowledge require high charge density cutoff.
HIGH_DUAL_ELEMENTS = ["O", "Fe", "Mn", "Hf", "Co", "Ni", "Cr"]

# The element that not has unaries structure from nat.phys.rev (maxium to Z=108)
UNSUPPORTED_ELEMENTS = [
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
]
