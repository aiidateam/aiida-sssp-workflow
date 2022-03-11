# -*- coding: utf-8 -*-
import os

from aiida import orm
from aiida.engine import calcfunction
from ase import Atoms

# This is all electrons data from 10.1016/j.commatsci.2014.07.030
# lattice unit in a.u.
element_latt = {
    "La": 10.0457,
    "Ce": 9.5612,
    "Pr": 9.5888,
    "Nd": 9.6424,
    "Pm": 9.5234,
    "Sm": 9.5177,
    "Eu": 9.5435,
    "Gd": 9.4306,
    "Tb": 9.2739,
    "Dy": 9.2383,
    "Ho": 9.2233,
    "Er": 9.1641,
    "Tm": 9.1124,
    "Yb": 9.0795,
    "Lu": 9.0103,
}


@calcfunction
def create_lanthanide_nitride(element, latt):
    """ """
    AU_TO_ANGSTROM = 0.529177249

    element = element.value
    latt = latt.value * AU_TO_ANGSTROM

    cell = [[latt, 0.0, 0.0], [0.0, latt, 0.0], [0.0, 0.0, latt]]
    pbc = [True, True, True]
    scaled_positions = [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
        [0.0, 0.0, 0.5],
        [0.5, 0.5, 0.5],
        [0.5, 0.0, 0.0],
        [0.0, 0.5, 0.0],
    ]
    ase_structure = Atoms(
        symbols=f"{element}4N4", cell=cell, pbc=pbc, scaled_positions=scaled_positions
    )
    structure = orm.StructureData(ase=ase_structure)
    cif = structure.get_cif()

    return cif


dir_path = os.path.abspath(
    "/home/unkcpz/Projs/sssp-workflow/aiida-sssp-workflow/aiida_sssp_workflow/REF/CIFs_REN/"
)
for element, latt in element_latt.items():
    cif_data = create_lanthanide_nitride(orm.Str(element), orm.Float(latt))
    cif_file = os.path.join(dir_path, f"{element}N.cif")
    with open(cif_file, "w") as f:
        f.write(cif_data.get_content())
