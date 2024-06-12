from pydantic import BaseModel
from importlib import resources
from enum import Enum

from pseudo_parser.upf_parser import parse
from aiida.plugins import DataFactory

UpfData = DataFactory("pseudo.upf")


class CurateType(Enum):
    SSSP = "sssp"
    NC = "nc"


class PseudoInfo(BaseModel):
    element: str
    type: str
    functional: str | None = None
    z_valence: int

    # TODO: more properties parsed from pseudo text
    # version: str
    # source_lib: str
    # ...


def extract_pseudo_info(pseudo_text: str) -> PseudoInfo:
    """Giving a pseudo, extract the pseudo info and return as a `PseudoInfo` object"""
    upf_info = parse(pseudo_text)

    return PseudoInfo(
        element=upf_info["element"],
        type=upf_info["type"],
        z_valence=upf_info["z_valence"],
    )


def parse_std_filename(filename: str, extension: str = "upf") -> PseudoInfo:
    """Parse the standard filename of pseudo and return the `PseudoInfo` object.
    The standard filename should be in the format of:
    `<element>.<type>.<functional>.z_<num_of_valence>.<gen_code>.<src_lib>.<src_lib_version>.<comment1>.<comment2>.<extension>`
    """
    ext = filename.split(".")[-1]
    if ext != extension:
        raise ValueError(f"Invalid extension: {ext}, expected: {extension}")

    element, type, _, z_valence = filename.split(".")[:4]
    num_of_valence = z_valence.split("_")[-1]

    return PseudoInfo(
        element=element,
        type=type,
        z_valence=int(num_of_valence),
    )


def compute_total_nelectrons(configuration, pseudos):
    """Compute the number of electrons of oxide configurations with pseudos

    This function is limited to only computer the total number of electrons of oxides.
    """
    if len(pseudos) != 2:
        raise ValueError(
            f"There are {len(pseudos)} != 2 pseudos, we expect for binary oxides."
        )

    z_O = None
    z_X = None
    for e, p in pseudos.items():
        if e == "O":
            z_O = p.z_valence
        else:
            z_X = p.z_valence

    if z_O is None or z_X is None:
        raise ValueError(
            "Either `O` or `X` pseudos not read properly to get number of valence electrons."
        )

    if configuration == "XO":
        return z_X + z_O

    elif configuration == "XO2":
        return z_X + z_O * 2

    elif configuration == "XO3":
        return z_X + z_O * 3

    elif configuration == "X2O":
        return z_X * 2 + z_O

    elif configuration == "X2O3":
        return z_X * 4 + z_O * 6

    elif configuration == "X2O5":
        return z_X * 4 + z_O * 10
    else:
        raise ValueError(
            f"Cannot compute the number electrons of configuration {configuration}."
        )


def get_pseudo_O(curate_type: CurateType | str = CurateType.SSSP):
    """Return pseudo of oxygen for oxides"""
    match curate_type:
        case CurateType.SSSP:
            import_path = resources.path(
                "aiida_sssp_workflow.statics.upf", "O.paw.pbe.z_6.ld1.psl.v0.1.upf"
            )
            with import_path as psp_path, open(psp_path, "rb") as stream:
                pseudo = UpfData(stream)
            ecutwfc, ecutrho = 70.0, 560.0
        case CurateType.NC:
            import_path = resources.path(
                "aiida_sssp_workflow.statics.upf",
                "O.nc.pbe.z_6.oncvpsp3.dojo.v0.4.1-std.upf",
            )
            with import_path as psp_path, open(psp_path, "rb") as stream:
                pseudo = UpfData(stream)
            ecutwfc, ecutrho = 80.0, 320.0
        case _:
            raise ValueError(f"Unknown curate_type = {curate_type}")

    return pseudo, ecutwfc, ecutrho


# XXX: should do the same as pseudo_O when using for LAN-Nitride band structure calculation.
# Depend on the target curate library type we use the corresponding N pseudos.
def get_pseudo_N():
    """Return pseudo of nitrogen for lanthanide nitrides"""
    import_path = resources.path(
        "aiida_sssp_workflow.statics.upf", "N.us.pbe.z_5.ld1.psl.v0.1.upf"
    )
    with import_path as psp_path, open(psp_path, "rb") as stream:
        pseudo_N = UpfData(stream)

    return pseudo_N, 55.0, 330.0
