import re

from pydantic import BaseModel
from importlib import resources
from enum import Enum
from aiida_pseudo.data.pseudo import UpfData

from .element import HIGH_DUAL_ELEMENTS


REGEX_ELEMENT_V1 = re.compile(r"""(?P<element>[a-zA-Z]{1,2})\s+Element""")
REGEX_ELEMENT_V2 = re.compile(
    r"""\s*element\s*=\s*['"]\s*(?P<element>[a-zA-Z]{1,2})\s*['"].*"""
)

PATTERN_FLOAT = r"[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?"
REGEX_Z_VALENCE_V1 = re.compile(
    r"""(?P<z_valence>""" + PATTERN_FLOAT + r""")\s+Z valence"""
)
REGEX_Z_VALENCE_V2 = re.compile(
    r"""\s*z_valence\s*=\s*['"]\s*(?P<z_valence>""" + PATTERN_FLOAT + r""")\s*['"].*"""
)

REGEX_PSEUDO_TYPE_V1 = re.compile(
    r"""\s*(?P<pseudo_type>(NC|SL|1\/r|US|USPP|PAW))\s*.*\s*pseudopotential"""
)
REGEX_PSEUDO_TYPE_V2 = re.compile(
    r"""\s*pseudo_type\s*=\s*['"]\s*(?P<pseudo_type>(NC|SL|1\/r|US|USPP|PAW))\s*['"].*"""
)


def parse(content: str):
    """parse all"""
    return {
        "element": _parse_element(content),
        "type": _parse_pseudo_type(content),
        "z_valence": _parse_z_valence(content),
    }


def _parse_element(content: str):
    """Parse the content of the UPF file to determine the element.
    :param stream: a filelike object with the binary content of the file.
    :return: the symbol of the element following the IUPAC naming standard.
    """
    for regex in [REGEX_ELEMENT_V2, REGEX_ELEMENT_V1]:
        match = regex.search(content)

        if match:
            return match.group("element")

    raise ValueError(f"could not parse the element from the UPF content: {content}")


def _parse_z_valence(content: str) -> int:
    """Parse the content of the UPF file to determine the Z valence.
    :param stream: a filelike object with the binary content of the file.
    :return: the Z valence.
    """
    for regex in [REGEX_Z_VALENCE_V2, REGEX_Z_VALENCE_V1]:
        match = regex.search(content)

        if match:
            z_valence = match.group("z_valence")

            try:
                z_valence = float(z_valence)
            except ValueError as exception:
                raise ValueError(
                    f"parsed value for the Z valence `{z_valence}` is not a valid number."
                ) from exception

            if int(z_valence) != z_valence:
                raise ValueError(
                    f"parsed value for the Z valence `{z_valence}` is not an integer."
                )

            return int(z_valence)

    raise ValueError(f"could not parse the Z valence from the UPF content: {content}")


def _parse_pseudo_type(content: str) -> str:
    """Parse the content of the UPF file to determien the pseudo type.
    either in one of the NC | SL | 1/r | US | PAW
    :param stream: a filelike object with the binary content of the file.
    :return: the pseudo type str.
    """
    for regex in [REGEX_PSEUDO_TYPE_V2, REGEX_PSEUDO_TYPE_V1]:
        match = regex.search(content)

        if match:
            raw_type = match.group("pseudo_type")
            if "US" in raw_type:
                return "us"
            elif "NC" in raw_type or "SL" in raw_type:
                return "nc"
            elif "PAW" in raw_type:
                return "paw"
            else:
                return raw_type

    raise ValueError(f"could not parse the pseudo_type from the UPF content: {content}")


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

class DualType(Enum):
    NC = "nc"
    AUGLOW = "charge augmentation low"
    AUGHIGH = "charge augmentation high"

def get_dual_type(pp_type: str, element: str) -> DualType:
        if element in HIGH_DUAL_ELEMENTS and pp_type != 'nc':
            return DualType.AUGHIGH
        elif pp_type == 'nc':
            return DualType.NC
        else:
            return DualType.AUGLOW

def extract_pseudo_info(pseudo_text: str) -> PseudoInfo:
    """Giving a pseudo, extract the pseudo info and return as a `PseudoInfo` object"""
    upf_info = parse(pseudo_text)

    return PseudoInfo(
        element=upf_info["element"],
        type=upf_info["type"],
        z_valence=upf_info["z_valence"],
    )


def _get_proper_dual(pp_info: PseudoInfo) -> int:
    if pp_info.type == "nc":
        dual = 4
    else:
        dual = 8

    if pp_info.element in HIGH_DUAL_ELEMENTS and pp_info.type != "nc":
        dual = 18

    return dual


def get_default_dual(pseudo: UpfData) -> int:
    """Based on the pseudo_type, give the cutoffs pairs"""
    pp_info = extract_pseudo_info(pseudo.get_content())
    dual = _get_proper_dual(pp_info)

    return dual


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
