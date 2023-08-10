# -*- coding: utf-8 -*-
import re

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
        "element": parse_element(content),
        "pp_type": parse_pseudo_type(content),
        "z_valence": parse_z_valence(content),
    }


def parse_element(content: str):
    """Parse the content of the UPF file to determine the element.
    :param stream: a filelike object with the binary content of the file.
    :return: the symbol of the element following the IUPAC naming standard.
    """
    for regex in [REGEX_ELEMENT_V2, REGEX_ELEMENT_V1]:
        match = regex.search(content)

        if match:
            return match.group("element")

    raise ValueError(f"could not parse the element from the UPF content: {content}")


def parse_z_valence(content: str) -> int:
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


def parse_pseudo_type(content: str) -> str:
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
