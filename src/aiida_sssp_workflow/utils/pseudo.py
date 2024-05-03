from pydantic import BaseModel

from pseudo_parser.upf_parser import parse


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
