from typing import List, Tuple
import yaml
from importlib import resources

from aiida_sssp_workflow.utils.pseudo import DualType, get_dual_type


def get_protocol(category: str, name: str | None = None):
    """Load and read protocol from faml file to a verbose dict
    if name not set, return whole protocol."""
    import_path = resources.path("aiida_sssp_workflow.protocol", f"{category}.yml")
    with import_path as pp_path, open(pp_path, "rb") as handle:
        protocol_dict = yaml.safe_load(handle)  # pylint: disable=attribute-defined-outside-init

    if name:
        return protocol_dict[name]
    else:
        return protocol_dict


def generate_cutoff_list(
    protocol_name: str, element: str, pp_type: str
) -> List[Tuple[int, int]]:
    """From the control protocol name, get the cutoff list"""
    match get_dual_type(pp_type, element):
        case DualType.NC:
            dual_type = "nc_dual_scan"
        case DualType.AUGLOW:
            dual_type = "nonnc_dual_scan"
        case DualType.AUGHIGH:
            dual_type = "nonnc_high_dual_scan"

    dual_scan_list = get_protocol("control", protocol_name)[dual_type]
    if len(dual_scan_list) > 0:
        max_dual = int(max(dual_scan_list))
    else:
        max_dual = 8

    ecutwfc_list = get_protocol("control", protocol_name)["wfc_scan"]

    return [(e, e * max_dual) for e in ecutwfc_list]
