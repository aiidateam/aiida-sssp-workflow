import yaml
from importlib import resources


def get_protocol(category, name=None):
    """Load and read protocol from faml file to a verbose dict
    if name not set, return whole protocol."""
    import_path = resources.path("aiida_sssp_workflow.protocol", f"{category}.yml")
    with import_path as pp_path, open(pp_path, "rb") as handle:
        protocol_dict = yaml.safe_load(handle)  # pylint: disable=attribute-defined-outside-init

    if name:
        return protocol_dict[name]
    else:
        return protocol_dict
