# -*- coding: utf-8 -*-
"""dump verification result"""
import json

import click
from aiida.common import AttributeDict
from aiida.orm import Dict, load_node

from aiida_sssp_workflow.cli import cmd_root


def flatten_output(attr_dict):
    """
    flaten output dict node

    For some namespace do not unfold dict
    - band_parameters
    """
    do_not_unfold = ["band_parameters", "scf_parameters", "seekpath_parameters"]

    for key, value in attr_dict.items():
        if isinstance(value, AttributeDict):
            # keep on unfold
            flatten_output(value)
        elif isinstance(value, Dict) and key not in do_not_unfold:
            attr_dict[key] = value.get_dict()
        else:
            # node type not handled attach uuid
            attr_dict[key] = value.uuid

    return attr_dict


@cmd_root.command("dump")
@click.argument("pk", type=int, nargs=-1)
@click.argument("filename", type=click.Path())
def dump_output(pk, filename):
    """dump the verification result"""
    res = {}
    for p in pk:
        node = load_node(p)
        label = node.extras.get("label")
        attr_dict = node.outputs._construct_attribute_dict(incoming=False)

        res[label] = flatten_output(attr_dict)

    with open(filename, "w") as fh:
        json.dump(dict(res), fh, indent=2)
