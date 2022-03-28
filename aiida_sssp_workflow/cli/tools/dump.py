# -*- coding: utf-8 -*-
"""dump verification result"""
import json

import click
from aiida.common import AttributeDict
from aiida.orm import Dict, load_node

from aiida_sssp_workflow.cli import cmd_root


def flatten_output(attr_dict, node_collection):
    """
    flaten output dict node
    node_collection is a list to accumulate the nodes that not unfolded

    For some namespace do not unfold dict
    - band_parameters

    For output nodes not being expanded, write down the uuid for making archive
    """
    do_not_unfold = ["band_parameters", "scf_parameters", "seekpath_parameters"]

    for key, value in attr_dict.items():
        if isinstance(value, AttributeDict):
            # keep on unfold
            flatten_output(value, node_collection)
        elif isinstance(value, Dict) and key not in do_not_unfold:
            attr_dict[key] = value.get_dict()
        else:
            # node type not handled attach uuid
            attr_dict[key] = value.uuid
            node_collection.append(value.uuid)

    # print(archive_uuids)
    return attr_dict


@cmd_root.command("dump")
@click.argument("nodes", type=str, nargs=-1)
@click.argument("filename", type=click.Path())
@click.argument("archive", type=click.Path())
def dump_output(nodes, filename, archive):
    """dump the verification result"""
    from aiida.tools.importexport import ExportFileFormat, export

    res = {}
    archive_uuids = []
    for node in nodes:
        _node = load_node(node)
        label = _node.extras.get("label")
        attr_dict = _node.outputs._construct_attribute_dict(incoming=False)

        res[label] = flatten_output(attr_dict, archive_uuids)

    with open(filename, "w") as fh:
        json.dump(dict(res), fh, indent=2)

    kwargs = {
        "input_calc_forward": False,
        "input_work_forward": False,
        "create_backward": False,
        "return_backward": False,
        "call_calc_backward": False,
        "call_work_backward": False,
        "include_comments": False,
        "include_logs": False,
        "overwrite": True,
    }

    # print(archive_uuids)
    export(
        [load_node(i) for i in archive_uuids],
        filename=archive,
        file_format=ExportFileFormat.ZIP,
        **kwargs
    )
