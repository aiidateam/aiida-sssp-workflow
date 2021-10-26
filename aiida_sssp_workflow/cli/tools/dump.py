# -*- coding: utf-8 -*-
"""dump verification result"""
import json
import click

from aiida.orm import load_node, Dict

from aiida_sssp_workflow.cli import cmd_root


def flaten_output(attr_dict):
    """flaten output dict node"""
    for key, value in attr_dict.items():
        if isinstance(value, Dict):
            attr_dict[key] = value.get_dict()
        else:
            flaten_output(value)


@cmd_root.command('dump')
@click.argument('pk', nargs=1, type=int)
def dump_output(pk):
    """dump the verification result"""
    node = load_node(pk)
    res = {}

    outputs = node.outputs._construct_attribute_dict(False)
    flaten_output(outputs)
    res.update(outputs)

    json_obj = json.dumps(dict(res), indent=2)
    print(json_obj)
