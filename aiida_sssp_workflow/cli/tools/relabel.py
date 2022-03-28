import click
from aiida.orm import load_node

from aiida_sssp_workflow.cli import cmd_root


@cmd_root.command("relabel")
@click.argument("label", type=str, nargs=1)
@click.argument("node", type=str, nargs=1)
def relabel_node(label, node):
    """Relabel verification node to extra"""
    _node = load_node(node)
    old_label = _node.extras.get("label")

    if label == old_label:
        click.echo("Please input a new label.")
    else:
        _node.set_extra("label", label)
        click.echo(
            f"uuid={_node.uuid} reset extra label {old_label} to {_node.extras.get('label')}"
        )
