#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Running verification workchain
"""
import os

import aiida
import click
from aiida import orm
from aiida.engine import run_get_node, submit
from aiida.plugins import DataFactory, WorkflowFactory

from aiida_sssp_workflow.cli import cmd_root
from aiida_sssp_workflow.workflows.verifications import (
    DEFAULT_CONVERGENCE_PROPERTIES_LIST,
    DEFAULT_PROPERTIES_LIST,
)

UpfData = DataFactory("pseudo.upf")
VerificationWorkChain = WorkflowFactory("sssp_workflow.verification")

SSSP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_sssp")


@cmd_root.command("launch")
@click.option(
    "--mode",
    type=click.Choice(["TEST", "PRECHECK", "STANDARD"], case_sensitive=False),
    help="mode of verification.",
)
@click.option(
    "--computer", type=str, help="computer (aiida) to run non-test verification."
)
@click.option(
    "--cleanup/--no-cleanup",
    default=True,
    help="Clean up the remote folder of all calculation, turn this off when your want to see the remote for details.",
)
@click.option("--property", multiple=True, default=[])
@click.argument("filename", type=click.Path(exists=True))
def launch(mode, filename, computer, property, cleanup):
    if not property:
        extra_desc = "all_prop"
        if mode == "PRECHECK":
            properties_list = DEFAULT_CONVERGENCE_PROPERTIES_LIST
        else:
            properties_list = DEFAULT_PROPERTIES_LIST
    else:
        properties_list = list(property)
        extra_desc = f"{properties_list}"

    _profile = aiida.load_profile()
    click.echo(f"Profile: {_profile.name}")

    inputs = inputs_from_mode(
        mode=mode, computer_label=computer, properties_list=properties_list
    )

    inputs["clean_workchain"] = cleanup

    basename = os.path.basename(filename)
    label, _ = os.path.splitext(basename)
    label = orm.Str(f"({mode}-{computer}) {label}")

    with open(filename, "rb") as stream:
        pseudo = UpfData(stream)

    node = run_verification(
        **inputs,
        **{
            "pseudo": pseudo,
            "label": label,
            "extra_desc": extra_desc,
        },
    )

    click.echo(node)
    click.echo(f"calculated on property: {'/'.join(properties_list)}")


def inputs_from_mode(mode, computer_label, properties_list):
    if "imx" in computer_label:
        mpiprocs = 32
        npool = 4
        walltime = 3600
    elif "eiger-mc" in computer_label:
        mpiprocs = 128
        npool = 16
        walltime = 1800
    elif "daint-mc" in computer_label:
        mpiprocs = 36
        npool = 4
        walltime = 3600

    inputs = {}
    if mode == "TEST":
        inputs["pw_code"] = orm.load_code("pw-7.0@localhost")
        inputs["ph_code"] = orm.load_code("ph-7.0@localhost")
        inputs["protocol"] = orm.Str("test")
        inputs["cutoff_control"] = orm.Str("test")
        inputs["criteria"] = orm.Str("efficiency")
        inputs["options"] = orm.Dict(
            dict={
                "resources": {
                    "num_machines": 1,
                    "num_mpiprocs_per_machine": 1,
                },
                "max_wallclock_seconds": 1800,
                "withmpi": False,
            }
        )
        inputs["parallization"] = orm.Dict(dict={})
        inputs["properties_list"] = orm.List(list=properties_list)

    if mode == "PRECHECK":
        inputs["pw_code"] = orm.load_code(f"pw-7.0@{computer_label}")
        inputs["ph_code"] = orm.load_code(f"ph-7.0@{computer_label}")
        inputs["protocol"] = orm.Str("acwf")
        inputs["cutoff_control"] = orm.Str("precheck")
        inputs["criteria"] = orm.Str("precision")
        inputs["options"] = orm.Dict(
            dict={
                "resources": {
                    "num_machines": 1,
                    "num_mpiprocs_per_machine": mpiprocs,
                },
                "max_wallclock_seconds": walltime,
                "withmpi": True,
            }
        )
        inputs["parallization"] = orm.Dict(dict={"npool": npool})
        inputs["properties_list"] = orm.List(list=properties_list)

    if mode == "STANDARD":
        inputs["pw_code"] = orm.load_code(f"pw-7.0@{computer_label}")
        inputs["ph_code"] = orm.load_code(f"ph-7.0@{computer_label}")
        inputs["protocol"] = orm.Str("acwf")
        inputs["cutoff_control"] = orm.Str("standard")
        inputs["criteria"] = orm.Str("efficiency")
        inputs["options"] = orm.Dict(
            dict={
                "resources": {
                    "num_machines": 1,
                    "num_mpiprocs_per_machine": mpiprocs,
                },
                "max_wallclock_seconds": walltime,
                "withmpi": True,
            }
        )
        inputs["parallization"] = orm.Dict(dict={"npool": npool})
        inputs["properties_list"] = orm.List(list=properties_list)

    return inputs


def run_verification(
    pseudo,
    pw_code,
    ph_code,
    protocol,
    cutoff_control,
    criteria,
    options,
    parallization,
    properties_list,
    label,
    extra_desc,
    clean_workchain,
):
    """
    pw_code: code for pw.x calculation
    ph_code: code for ph.x calculation
    upf: upf file to verify
    properties_list: propertios to verified
    label: if None, label will parsed from filename
    mode:
        test to run on localhost with test protocol
        precheck: precheck control protocol on convergence verification
        standard: running a production on eiger
    """
    inputs = {
        "accuracy": {
            "protocol": protocol,
            "cutoff_control": cutoff_control,
        },
        "convergence": {
            "protocol": protocol,
            "cutoff_control": cutoff_control,
            "criteria": criteria,
        },
        "pw_code": pw_code,
        "ph_code": ph_code,
        "pseudo": pseudo,
        "label": label,
        "properties_list": properties_list,
        "options": options,
        "parallelization": parallization,
        "clean_workchain": orm.Bool(clean_workchain),
    }

    if cutoff_control.value == "test":
        _, node = run_get_node(VerificationWorkChain, **inputs)
    else:
        node = submit(VerificationWorkChain, **inputs)

    node.description = f"{label.value} ({extra_desc})"

    return node


if __name__ == "__main__":
    launch()
