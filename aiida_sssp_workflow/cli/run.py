#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Running verification workchain
"""
import os

import aiida
import click
from aiida import orm
from aiida.cmdline.params import options, types
from aiida.engine import run_get_node, submit
from aiida.plugins import DataFactory, WorkflowFactory

from aiida_sssp_workflow.cli import cmd_root
from aiida_sssp_workflow.workflows.verifications import DEFAULT_PROPERTIES_LIST

UpfData = DataFactory("pseudo.upf")
VerificationWorkChain = WorkflowFactory("sssp_workflow.verification")

SSSP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_sssp")

# Trigger the launch by running:
# aiida-sssp-workflow launch --property accuracy.delta --pw-code pw-7.0@localhost --ph-code ph-7.0@localhost --protocol test --cutoff-control test --criteria efficiency --withmpi True -- examples/_static/Si_ONCV_PBE-1.2.upf


@cmd_root.command("launch")
@options.OverridableOption(
    "--pw-code", "pw_code", type=types.CodeParamType(entry_point="quantumespresso.pw")
)(required=True)
@options.OverridableOption(
    "--ph-code", "ph_code", type=types.CodeParamType(entry_point="quantumespresso.ph")
)(required=True)
@click.option(
    "--property",
    multiple=True,
    default=[],
    help="Property to verify, can be: accuracy.delta, accuracy.bands, convergence ...",
)
@click.option(
    "protocol",
    "--protocol",
    default="standard",
    help="Protocol to use for the verification.",
)
@click.option(
    "cutoff_control",
    "--cutoff-control",
    default="standard",
    help="Control of convergence.",
)
@click.option(
    "criteria", "--criteria", default="efficiency", help="Criteria of convergence."
)
@click.option("withmpi", "--withmpi", default=True, help="Run with mpi.")
@click.option("npool", "--npool", default=1, help="Number of pool.")
@click.option("walltime", "--walltime", default=3600, help="Walltime.")
@click.option("num_mpiprocs", "--num-mpiprocs", default=1, help="Number of mpiprocs.")
@click.option(
    "--clean-workchain/--no-clean-workchain",
    default=True,
    help="Clean up the remote folder of all calculation, turn this off when your want to see the remote for details.",
)
@click.option(
    "--daemon/--no-daemon",
    default=True,
    help="Launch the verfication to daemon by submit or run directly.",
)
@click.argument("pseudo", type=click.Path(exists=True))
def launch(
    pw_code,
    ph_code,
    property,
    protocol,
    cutoff_control,
    criteria,
    withmpi,
    npool,
    walltime,
    num_mpiprocs,
    pseudo,
    clean_workchain,
    daemon,
):
    """Launch the verification workchain."""
    from aiida.orm import load_code

    # if the property is not specified, use the default list with all properties calculated.
    # otherwise, use the specified properties.
    if not property:
        properties_list = DEFAULT_PROPERTIES_LIST
        extra_desc = "All properties"
    else:
        properties_list = list(property)
        extra_desc = f"{properties_list}"

    _profile = aiida.load_profile()
    click.echo(f"Current profile: {_profile.name}")

    basename = os.path.basename(pseudo)

    computer = pw_code.computer.label
    label, _ = os.path.splitext(basename)
    label = orm.Str(f"({protocol}-{criteria}-{cutoff_control} at {computer}) {label}")

    with open(pseudo, "rb") as stream:
        pseudo = UpfData(stream)

    _basic_inputs = {
        "pw_code": pw_code,
        "ph_code": ph_code,
        "protocol": orm.Str(protocol),
        "cutoff_control": orm.Str(cutoff_control),
        "criteria": orm.Str(criteria),
        "options": orm.Dict(
            dict={
                "resources": {
                    "num_machines": 1,
                    "num_mpiprocs_per_machine": num_mpiprocs,
                },
                "max_wallclock_seconds": walltime,
                "withmpi": withmpi,
            }
        ),
        "parallization": orm.Dict(dict={"npool": npool}),
        "properties_list": orm.List(list=properties_list),
    }

    _pseudo_inputs = {
        "pseudo": pseudo,
        "label": label,
        "extra_desc": extra_desc,
    }

    node = run_verification(
        **_basic_inputs,
        **_pseudo_inputs,
        clean_workchain=clean_workchain,
        on_daemon=daemon,
    )

    click.echo(node)
    click.echo(f"calculated on property: {'/'.join(properties_list)}")


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
    on_daemon,
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

    if on_daemon:
        node = submit(VerificationWorkChain, **inputs)
    else:
        _, node = run_get_node(VerificationWorkChain, **inputs)

    node.description = f"{label.value} ({extra_desc})"

    return node


if __name__ == "__main__":
    launch()
