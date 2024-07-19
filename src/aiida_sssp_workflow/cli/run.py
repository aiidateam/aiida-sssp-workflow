#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Running verification workchain
"""

from typing import List, Tuple
from pathlib import Path

import aiida
import click
from aiida import orm
from aiida.cmdline.params import options, types
from aiida.cmdline.utils import echo
from aiida.engine import ProcessBuilder, run_get_node, submit
from aiida.plugins import WorkflowFactory

from aiida_pseudo.data.pseudo.upf import UpfData
from aiida_sssp_workflow.cli import cmd_root
from aiida_sssp_workflow.workflows.verifications import (
    DEFAULT_CONVERGENCE_PROPERTIES_LIST,
    DEFAULT_MEASURE_PROPERTIES_LIST,
    DEFAULT_PROPERTIES_LIST,
)

VerificationWorkChain = WorkflowFactory("sssp_workflow.verification")


def guess_properties_list(property: list) -> Tuple[List[str], str]:
    # if the property is not specified, use the default list with all properties calculated.
    # otherwise, use the specified properties.
    if not property:
        properties_list = DEFAULT_PROPERTIES_LIST
        extra_desc = "All properties"
    elif len(property) == 1 and property[0] == "convergence":
        properties_list = DEFAULT_CONVERGENCE_PROPERTIES_LIST
        extra_desc = "Convergence"
    elif len(property) == 1 and property[0] == "measure":
        properties_list = DEFAULT_MEASURE_PROPERTIES_LIST
        extra_desc = "Measure"
    else:
        properties_list = list(property)
        extra_desc = f"{properties_list}"

    return properties_list, extra_desc


def guess_is_convergence(properties_list: list) -> bool:
    """Check if it is a convergence test"""

    return any([c for c in properties_list if c.startswith("convergence")])


def guess_is_full_convergence(properties_list: list) -> bool:
    """Check if all properties are run for convergence test"""

    return len([c for c in properties_list if c.startswith("convergence")]) == len(
        DEFAULT_CONVERGENCE_PROPERTIES_LIST
    )


def guess_is_measure(properties_list: list) -> bool:
    """Check if it is a measure test"""

    return any([c for c in properties_list if c.startswith("measure")])


def guess_is_ph(properties_list: list) -> bool:
    """Check if it has a measure test"""

    return any([c for c in properties_list if "phonon_frequencies" in c])


# Trigger the launch by running:
# aiida-sssp-workflow launch --property measure.precision --pw-code pw-7.0@localhost --ph-code ph-7.0@localhost --protocol test --cutoff-control test --withmpi True -- examples/_static/Si_ONCV_PBE-1.2.upf
@cmd_root.command("launch")
@click.argument("pseudo", type=click.Path(exists=True))
@options.OverridableOption(
    "--pw-code", "pw_code", type=types.CodeParamType(entry_point="quantumespresso.pw")
)(required=True)
@click.option(
    "--oxygen-pseudo",
    "oxygen_pseudo",
    type=click.Path(exists=True),
    help="Oxygen pseudo to use for oxides precision measure workflow.",
)
@click.option(
    "--oxygen-ecutwfc",
    "oxygen_ecutwfc",
    type=click.FLOAT,
    help="Oxygen ecutwfc to use for oxides precision measure workflow.",
)
@click.option(
    "--oxygen-ecutrho",
    "oxygen_ecutrho",
    type=click.FLOAT,
    help="Oxygen ecutrho to use for oxides precision measure workflow.",
)
@options.OverridableOption(
    "--ph-code", "ph_code", type=types.CodeParamType(entry_point="quantumespresso.ph")
)(required=False)
@click.option(
    "--property",
    multiple=True,
    default=[],
    help="Property to verify, can be: measure.precision, measure.bands, convergence ...",
)
@click.option(
    "protocol",
    "--protocol",
    default="standard",
    help="Protocol to use for the verification, (standard, quick, test).",
)
@click.option("npool", "--npool", default=1, help="Number of pool.")
@click.option("walltime", "--walltime", default=3600, help="Walltime.")
@click.option(
    "resources",
    "--resources",
    multiple=True,
    type=(str, str),
    help="key value pairs pass to resource setting.",
)
@click.option(
    "--clean-workdir/--no-clean-workdir",
    default=True,
    help="Clean up the remote folder of all calculation, turn this off when your want to see the remote for details.",
)
@click.option(
    "--daemon/--no-daemon",
    default=True,
    help="Launch the verfication to daemon by submit or run directly.",
)
@click.option(
    "comment",
    "--comment",
    help="Add a comment word as the prefix the workchain description.",
)
# ecutwfc and ecutrho are for measure workflows only
@click.option(
    "ecutwfc",
    "--ecutwfc",
    type=float,
    help="Cutoff energy for wavefunctions in Rydberg.",
)
@click.option(
    "ecutrho",
    "--ecutrho",
    type=float,
    help="Cutoff energy for charge density in Rydberg.",
)
# configuration is hard coded for convergence workflow, but here is an interface for experiment purpose
# when this is passed with convergence test, only one can be passed.
# When it is passed with measure test, can be multiple configurations.
@click.option(
    "configuration",
    "--configuration",
    multiple=True,
    default=[],
    help="Configuration of structure, can be: SC, FCC, BCC, Diamond, XO, XO2, XO3, X2O, X2O3, X2O5, GS, RE",
)
def launch(
    pw_code: orm.Code,
    ph_code: orm.Code,
    property: list,
    protocol: str,
    configuration: list,
    npool: int,
    walltime: int,
    resources,
    pseudo: Path,
    clean_workdir: bool,
    daemon: bool,
    comment: str,
):
    """Launch the verification workchain."""
    # raise error if the options are not provided
    properties_list, extra_desc = guess_properties_list(property)

    is_convergence = guess_is_convergence(properties_list)
    is_full_convergence = guess_is_full_convergence(properties_list)
    is_measure = guess_is_measure(properties_list)
    is_ph = guess_is_ph(properties_list)

    if is_ph and not ph_code:
        echo.echo_critical(
            "ph_code must be provided since we run on it for phonon frequencies."
        )

    if is_convergence and len(configuration) > 1:
        echo.echo_critical(
            "Only one configuration is allowed for convergence workflow."
        )

    if is_measure and not is_full_convergence:
        echo.echo_warning(
            "Full convergence tests are not run, so we use maximum cutoffs for transferability verification."
        )

    # Load the curent AiiDA profile and log to user
    _profile = aiida.load_profile()
    echo.echo_info(f"Current profile: {_profile.name}")

    # convert configuration to list
    configuration_list = list(configuration)

    if len(configuration_list) == 0:
        conf_label = "default"
    elif len(configuration_list) == 1:
        conf_label = configuration_list[0]
    else:
        conf_label = "/".join(configuration_list)

    resources = dict(resources)
    if "num_machines" not in resources:
        resources["num_machines"] = 1

    builder: ProcessBuilder = FullVerificationWorkChain.get_builder(
        pseudo=pseudo,
        protocol=protocol,
        properties_list=properties_list,
        configuration_list=configuration_list,
        clean_workdir=clean_workdir,
    )

    builder.metadata.label = (
        f"({protocol} at {pw_code.computer.label} - {conf_label}) {pseudo.stem}"
    )
    builder.metadata.description = f"""Calculation is run on protocol: {protocol}; on {pw_code.computer.label}; on configuration {conf_label}; on pseudo {pseudo.stem}."""

    builder.pw_code = pw_code
    if is_ph:
        builder.ph_code = ph_code

    inputs = {
        "measure": {
            "protocol": orm.Str(protocol),
            "wavefunction_cutoff": orm.Float(ecutwfc),
            "charge_density_cutoff": orm.Float(ecutrho),
        },
        "convergence": {
            "protocol": orm.Str(protocol),
            "cutoff_control": orm.Str(cutoff_control),
            "criteria": orm.Str(criteria),
        },
        "pw_code": pw_code,
        "pseudo": pseudo,
        "label": label,
        "properties_list": orm.List(properties_list),
        "options": orm.Dict(
            dict={
                "resources": resources,
                "max_wallclock_seconds": walltime,
                "withmpi": withmpi,
            },
        ),
        "parallelization": orm.Dict(dict={"npool": npool}),
        "clean_workdir": orm.Bool(clean_workdir),
    }

    if is_ph:
        inputs["ph_code"] = ph_code

    if pw_code_large_memory:
        inputs["pw_code_large_memory"] = pw_code_large_memory

    if oxygen_pseudo:
        if not (oxygen_ecutwfc and oxygen_ecutrho):
            echo.echo_critical(
                "oxygen_ecutwfc and oxygen_ecutrho must be provided if using custmized oxygen pseudo."
            )

        with open(oxygen_pseudo, "rb") as stream:
            oxygen_pseudo = UpfData(stream)

        inputs["measure"]["oxygen_pseudo"] = oxygen_pseudo
        inputs["measure"]["oxygen_ecutwfc"] = orm.Float(oxygen_ecutwfc)
        inputs["measure"]["oxygen_ecutrho"] = orm.Float(oxygen_ecutrho)

    if len(configuration) == 0:
        pass
    elif len(configuration) == 1:
        inputs["convergence"]["configuration"] = orm.Str(configuration[0])
        inputs["measure"]["configurations"] = orm.List(list=configuration)
    else:
        inputs["measure"]["configurations"] = orm.List(list=configuration)

    if daemon:
        node = submit(VerificationWorkChain, **inputs)
    else:
        _, node = run_get_node(VerificationWorkChain, **inputs)

    description = f"{label.value} ({extra_desc})"
    node.description = f"({comment}) {description}" if comment else description

    echo.echo_info(node)
    echo.echo_success(f"calculated on property: {'/'.join(properties_list)}")


if __name__ == "__main__":
    launch()
