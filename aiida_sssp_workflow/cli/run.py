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
from aiida.cmdline.utils import echo
from aiida.engine import run_get_node, submit
from aiida.plugins import DataFactory, WorkflowFactory

from aiida_sssp_workflow.cli import cmd_root
from aiida_sssp_workflow.workflows.verifications import (
    DEFAULT_CONVERGENCE_PROPERTIES_LIST,
    DEFAULT_MEASURE_PROPERTIES_LIST,
    DEFAULT_PROPERTIES_LIST,
)

UpfData = DataFactory("pseudo.upf")
VerificationWorkChain = WorkflowFactory("sssp_workflow.verification")

# Trigger the launch by running:
# aiida-sssp-workflow launch --property measure.precision --pw-code pw-7.0@localhost --ph-code ph-7.0@localhost --protocol test --cutoff-control test --criteria efficiency --withmpi True -- examples/_static/Si_ONCV_PBE-1.2.upf


@cmd_root.command("launch")
@click.argument("pseudo", type=click.Path(exists=True))
@options.OverridableOption(
    "--pw-code", "pw_code", type=types.CodeParamType(entry_point="quantumespresso.pw")
)(required=True)
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
    default="acwf",
    help="Protocol to use for the verification, (acwf, test).",
)
@click.option("withmpi", "--withmpi", default=True, help="Run with mpi.")
@click.option("npool", "--npool", default=1, help="Number of pool.")
@click.option("walltime", "--walltime", default=3600, help="Walltime.")
@click.option("num_mpiprocs", "--num-mpiprocs", default=1, help="Number of mpiprocs.")
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
    help="Cutoff energy for wavefunctions in Rydberg.",
)
@click.option(
    "ecutrho",
    "--ecutrho",
    help="Cutoff energy for charge density in Rydberg.",
)
# cutoff_control, criteria, configuration is for convergence workflows only
@click.option(
    "cutoff_control",
    "--cutoff-control",
    help="Cutoff control for convergence workflow, (standard, quick, opsp).",
)
@click.option(
    "criteria", "--criteria", help="Criteria for convergence (efficiency, precision)."
)
# configuration is hard coded for convergence workflow, but here is an interface for experiment purpose
@click.option(
    "configuration",
    "--configuration",
    multiple=True,
    default=(),
    help="Configuration of structure, can be: SC, FCC, BCC, Diamond, XO, XO2, XO3, X2O, X2O3, X2O5, GS, RE",
)
def launch(
    pw_code,
    ph_code,
    property,
    protocol,
    ecutwfc,
    ecutrho,
    cutoff_control,
    criteria,
    configuration,
    withmpi,
    npool,
    walltime,
    num_mpiprocs,
    pseudo,
    clean_workdir,
    daemon,
    comment,
):
    """Launch the verification workchain."""
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

    # validate the options are all provide for the property
    is_convergence = False
    is_measure = False
    is_ph = False
    for prop in properties_list:
        if prop.startswith("convergence"):
            is_convergence = True
        if prop.startswith("measure"):
            is_measure = True
        if "phonon_frequencies" in prop:
            is_ph = True

    # raise error if the options are not provided
    if is_convergence and not (cutoff_control and criteria):
        echo.echo_critical(
            "cutoff_control, criteria must be provided for convergence workflow."
        )

    if is_measure and not (ecutwfc and ecutrho):
        echo.echo_critical("ecutwfc and ecutrho must be provided for measure workflow.")

    if is_ph and not ph_code:
        echo.echo_critical("ph_code must be provided for phonon frequencies.")

    # raise warning if the options are over provided, e.g. cutoff_control is provided for measure workflow
    if is_measure and (cutoff_control or criteria):
        echo.echo_warning("cutoff_control, criteria are not used for measure workflow.")

    if is_convergence and len(configuration) > 1:
        echo.echo_critical(
            "Only one configuration is allowed for convergence workflow."
        )

    if is_convergence and (ecutwfc or ecutrho):
        echo.echo_warning("ecutwfc and ecutrho are not used for convergence workflow.")

    _profile = aiida.load_profile()
    echo.echo_info(f"Current profile: {_profile.name}")

    basename = os.path.basename(pseudo)

    computer = pw_code.computer.label
    label, _ = os.path.splitext(basename)

    # convert configuration to list
    configuration = list(configuration)

    if len(configuration) == 0:
        conf_label = "default"
    elif len(configuration) == 1:
        conf_label = configuration[0]
    else:
        conf_label = "/".join(configuration)

    pre_label = (
        f"{protocol}"
        if not is_convergence
        else f"{protocol}-{criteria}-{cutoff_control}"
    )
    label = orm.Str(f"({pre_label} at {computer} - {conf_label}) {label}")

    with open(pseudo, "rb") as stream:
        pseudo = UpfData(stream)

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
                "resources": {
                    "num_machines": 1,
                    "num_mpiprocs_per_machine": num_mpiprocs,
                },
                "max_wallclock_seconds": walltime,
                "withmpi": withmpi,
            }
        ),
        "parallelization": orm.Dict(dict={"npool": npool}),
        "clean_workdir": orm.Bool(clean_workdir),
    }

    if is_ph:
        inputs["ph_code"] = ph_code

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
