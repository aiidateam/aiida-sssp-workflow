"""
Module contain the launch cmdline method of delta-factor workflow
"""
import os
import click

from aiida.plugins import WorkflowFactory
from aiida import orm
from aiida.cmdline.utils import decorators
from aiida.cmdline.params import options as options_core
from aiida.cmdline.params import types
from aiida.cmdline.params.types import PathOrUrl

from . import cmd_launch
from .. import options
from .. import launch
from .. import validate


@cmd_launch.command('delta-factor')
@click.argument('pseudo', nargs=1, type=PathOrUrl(exists=True, readable=True))
@options_core.CODE(required=True,
                   type=types.CodeParamType(entry_point='quantumespresso.pw'))
@options.SMEARING()
@options.KPOINTS_DISTANCE()
@options.ECUTWFC()
@options.ECUTRHO()
@options.SCALE_COUNT()
@options.SCALE_INCREMENT()
@options.CLEAN_WORKDIR()
@options.MAX_NUM_MACHINES()
@options.MAX_WALLCLOCK_SECONDS()
@options.WITH_MPI()
@options.DAEMON()
@decorators.with_dbenv()
def launch_workflow(code, pseudo, smearing, kpoints_distance, scale_count,
                    scale_increment, ecutwfc, ecutrho, clean_workdir,
                    max_num_machines, max_wallclock_seconds, with_mpi, daemon):
    """Run the workflow to calculate delta factor"""
    from aiida_quantumespresso.utils.resources import get_default_options

    parameters = {
        'SYSTEM': {},
    }
    try:
        validate.validate_smearing(parameters, smearing)
    except ValueError as exception:
        raise click.BadParameter(str(exception))

    builder = WorkflowFactory('sssp_workflow.delta_factor').get_builder()

    pseudo_abspath = os.path.abspath(pseudo)
    pseudo = orm.UpfData.get_or_create(pseudo_abspath)[0]
    builder.pseudo = pseudo

    builder.code = code
    builder.parameters.pw = orm.Dict(dict=parameters)

    builder.parameters.ecutwfc = orm.Float(ecutwfc)
    if ecutrho:
        builder.parameters.ecutrho = orm.Float(ecutrho)

    builder.parameters.kpoints_distance = orm.Float(kpoints_distance)
    builder.parameters.scale_count = orm.Int(scale_count)
    builder.parameters.scale_increment = orm.Float(scale_increment)

    metadata_options = get_default_options(max_num_machines,
                                           max_wallclock_seconds, with_mpi)
    builder.options = orm.Dict(dict=metadata_options)

    builder.clean_workdir = orm.Bool(clean_workdir)

    launch.launch_process(builder, daemon)
