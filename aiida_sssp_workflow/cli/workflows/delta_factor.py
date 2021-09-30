# -*- coding: utf-8 -*-
"""
Module contain the launch cmdline method of delta-factor workflow
"""
import os
import click

from aiida.plugins import WorkflowFactory, DataFactory
from aiida import orm
from aiida.cmdline.utils import decorators
from aiida.cmdline.params import options as options_core
from aiida.cmdline.params import types
from aiida.cmdline.params.types import PathOrUrl

from . import cmd_launch
from .. import options
from .. import launch


@cmd_launch.command('delta-factor')
@click.argument('pseudo', nargs=1, type=PathOrUrl(exists=True, readable=True))
@options_core.CODE(required=True,
                   type=types.CodeParamType(entry_point='quantumespresso.pw'))
@options.PROTOCOL()
@options.CLEAN_WORKDIR()
@options.MAX_NUM_MACHINES()
@options.MAX_WALLCLOCK_SECONDS()
@options.WITH_MPI()
@options.DAEMON()
@decorators.with_dbenv()
def launch_workflow(code, pseudo, protocol, clean_workdir, max_num_machines,
                    max_wallclock_seconds, with_mpi, daemon):
    """Run the workflow to calculate delta factor"""
    from aiida_sssp_workflow.utils import get_default_options
    UpfData = DataFactory('pseudo.upf')

    builder = WorkflowFactory('sssp_workflow.delta_factor').get_builder()

    pseudo_abspath = os.path.abspath(pseudo)
    with open(pseudo_abspath, 'rb') as stream:
        pseudo = UpfData(stream)

    builder.pseudo = pseudo

    builder.code = code
    builder.protocol = orm.Str(protocol)

    metadata_options = get_default_options(max_num_machines,
                                           max_wallclock_seconds, with_mpi)
    builder.options = orm.Dict(dict=metadata_options)
    builder.clean_workdir = orm.Bool(clean_workdir)

    launch.launch_process(builder, daemon)
