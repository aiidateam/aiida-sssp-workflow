# -*- coding: utf-8 -*-
"""
Module contain the launch cmdline method of verification workflow
"""
import os
import click

from aiida.plugins import WorkflowFactory, DataFactory
from aiida import orm
from aiida.cmdline.utils import decorators
from aiida.cmdline.params.types import PathOrUrl
from aiida.cmdline.params import types
from aiida.cmdline.params.options import OverridableOption

from . import cmd_launch
from .. import launch
from .. import options

PW_CODE = OverridableOption(
    '-X',
    '--pw-code',
    'pw_code',
    type=types.CodeParamType(entry_point='quantumespresso.pw'),
    help='A single code identified by its ID, UUID or label.')

PH_CODE = OverridableOption(
    '-Y',
    '--ph-code',
    'ph_code',
    type=types.CodeParamType(entry_point='quantumespresso.ph'),
    help='A single code identified by its ID, UUID or label.')


@cmd_launch.command('verification')
@click.argument('pseudo', nargs=1, type=PathOrUrl(exists=True, readable=True))
@PW_CODE(required=True)
@PH_CODE(required=True)
@options.PROTOCOL()
@options.DUAL()
@options.CLEAN_WORKDIR()
@options.MAX_NUM_MACHINES()
@options.MAX_WALLCLOCK_SECONDS()
@options.WITH_MPI()
@options.DAEMON()
@options.DESCRIPTION()
@decorators.with_dbenv()
def launch_workflow(pw_code, ph_code, pseudo, protocol, dual, clean_workdir,
                    max_num_machines, max_wallclock_seconds, with_mpi, daemon,
                    description):
    """Run the workflow to verification"""
    from aiida_sssp_workflow.utils import get_default_options
    UpfData = DataFactory('pseudo.upf')

    builder = WorkflowFactory('sssp_workflow.verification').get_builder()

    pseudo_abspath = os.path.abspath(pseudo)
    with open(pseudo_abspath, 'rb') as stream:
        pseudo = UpfData(stream)

    builder.pseudo = pseudo

    builder.pw_code = pw_code
    builder.ph_code = ph_code
    builder.dual = orm.Float(dual)
    builder.protocol = orm.Str(protocol)

    metadata_options = get_default_options(max_num_machines,
                                           max_wallclock_seconds, with_mpi)
    builder.options = orm.Dict(dict=metadata_options)
    builder.clean_workdir = orm.Bool(clean_workdir)

    launch.launch_process(builder, daemon, description)
