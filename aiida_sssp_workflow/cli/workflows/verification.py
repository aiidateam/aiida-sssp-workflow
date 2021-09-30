# -*- coding: utf-8 -*-
"""
Module contain the launch cmdline method of verification workflow
"""
import os
import click
import numpy as np

from aiida.plugins import WorkflowFactory
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

REF_ECUTWFC = 200
ecutwfc_list = np.array(
    [30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 90, 100, 120, 150, 200])


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
@decorators.with_dbenv()
def launch_workflow(pw_code, ph_code, pseudo, protocol, dual, clean_workdir,
                    max_num_machines, max_wallclock_seconds, with_mpi, daemon):
    """Run the workflow to calculate delta factor"""
    from aiida_sssp_workflow.utils import get_default_options

    builder = WorkflowFactory('sssp_workflow.verification').get_builder()

    pseudo_abspath = os.path.abspath(pseudo)
    pseudo = orm.UpfData.get_or_create(pseudo_abspath)[0]
    builder.pseudo = pseudo

    builder.pw_code = pw_code
    builder.ph_code = ph_code
    builder.protocol = orm.Str(protocol)

    ecutrho_list = ecutwfc_list * dual
    builder.parameters.ecutwfc_list = orm.List(list=list(ecutwfc_list))
    builder.parameters.ecutrho_list = orm.List(list=list(ecutrho_list))
    builder.parameters.ref_cutoff_pair = orm.List(
        list=[REF_ECUTWFC, REF_ECUTWFC * dual])

    metadata_options = get_default_options(max_num_machines,
                                           max_wallclock_seconds, with_mpi)
    builder.options = orm.Dict(dict=metadata_options)
    builder.clean_workdir = orm.Bool(clean_workdir)

    launch.launch_process(builder, daemon)
