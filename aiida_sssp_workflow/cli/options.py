# -*- coding: utf-8 -*-
"""Pre-defined overridable options for commonly used command line interface parameters."""
import click

from aiida.cmdline.params.options import OverridableOption

ECUTWFC = OverridableOption('-W',
                            '--ecutwfc',
                            default=200.0,
                            show_default=True,
                            type=click.FLOAT,
                            help='The plane wave cutoff energy in Ry.')

ECUTRHO = OverridableOption('-R',
                            '--ecutrho',
                            type=click.FLOAT,
                            help='The charge density cutoff energy in Ry.')

KPOINTS_DISTANCE = OverridableOption(
    '-K',
    '--kpoints-distance',
    type=click.FLOAT,
    default=0.1,
    show_default=True,
    help=
    'The minimal distance between k-points in reciprocal space in inverse Ångström.'
)

PROTOCOL = OverridableOption('-P',
                             '--protocol',
                             type=click.STRING,
                             default='efficiency',
                             show_default=True,
                             help='The protocol used in verification.')

DUAL = OverridableOption('-D',
                         '--dual',
                         type=click.INT,
                         default=8,
                         show_default=True,
                         help='The dual between ecutwfc and ecutrho.')

SCALE_COUNT = OverridableOption('-C',
                                '--scale-count',
                                type=click.INT,
                                default=7,
                                show_default=True,
                                help='Numbers of scale points in eos step.')

SCALE_INCREMENT = OverridableOption('-C',
                                    '--scale-increment',
                                    type=click.FLOAT,
                                    default=0.02,
                                    show_default=True,
                                    help='The scale increment in eos step.')

MAX_NUM_MACHINES = OverridableOption(
    '-m',
    '--max-num-machines',
    type=click.INT,
    default=1,
    show_default=True,
    help='The maximum number of machines (nodes) to use for the calculations.')

MAX_WALLCLOCK_SECONDS = OverridableOption(
    '-w',
    '--max-wallclock-seconds',
    type=click.INT,
    default=1800,
    show_default=True,
    help='the maximum wallclock time in seconds to set for the calculations.')

WITH_MPI = OverridableOption('-i',
                             '--with-mpi',
                             is_flag=True,
                             default=True,
                             show_default=True,
                             help='Run the calculations with MPI enabled.')

DAEMON = OverridableOption(
    '-d',
    '--daemon',
    is_flag=True,
    default=False,
    show_default=True,
    help='Submit the process to the daemon instead of running it locally.')

AUTOMATIC_PARALLELIZATION = OverridableOption(
    '-a',
    '--automatic-parallelization',
    is_flag=True,
    default=False,
    show_default=True,
    help='Enable the automatic parallelization option of the workchain.')

CLEAN_WORKDIR = OverridableOption(
    '-x',
    '--clean-workdir',
    is_flag=True,
    default=False,
    show_default=True,
    help=
    'Clean the remote folder of all the launched calculations after completion of the workchain.'
)

SMEARING = OverridableOption(
    '--smearing',
    nargs=2,
    default=('marzari-vanderbilt', 0.00735),
    show_default=True,
    type=click.Tuple([str, float]),
    help=
    'Add smeared occupations by specifying the type and amount of smearing.',
    metavar='<TYPE DEGAUSS>')
