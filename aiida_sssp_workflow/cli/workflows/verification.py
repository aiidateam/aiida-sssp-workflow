"""
Module contain the launch cmdline method of verification workflow
"""
from aiida.cmdline.utils import decorators

from . import cmd_launch


@cmd_launch.command('verification')
@decorators.with_dbenv()
def launch_workflow():
    pass
