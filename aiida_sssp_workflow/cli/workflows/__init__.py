# -*- coding: utf-8 -*-
# pylint: disable=cyclic-import,reimported,unused-import,wrong-import-position
"""Module with CLI commands for the various work chain implementations."""
from .. import cmd_root


@cmd_root.group('workflow')
def cmd_workflow():
    """Commands to launch and interact with workflows."""


@cmd_workflow.group('launch')
def cmd_launch():
    """Launch workflows."""


from .delta_factor import launch_workflow
from .verification import launch_workflow
