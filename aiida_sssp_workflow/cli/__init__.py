# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position,wildcard-import
"""Module for the command line interface."""
import click
from aiida.cmdline.groups import VerdiCommandGroup
from aiida.cmdline.params import options, types


@click.group(
    "aiida-sssp-workflow",
    cls=VerdiCommandGroup,
    context_settings={"help_option_names": ["-h", "--help"]},
)
@options.PROFILE(type=types.ProfileParamType(load_profile=True), expose_value=False)
def cmd_root():
    """CLI for the `aiida-sssp-workflow` plugin."""


from .extract import extract
from .inspect import inspect
from .run import launch
