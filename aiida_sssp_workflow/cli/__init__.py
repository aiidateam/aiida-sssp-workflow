# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position,wildcard-import
"""Module for the command line interface."""
import click
import click_completion
from aiida.cmdline.params import options as options_core
from aiida.cmdline.params import types

# Activate the completion of parameter types provided by the click_completion package
click_completion.init()


@click.group(
    "aiida-sssp-workflow", context_settings={"help_option_names": ["-h", "--help"]}
)
@options_core.PROFILE(type=types.ProfileParamType(load_profile=True))
def cmd_root(profile):  # pylint: disable=unused-argument
    """CLI for the `aiida-sssp-workflow` plugin."""


from aiida_sssp_workflow.cli.tools import dump_output
