# -*- coding: utf-8 -*-
"""fixtures"""
import io
import pytest
import os
import shutil

from collections.abc import Mapping


pytest_plugins = ['aiida.manage.tests.pytest_fixtures']  # pylint: disable=invalid-name

@pytest.fixture(scope='function')
def fixture_sandbox():
    """Return a `SandboxFolder`."""
    from aiida.common.folders import SandboxFolder
    with SandboxFolder() as folder:
        yield folder
        
@pytest.fixture
def fixture_localhost(aiida_localhost):
    """Return a localhost `Computer`."""
    localhost = aiida_localhost
    localhost.set_default_mpiprocs_per_machine(1)
    return localhost
        
@pytest.fixture
def fixture_code(fixture_localhost):
    """Return a ``Code`` instance configured to run calculations of given entry point on localhost ``Computer``."""

    def _fixture_code(entry_point_name):
        from aiida.common import exceptions
        from aiida.orm import Code

        label = f'test.{entry_point_name}'

        try:
            return Code.objects.get(label=label)  # pylint: disable=no-member
        except exceptions.NotExistent:
            return Code(
                label=label,
                input_plugin_name=entry_point_name,
                remote_computer_exec=[fixture_localhost, '/bin/true'],
            )

    return _fixture_code

@pytest.fixture(scope='session')
def generate_parser():
    """Fixture to load a parser class for testing parsers."""

    def _generate_parser(entry_point_name):
        """Fixture to load a parser class for testing parsers.
        :param entry_point_name: entry point name of the parser class
        :return: the `Parser` sub class
        """
        from aiida.plugins import ParserFactory
        return ParserFactory(entry_point_name)

    return _generate_parser

@pytest.fixture
def generate_inputs_ld1(fixture_code):
    """Generate default inputs for a `OncvPseudoCalculation."""

    def _generate_inputs_ld1(filename="pseudo.upf", inp_str=None):
        """Generate default inputs for a `Ld1Calculation."""
        from aiida import orm

        inp = inp_str or f""" &input
   zed=3,
   rel=0,
   config='default',
   iswitch=3,
   dft='PBE'
 /
 &inputp
   lpaw=.false.,
   pseudotype=1,
   file_pseudopw='{filename}',
   author='anonymous',
   lloc=-1,
   rcloc=0.1
   which_augfun='PSQ',
   rmatch_augfun_nc=.true.,
   tm=.true.
 /
4
1S  1  0  2.00  0.00  0.80  1.00  0.0
2S  2  0  1.00  0.00  0.80  1.00  0.0
2P  2  1  0.00  0.00  0.85  1.20  0.0
2P  2  1  0.00  1.00  0.85  1.20  0.0"""

        parameters =  orm.SinglefileData(file=io.BytesIO(inp.encode('utf-8')))
        inputs = {
            'code': fixture_code('sssp.pseudo.ld1'),
            'filename': orm.Str(f"{filename}"),
            'parameters': parameters,
            'metadata': {
                'options': {
                    'resources': {
                        'num_machines': int(1)
                    },
                    'max_wallclock_seconds': int(60),
                    'withmpi': False,
                }
            }
        }

        return inputs

    return _generate_inputs_ld1

@pytest.fixture
def generate_calc_job():
    """Fixture to construct a new `CalcJob` instance and call `prepare_for_submission` for testing `CalcJob` classes.
    The fixture will return the `CalcInfo` returned by `prepare_for_submission` and the temporary folder that was passed
    to it, into which the raw input files will have been written.
    """

    def _generate_calc_job(folder, entry_point_name, inputs=None):
        """Fixture to generate a mock `CalcInfo` for testing calculation jobs."""
        from aiida.engine.utils import instantiate_process
        from aiida.manage.manager import get_manager
        from aiida.plugins import CalculationFactory

        manager = get_manager()
        runner = manager.get_runner()

        process_class = CalculationFactory(entry_point_name)
        process = instantiate_process(runner, process_class, **inputs)

        calc_info = process.prepare_for_submission(folder)

        return calc_info

    return _generate_calc_job


@pytest.fixture
def generate_calc_job_node(fixture_localhost):
    """Fixture to generate a mock `CalcJobNode` for testing parsers."""

    def flatten_inputs(inputs, prefix=''):
        """Flatten inputs recursively like :meth:`aiida.engine.processes.process::Process._flatten_inputs`."""
        flat_inputs = []
        for key, value in inputs.items():
            if isinstance(value, Mapping):
                flat_inputs.extend(flatten_inputs(value, prefix=prefix + key + '__'))
            else:
                flat_inputs.append((prefix + key, value))
        return flat_inputs

    def _generate_calc_job_node(
        entry_point_name='opsp.pseudo.oncv', computer=None, test_name=None, inputs=None, attributes=None
    ):
        """Fixture to generate a mock `CalcJobNode` for testing parsers.
        :param entry_point_name: entry point name of the calculation class
        :param computer: a `Computer` instance
        :param test_name: relative path of directory with test output files in the `fixtures/{entry_point_name}` folder.
        :param inputs: any optional nodes to add as input links to the corrent CalcJobNode
        :param attributes: any optional attributes to set on the node

        :return: `CalcJobNode` instance with an attached `FolderData` as the `retrieved` node.
        """
        from aiida import orm
        from aiida.common import LinkType
        from aiida.plugins.entry_point import format_entry_point_string

        if computer is None:
            computer = fixture_localhost

        filepath_folder = None

        if test_name is not None:
            basepath = os.path.dirname(os.path.abspath(__file__))
            filename = os.path.join(entry_point_name[len('opsp.pseudo.'):], test_name)
            filepath_folder = os.path.join(basepath, 'parsers', 'fixtures', filename)

        entry_point = format_entry_point_string('aiida.calculations', entry_point_name)

        node = orm.CalcJobNode(computer=computer, process_type=entry_point)
        node.set_attribute('input_filename', 'aiida.in')
        node.set_attribute('output_filename', 'aiida.out')
        node.set_attribute('error_filename', 'aiida.err')
        node.set_option('resources', {'num_machines': 1, 'num_mpiprocs_per_machine': 1})
        node.set_option('max_wallclock_seconds', 1800)

        if attributes:
            node.set_attribute_many(attributes)

        if inputs:
            metadata = inputs.pop('metadata', {})
            options = metadata.get('options', {})

            for name, option in options.items():
                node.set_option(name, option)

            for link_label, input_node in flatten_inputs(inputs):
                input_node.store()
                node.add_incoming(input_node, link_type=LinkType.INPUT_CALC, link_label=link_label)

        node.store()

        if filepath_folder:
            retrieved = orm.FolderData()
            retrieved.put_object_from_tree(filepath_folder)

            retrieved.add_incoming(node, link_type=LinkType.CREATE, link_label='retrieved')
            retrieved.store()

            remote_folder = orm.RemoteData(computer=computer, remote_path='/tmp')
            remote_folder.add_incoming(node, link_type=LinkType.CREATE, link_label='remote_folder')
            remote_folder.store()

        return node

    return _generate_calc_job_node
