# -*- coding: utf-8 -*-
"""
The base convergence WorkChain
"""
from abc import ABCMeta, abstractmethod
import typing as ty
import pathlib
import yaml

from aiida_tools.process_inputs import get_fullname

from aiida.engine import WorkChain, ToContext
from aiida import orm
from aiida.plugins import WorkflowFactory, DataFactory

from aiida_sssp_workflow.workflows.convergence.engine import TwoInputsTwoFactorsConvergence
from aiida_sssp_workflow.utils import helper_parse_upf
from aiida_sssp_workflow.helpers import get_pw_inputs_from_pseudo

CreateEvaluateWorkChain = WorkflowFactory('optimize.wrappers.create_evaluate')
OptimizationWorkChain = WorkflowFactory('optimize.optimize')
UpfData = DataFactory('pseudo.upf')

PARA_ECUTWFC_LIST = lambda: orm.List(list=[
    20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 110,
    120, 130, 140, 160, 180, 200
])

PARA_ECUTRHO_LIST = lambda: orm.List(list=[
    160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680, 720,
    760, 800, 880, 960, 1040, 1120, 1280, 1440, 1600
])


class BaseConvergenceWorkChain(WorkChain):
    """Meta WorkChain to run convergence test"""
    __metaclass__ = ABCMeta

    # hard code parameters of convergence workflow
    # can be(should be) overwrite by subclass
    _TOLERANCE = 1e-1
    _CONV_THR_CONV = 1e-1
    _CONV_WINDOW = 3

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        # yapf: disable
        super().define(spec)
        spec.input('pseudo', valid_type=UpfData, required=True,
                   help='Pseudopotential to be verified')
        spec.input('options', valid_type=orm.Dict, required=False,
                   help='Optional `options` to use for the `PwCalculations`.')
        spec.input('protocol', valid_type=orm.Str, default=lambda: orm.Str('efficiency'),
                   help='The protocol to use for the workchain.')
        spec.input_namespace('parameters', help='Para')
        spec.input('parameters.ecutrho_list', valid_type=orm.List, default=PARA_ECUTRHO_LIST,
                   help='dual value for ecutrho list.')
        spec.input('parameters.ecutwfc_list', valid_type=orm.List, default=PARA_ECUTWFC_LIST,
                   help='list of ecutwfc evaluate list.')
        spec.input('parameters.ref_cutoff_pair', valid_type=orm.List, required=True,
                   default=lambda: orm.List(list=[200, 1600]),
                   help='ecutwfc/ecutrho pair for reference calculation.')
        spec.outline(
            cls.setup_protocol,
            cls.init_step,
            cls.setup,
            cls.validate_structure,
            cls.run_ref,
            cls.run_all,
            cls.results,
            cls.final_step,
        )
        spec.output('output_parameters', valid_type=orm.Dict, required=True,
                    help='The output parameters include results of all calculations.')
        spec.output('xy_data_ecutwfc', valid_type=orm.XyData, required=True,
                    help='The output XY data for plot use; the x axis is ecutwfc.')
        spec.output('xy_data_ecutrho', valid_type=orm.XyData, required=True,
                    help='The output XY data for plot use; the x axis is ecutrho.')
        spec.output('output_convergence_parameters', valid_type=orm.Dict, required=False,
                    help='The result point of convergence test.')
        spec.output('output_pseudo_header', valid_type=orm.Dict, required=True,
                    help='The header(important parameters) of the pseudopotential.')
        spec.exit_code(400, 'ERROR_SUB_PROCESS_FAILED',
                    message='The sub processes {pk} did not finish successfully.')
        spec.exit_code(600, 'ERROR_NOT_ENOUGH_EVALUATE_WORKFLOW',
                    message='The number of sub evaluation processes {n} is not enough.')
        spec.exit_code(510, 'ERROR_DIFFERENT_SIZE_OF_ECUTOFF_PAIRS',
                    message='The ecutwfc_list and ecutrho_list have incompatible size.')
        # yapf: enable

    def _get_protocol(self):
        """Load and read protocol from faml file to a verbose dict"""
        with open(
                str(
                    pathlib.Path(__file__).resolve().parents[1] /
                    'protocol.yml')) as handle:
            self._protocol = yaml.safe_load(handle)  # pylint: disable=attribute-defined-outside-init

            return self._protocol

    @abstractmethod
    def setup_protocol(self):
        """
        For different convergence sub-workflow implement only needed parameters
        from protocol.
        """

    @abstractmethod
    def get_create_process(self):
        """
        the create process
        """

    @abstractmethod
    def get_evaluate_process(self):
        """
        the evaluate process
        """

    @abstractmethod
    def get_parsed_results(self) -> ty.Dict[str, str]:
        """
        Defined the return values will be recorded in output_parameters
        This method return a dict with keys as name of results and values as
        the unit of corresponding results.
        The results are read from the evaluate helper function defined in `get_evaluate_process`.
        """

    @abstractmethod
    def get_converge_y(self) -> ty.Tuple[str, ty.Tuple[str, str]]:
        """
        The name of value in the output of evaluate process, use in as the
        convergence value.
        The first element of tuple is the name of the value and the second is
        a tuple of (<description>, <unit>).
        """

    def init_step(self):
        """
        A empty initial step which customized by user
        """

    def final_step(self):
        """
        A empty final step which customized by user
        """

    def setup(self):
        """setup"""
        self.ctx.ecutwfc_list = self.inputs.parameters.ecutwfc_list.get_list()
        self.ctx.ecutrho_list = self.inputs.parameters.ecutrho_list.get_list()
        if not len(self.ctx.ecutwfc_list) == len(self.ctx.ecutrho_list):
            return self.exit_codes.ERROR_DIFFERENT_SIZE_OF_ECUTOFF_PAIRS

        # parse pseudo and output its header information
        upf_info = helper_parse_upf(self.inputs.pseudo)
        self.ctx.element = self.inputs.pseudo.element

        self.out('output_pseudo_header', orm.Dict(dict=upf_info).store())

    def validate_structure(self):
        """validate structure"""
        res = get_pw_inputs_from_pseudo(pseudo=self.inputs.pseudo)

        self.ctx.structure = res['structure']
        self.ctx.pseudos = res['pseudos']
        self.ctx.base_pw_parameters = res['base_pw_parameters']

    def run_ref(self):
        """
        Running the calculation for the reference point
        hard code to 200Ry at the moment
        """
        cutoff_pair = self.inputs.parameters.ref_cutoff_pair.get_list()
        ecutwfc = cutoff_pair[0]
        ecutrho = cutoff_pair[1]
        inputs = self.get_create_process_inputs()

        inputs['parameters']['ecutwfc'] = orm.Float(ecutwfc)
        inputs['parameters']['ecutrho'] = orm.Float(ecutrho)

        running = self.submit(self.get_create_process(), **inputs)

        self.report(
            f'launching reference CohesiveEnergyWorkChain<{running.pk}>.')

        return ToContext(ref_workchain=running)

    @abstractmethod
    def get_create_process_inputs(self) -> ty.Dict[str, ty.Any]:
        """
        The inputs used to running the create workflow
        Normally the corresponding property evaluation workflow.
        """

    @abstractmethod
    def get_evaluate_process_inputs(self) -> ty.Dict[str, ty.Any]:
        """
        The inputs used to running the evaluate workflow
        Including inputs not passed from the previous create workflow
        Normally the reference value of `run_ref` step.
        """

    @abstractmethod
    def get_output_input_mapping(self) -> orm.Dict:
        """
        return output_input_mapping from create workflow to
        evaluate workflow
        """

    def run_all(self):
        """
        Running the calculation for other points
        """
        ref_workchain = self.ctx.ref_workchain

        if not ref_workchain.is_finished_ok:
            self.report(
                f'Reference run of CohesiveEnergyWorkChain failed with exit status {ref_workchain.exit_status}'
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED.format(
                pk=ref_workchain.pk)

        create_evaluate_inputs = {
            'create_process': get_fullname(self.get_create_process()),
            'evaluate_process': get_fullname(self.get_evaluate_process()),
            'create': self.get_create_process_inputs(),
            'evaluate': self.get_evaluate_process_inputs(),
            'output_input_mapping': self.get_output_input_mapping(),
        }

        input_values = list(zip(self.ctx.ecutwfc_list, self.ctx.ecutrho_list))

        self.ctx.converge_y_name, self.ctx.converge_y_unit = self.get_converge_y(
        )
        inputs = {
            'engine':
            TwoInputsTwoFactorsConvergence,
            'engine_kwargs':
            orm.Dict(
                dict={
                    'input_values': input_values,
                    'tol': self._TOLERANCE,
                    'conv_thr': self._CONV_THR_CONV,
                    'input_key': 'create.parameters.ecutwfc',
                    'extra_input_key': 'create.parameters.ecutrho',
                    'result_key':
                    f'evaluate.result:{self.ctx.converge_y_name}',
                    'convergence_window': self._CONV_WINDOW
                }),
            'evaluate_process':
            CreateEvaluateWorkChain,
            'evaluate':
            create_evaluate_inputs,
        }

        running = self.submit(OptimizationWorkChain, **inputs)
        self.report(
            f'submitting cohesive energy convergence workflow pk={running.pk}.'
        )
        return ToContext(convergence_workchain=running)

    def results(self):
        """
        doc
        """
        workchain = self.ctx.convergence_workchain

        if workchain.is_finished_ok:
            self.report(f'Convergence workflow pk={workchain.pk} finish ok.')
            optimal_process_uuid = workchain.outputs.optimal_process_uuid.value
            optimal_process = orm.load_node(optimal_process_uuid)
            res = {
                'converge_ecutwfc':
                optimal_process.inputs.create__parameters__ecutwfc,
                'converge_ecutrho':
                optimal_process.inputs.create__parameters__ecutrho,
                'converge_value':
                workchain.outputs.optimal_process_output.value,
                'converge_process_uuid':
                workchain.outputs.optimal_process_uuid.value,
            }
            self.out('output_convergence_parameters',
                     orm.Dict(dict=res).store())

        if not workchain.is_finished:
            self.report(
                f'Reference run of Convergence optimize workchain pk={workchain.pk} '
                f'failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED.format(
                pk=workchain.pk)

        if workchain.is_finished and workchain.exit_status == 202:
            self.report(
                f'Convergence workflow pk={workchain.pk} is finished but not converge.'
            )

        import numpy as np

        self.ctx.children = [
            child for child in workchain.get_outgoing().all_nodes()
            if isinstance(child, orm.WorkChainNode)
        ]
        pks = [child.pk for child in self.ctx.children if child.is_finished_ok]
        if len(pks) / len(self.ctx.children) < 0.8:
            self.report(
                f'Not enough finish ok evaluate workflow, '
                f'expect 80% only get number of {len(pks)} workflow, '
                f'that is {len(pks) / len(self.ctx.children)} of total.')
            return self.exit_codes.ERROR_NOT_ENOUGH_EVALUATE_WORKFLOW.format(
                n=len(pks))

        success_child = [
            child for child in self.ctx.children if child.is_finished_ok
        ]
        ecutwfc_list = []
        ecutrho_list = []

        results_parser_dict = self.get_parsed_results()
        output_parameters_dict = {}
        for key in results_parser_dict:
            output_parameters_dict[key] = []

        for child in success_child:
            ecutwfc = child.inputs.create__parameters__ecutwfc.value
            ecutrho = child.inputs.create__parameters__ecutrho.value

            ecutwfc_list.append(ecutwfc)
            ecutrho_list.append(ecutrho)

            # loop over all output properties of evaluate process
            # set the results in the list of corresponding name.
            for key in output_parameters_dict:
                res = child.outputs.evaluate__result[key]
                output_parameters_dict[key].append(res)

        for key in list(output_parameters_dict):
            output_parameters_dict[f'{key}_description'] = results_parser_dict[
                key]

        output_parameters_dict['ecutwfc_list'] = ecutwfc_list
        output_parameters_dict['ecutrho_list'] = ecutrho_list

        xy_data_ecutwfc = orm.XyData()
        xy_data_ecutwfc.set_x(np.array(ecutwfc_list), 'wavefunction cutoff',
                              'Rydberg')

        xy_data_ecutrho = orm.XyData()
        xy_data_ecutrho.set_x(np.array(ecutrho_list), 'charge density cutoff',
                              'Rydberg')

        ys_list = []
        ys_description = []
        ys_unit = []
        for y_key, y_value in results_parser_dict.items():
            ys_list.append(np.array(output_parameters_dict[y_key]))
            ys_description.append(y_value[0])
            ys_unit.append(y_value[1])
        xy_data_ecutwfc.set_y(ys_list, ys_description, ys_unit)
        self.out('xy_data_ecutwfc', xy_data_ecutwfc.store())
        xy_data_ecutrho.set_y(ys_list, ys_description, ys_unit)
        self.out('xy_data_ecutrho', xy_data_ecutrho.store())

        output_parameters = orm.Dict(dict=output_parameters_dict)
        self.out('output_parameters', output_parameters.store())
