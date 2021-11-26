# -*- coding: utf-8 -*-
"""
WorkChain calculate phonon frequencies at Gamma
"""
from aiida import orm
from aiida.engine import WorkChain, ToContext
from aiida.common import NotExistentAttributeError
from aiida.plugins import WorkflowFactory, DataFactory

from aiida_sssp_workflow.utils import update_dict

PwBaseWorkflow = WorkflowFactory('quantumespresso.pw.base')
PhBaseWorkflow = WorkflowFactory('quantumespresso.ph.base')
UpfData = DataFactory('pseudo.upf')


class PhononFrequenciesWorkChain(WorkChain):
    """WorkChain to calculate cohisive energy of input structure"""
    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        # yapf: disable
        super().define(spec)
        spec.input('pw_code', valid_type=orm.Code,
                    help='The `pw.x` code use for the `PwCalculation`.')
        spec.input('ph_code', valid_type=orm.Code,
                    help='The `ph.x` code use for the `PwCalculation`.')
        spec.input_namespace('pseudos', valid_type=UpfData, dynamic=True,
                    help='A mapping of `UpfData` nodes onto the kind name to which they should apply.')
        spec.input('structure', valid_type=orm.StructureData, required=True,
                    help='Ground state structure which the verification perform')
        spec.input('pw_base_parameters', valid_type=orm.Dict,
                    help='parameters for pw.x.')
        spec.input('ph_base_parameters', valid_type=orm.Dict,
                    help='parameters for ph.x.')
        spec.input('ecutwfc', valid_type=(orm.Float, orm.Int),
                    help='The ecutwfc set for both atom and bulk calculation. Please also set ecutrho if ecutwfc is set.')
        spec.input('ecutrho', valid_type=(orm.Float, orm.Int),
                    help='The ecutrho set for both atom and bulk calculation.  Please also set ecutwfc if ecutrho is set.')
        spec.input('qpoints', valid_type=orm.List,
                    help='qpoints for ph calculation.')
        spec.input('kpoints_distance', valid_type=orm.Float,
                    help='Kpoints distance setting for bulk energy calculation in pw calculation.')
        spec.input('options', valid_type=orm.Dict, required=False,
                    help='Optional `options` to use for the `PwCalculations`.')
        spec.input('parallelization', valid_type=orm.Dict, required=False,
                    help='Parallelization options for the `PwCalculations`.')
        spec.input('clean_workdir', valid_type=orm.Bool, default=lambda: orm.Bool(False),
                    help='If `True`, work directories of all called calculation will be cleaned at the end of execution.')
        spec.outline(
            cls.setup_base_parameters,
            cls.validate_structure,
            cls.setup_code_resource_options,
            cls.run_scf,
            cls.inspect_scf,
            cls.run_ph,
            cls.inspect_ph,
            cls.results,
        )
        spec.output('output_parameters', valid_type=orm.Dict, required=True,
                    help='The output parameters include phonon frequencies.')
        spec.exit_code(201, 'ERROR_SUB_PROCESS_FAILED_SCF',
                    message='The `PwBaseWorkChain` sub process failed.')
        spec.exit_code(211, 'ERROR_NO_REMOTE_FOLDER',
                    message='The remote folder node not exist')
        spec.exit_code(202, 'ERROR_SUB_PROCESS_FAILED_PH',
                    message='The `PhBaseWorkChain` sub process failed.')
        # yapf: enable

    def setup_base_parameters(self):
        """Input validation"""
        pw_parameters = self.inputs.pw_base_parameters.get_dict()

        parameters = {
            'SYSTEM': {
                'ecutwfc': self.inputs.ecutwfc,
                'ecutrho': self.inputs.ecutrho,
            },
        }
        pw_parameters = update_dict(pw_parameters, parameters)

        self.ctx.pw_parameters = pw_parameters
        self.ctx.kpoints_distance = self.inputs.kpoints_distance

        # ph.x
        self.ctx.ph_parameters = self.inputs.ph_base_parameters.get_dict()

        qpoints = orm.KpointsData()
        qpoints.set_cell_from_structure(self.inputs.structure)
        qpoints.set_kpoints(self.inputs.qpoints.get_list())
        self.ctx.qpoints = qpoints

    def validate_structure(self):
        """validate structure and set pseudos"""
        self.ctx.pseudos = self.inputs.pseudos

    def setup_code_resource_options(self):
        """
        setup resource options and parallelization for `PwCalculation` and
        `PhCalculation` from inputs
        """
        if 'options' in self.inputs:
            self.ctx.options = self.inputs.options.get_dict()
        else:
            from aiida_sssp_workflow.utils import get_default_options

            self.ctx.options = get_default_options(
                max_wallclock_seconds=self._MAX_WALLCLOCK_SECONDS,
                with_mpi=True)

        if 'parallelization' in self.inputs:
            self.ctx.parallelization = self.inputs.parallelization.get_dict()
        else:
            self.ctx.parallelization = {}

        self.report(f'resource options set to {self.ctx.options}')
        self.report(
            f'parallelization options set to {self.ctx.parallelization}')

    def run_scf(self):
        """
        set the inputs and submit scf
        """
        inputs = {
            'metadata': {
                'call_link_label': 'SCF'
            },
            'pw': {
                'structure': self.inputs.structure,
                'code': self.inputs.pw_code,
                'pseudos': self.ctx.pseudos,
                'parameters': orm.Dict(dict=self.ctx.pw_parameters),
                'metadata': {
                    'options': self.ctx.options,
                },
                'parallelization': orm.Dict(dict=self.ctx.parallelization),
            },
            'kpoints_distance': self.ctx.kpoints_distance,
        }

        running = self.submit(PwBaseWorkflow, **inputs)
        self.report(f'Running pw calculation pk={running.pk}')
        return ToContext(workchain_scf=running)

    def inspect_scf(self):
        """inspect the result of scf calculation if fail do not continue."""
        workchain = self.ctx.workchain_scf

        if not workchain.is_finished_ok:
            self.report(
                f'PwBaseWorkChain failed with exit status {workchain.exit_status}'
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_SCF

        try:
            self.ctx.scf_remote_folder = workchain.outputs.remote_folder
        except NotExistentAttributeError:
            return self.exit_codes.ERROR_NO_REMOTE_FOLDER

    def run_ph(self):
        """
        set the inputs and submit ph calculation to get quantities for phonon evaluation
        """
        # convert parallelization to CMDLINE for PH
        # since ph calculation now doesn't support parallelization
        cmdline_list = []
        for key, value in self.ctx.parallelization.items():
            cmdline_list.append(f'-{str(key)}')
            cmdline_list.append(str(value))

        inputs = {
            'metadata': {
                'call_link_label': 'PH'
            },
            'ph': {
                'code': self.inputs.ph_code,
                'qpoints': self.ctx.qpoints,
                'parameters': orm.Dict(dict=self.ctx.ph_parameters),
                'parent_folder': self.ctx.scf_remote_folder,
                'metadata': {
                    'options': self.ctx.options,
                },
                'settings': orm.Dict(dict={'CMDLINE': cmdline_list}),
            },
        }

        running = self.submit(PhBaseWorkflow, **inputs)
        self.report(f'Running ph calculation pk={running.pk}')
        return ToContext(workchain_ph=running)

    def inspect_ph(self):
        """inspect the result of ph calculation."""
        workchain = self.ctx.workchain_ph

        if not workchain.is_finished_ok:
            self.report(
                f'PhBaseWorkChain for pressure evaluation failed with exit status {workchain.exit_status}'
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_PH

        self.out('output_parameters', workchain.outputs.output_parameters)

    def results(self):
        """
        doc
        """
