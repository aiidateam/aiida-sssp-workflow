# -*- coding: utf-8 -*-
"""
WorkChain calculate phonon frequencies at Gamma
"""
from aiida import orm
from aiida.engine import calcfunction, WorkChain, ToContext
from aiida.common import AttributeDict, NotExistentAttributeError
from aiida.plugins import WorkflowFactory

PwBaseWorkflow = WorkflowFactory('quantumespresso.pw.base')
PhBaseWorkflow = WorkflowFactory('quantumespresso.ph.base')

@calcfunction
def helper_parse_upf(upf):
    return orm.Str(upf.element)

PW_PARAS = lambda: orm.Dict(
    dict={
        'SYSTEM': {
            'degauss': 0.02,
            'ecutrho': 800,
            'ecutwfc': 200,
            'occupations': 'smearing',
            'smearing': 'marzari-vanderbilt',
        },
        'ELECTRONS': {
            'conv_thr': 1e-10,
        },
    })


class PhononFrequenciesWorkChain(WorkChain):
    """WorkChain to calculate cohisive energy of input structure"""
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input('pw_code',
                   valid_type=orm.Code,
                   help='The `pw.x` code use for the `PwCalculation`.')
        spec.input('ph_code',
                   valid_type=orm.Code,
                   help='The `ph.x` code use for the `PwCalculation`.')
        spec.input('pseudo',
                   valid_type=orm.UpfData,
                   required=True,
                   help='Pseudopotential to be verified')
        spec.input(
            'structure',
            valid_type=orm.StructureData,
            required=True,
            help='Ground state structure which the verification perform')
        spec.input('options',
                   valid_type=orm.Dict,
                   required=False,
                   help='Optional `options` to use for the `PwCalculations`.')
        spec.input_namespace('parameters', help='Para')
        spec.input('parameters.pw',
                   valid_type=orm.Dict,
                   default=PW_PARAS,
                   help='parameters for pw.x.')
        spec.input('parameters.ph',
                   valid_type=orm.Dict,
                   default=PW_PARAS,
                   help='parameters for ph.x.')
        spec.input('parameters.qpoints', valid_type=orm.List, default=lambda: orm.List(list=[[0.,0.,0.]]),
                   help='qpoints')
        spec.input(
            'parameters.kpoints_distance',
            valid_type=orm.Float,
            default=lambda: orm.Float(0.15),
            help='Kpoints distance setting for bulk energy calculation.')
        spec.outline(
            cls.setup,
            cls.validate_structure,
            cls.run_scf,
            cls.inspect_scf,
            cls.run_ph,
            cls.inspect_ph,
            cls.results,
        )
        spec.output(
            'output_parameters',
            valid_type=orm.Dict,
            required=True,
            help=
            'The output parameters include phonon frequencies.')
        spec.exit_code(
            201,
            'ERROR_SUB_PROCESS_FAILED_SCF',
            message='The `PwBaseWorkChain` sub process failed.')
        spec.exit_code(
            211,
            'ERROR_NO_REMOTE_FOLDER',
            message='The remote folder node not exist')

    def setup(self):
        """Input validation"""
        # TODO set ecutwfc and ecutrho according to certain protocol
        self.ctx.pw_parameters = orm.Dict(dict={
            'CONTROL': {
                'calculation': 'scf',
                'wf_collect': True,
            },
        })
        self.ctx.pw_parameters.update_dict(self.inputs.parameters.pw.get_dict())
        self.ctx.kpoints_distance = self.inputs.parameters.kpoints_distance

        # ph.x
        self.ctx.ph_parameters = orm.Dict(dict={
            'INPUTPH': {
                'tr2_ph': 1e-16,
                'epsil': False,
            }
        })
        self.ctx.ph_parameters.update_dict(self.inputs.parameters.ph.get_dict())
        qpoints = orm.KpointsData()
        qpoints.set_cell_from_structure(self.inputs.structure)
        qpoints.set_kpoints(self.inputs.parameters.qpoints.get_list())
        self.ctx.qpoints = qpoints


    def validate_structure(self):
        """Create isolate atom and validate structure"""
        # create isolate atom structure
        self.ctx.element = helper_parse_upf(self.inputs.pseudo)
        self.ctx.pseudos = {self.ctx.element.value: self.inputs.pseudo}

    def run_scf(self):
        """
        set the inputs and submit scf
        """
        inputs = AttributeDict({
            'metadata': {
                'call_link_label': 'scf'
            },
            'pw': {
                'structure': self.inputs.structure,
                'code': self.inputs.pw_code,
                'pseudos': self.ctx.pseudos,
                'parameters': self.ctx.pw_parameters,
                'settings': orm.Dict(dict={'CMDLINE': ['-ndiag', '1', '-nk', '2']}),
                'metadata': {},
            },
            'kpoints_distance':
            self.ctx.kpoints_distance,
        })

        if 'options' in self.inputs:
            options = self.inputs.options.get_dict()
        else:
            from aiida_quantumespresso.utils.resources import get_default_options

            options = get_default_options(with_mpi=True)

        inputs.pw.metadata.options = options

        running = self.submit(PwBaseWorkflow, **inputs)
        self.report(f'Running pw calculation pk={running.pk}')
        return ToContext(workchain_scf=running)

    def inspect_scf(self):
        """inspect the result of scf calculation."""
        workchain = self.ctx.workchain_scf

        if not workchain.is_finished_ok:
            self.report(
                f'PwBaseWorkChain for pressure evaluation failed with exit status {workchain.exit_status}'
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
        inputs = AttributeDict({
            'metadata': {
                'call_link_label': 'ph'
            },
            'ph': {
                'code': self.inputs.ph_code,
                'qpoints': self.ctx.qpoints,
                'parameters': self.ctx.ph_parameters,
                'parent_folder': self.ctx.scf_remote_folder,
                'settings': orm.Dict(dict={'CMDLINE': ['-nk', '2']}),
                'metadata': {},
            },
        })
        if 'options' in self.inputs:
            options = self.inputs.options.get_dict()
        else:
            from aiida_quantumespresso.utils.resources import get_default_options

            options = get_default_options(with_mpi=True)

        inputs.ph.metadata.options = options

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
        pass
