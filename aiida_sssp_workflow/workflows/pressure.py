# -*- coding: utf-8 -*-
"""
A calcfunctian create_isolate_atom
Create the structure of isolate atom
"""
from aiida import orm
from aiida.engine import calcfunction, WorkChain, ToContext
from aiida.common import AttributeDict
from aiida.plugins import WorkflowFactory

PwBaseWorkflow = WorkflowFactory('quantumespresso.pw.base')


@calcfunction
def helper_get_hydrostatic_stress(output_trajectory, output_parameters):
    """
    doc
    """
    import numpy as np

    output_stress = output_trajectory.get_array('stress')[0]
    stress_unit = output_parameters['stress_units']
    hydrostatic_stress = np.trace(output_stress) / 3.0
    return orm.Dict(dict={
        'stress_unit': stress_unit,
        'hydrostatic_stress': hydrostatic_stress,
    })


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


class PressureEvaluationWorkChain(WorkChain):
    """WorkChain to calculate cohisive energy of input structure"""
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input('code',
                   valid_type=orm.Code,
                   help='The `pw.x` code use for the `PwCalculation`.')
        spec.input_namespace(
            'pseudos',
            valid_type=orm.UpfData,
            dynamic=True,
            help=
            'A mapping of `UpfData` nodes onto the kind name to which they should apply.'
        )
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
                   help='parameters for pwscf.')
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
            cls.results,
        )
        spec.output(
            'output_parameters',
            valid_type=orm.Dict,
            required=True,
            help=
            'The output parameters include cohesive energy of the structure.')
        spec.exit_code(201,
                       'ERROR_SUB_PROCESS_FAILED_SCF',
                       message='The `PwBaseWorkChain` sub process failed.')

    def setup(self):
        """Input validation"""
        # TODO set ecutwfc and ecutrho according to certain protocol
        # In order to get pressure set CONTROL tetress=.TRUE.
        self.ctx.pw_parameters = orm.Dict(dict={
            'CONTROL': {
                'tstress': True,
            },
        })
        self.ctx.pw_parameters.update_dict(
            self.inputs.parameters.pw.get_dict())
        self.ctx.kpoints_distance = self.inputs.parameters.kpoints_distance

    def validate_structure(self):
        """Create isolate atom and validate structure"""
        # create isolate atom structure
        self.ctx.pseudos = self.inputs.pseudos

    def run_scf(self):
        """
        set the inputs and submit scf to get quantities for pressure evaluation
        """
        # TODO tstress to cache the calculation for pressure convergence
        inputs = AttributeDict({
            'metadata': {
                'call_link_label': 'scf'
            },
            'pw': {
                'structure': self.inputs.structure,
                'code': self.inputs.code,
                'pseudos': self.ctx.pseudos,
                'parameters': self.ctx.pw_parameters,
                'settings': orm.Dict(dict={'CMDLINE': ['-ndiag', '1']}),
                'metadata': {},
            },
            'kpoints_distance': self.ctx.kpoints_distance,
        })

        if 'options' in self.inputs:
            options = self.inputs.options.get_dict()
        else:
            from aiida_quantumespresso.utils.resources import get_default_options

            options = get_default_options(with_mpi=True)

        inputs.pw.metadata.options = options

        running = self.submit(PwBaseWorkflow, **inputs)
        return ToContext(workchain_scf=running)

    def inspect_scf(self):
        """inspect the result of scf calculation."""
        workchain = self.ctx.workchain_scf

        if not workchain.is_finished_ok:
            self.report(
                f'PwBaseWorkChain for pressure evaluation failed with exit status {workchain.exit_status}'
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_SCF

        output_trajectory = workchain.outputs.output_trajectory
        output_pamameters = workchain.outputs.output_parameters

        # Return the output parameters of current workchain
        output_parameters = helper_get_hydrostatic_stress(
            output_trajectory, output_pamameters)
        self.out('output_parameters', output_parameters)

    def results(self):
        pass
