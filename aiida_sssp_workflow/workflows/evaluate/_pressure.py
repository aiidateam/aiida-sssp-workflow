# -*- coding: utf-8 -*-
"""
A calcfunctian create_isolate_atom
Create the structure of isolate atom
"""
from aiida import orm
from aiida.engine import calcfunction, WorkChain, ToContext
from aiida.plugins import WorkflowFactory, DataFactory

from aiida_sssp_workflow.utils import update_dict

PwBaseWorkflow = WorkflowFactory('quantumespresso.pw.base')
UpfData = DataFactory('pseudo.upf')


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


class PressureWorkChain(WorkChain):
    """WorkChain to calculate cohisive energy of input structure"""
    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        # yapf: disable
        super().define(spec)
        spec.input('code', valid_type=orm.Code,
                    help='The `pw.x` code use for the `PwCalculation`.')
        spec.input_namespace('pseudos', valid_type=UpfData, dynamic=True,
                    help='A mapping of `UpfData` nodes onto the kind name to which they should apply.')
        spec.input('structure', valid_type=orm.StructureData,
                    help='Ground state structure which the verification perform')
        spec.input('pw_base_parameters', valid_type=orm.Dict,
                    help='parameters for pwscf.')
        spec.input('ecutwfc', valid_type=orm.Float,
                    help='The ecutwfc set for both atom and bulk calculation. Please also set ecutrho if ecutwfc is set.')
        spec.input('ecutrho', valid_type=orm.Float,
                    help='The ecutrho set for both atom and bulk calculation.  Please also set ecutwfc if ecutrho is set.')
        spec.input('kpoints_distance', valid_type=orm.Float,
                    help='Kpoints distance setting for bulk energy calculation.')
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
        )
        spec.output('output_parameters', valid_type=orm.Dict, required=True,
                    help='The output parameters include cohesive energy of the structure.')
        spec.exit_code(211, 'ERROR_SUB_PROCESS_FAILED_ATOM_ENERGY',
                    message='PwBaseWorkChain of atom energy evaluation failed.')
        spec.exit_code(212, 'ERROR_SUB_PROCESS_FAILED_BULK_ENERGY',
                    message='PwBaseWorkChain of bulk structure energy evaluation failed with exit status.')
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

    def validate_structure(self):
        """doc"""
        self.ctx.pseudos = self.inputs.pseudos

    def setup_code_resource_options(self):
        """
        setup resource options and parallelization for `PwCalculation` from inputs
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
                'code': self.inputs.code,
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
        """inspect the result of scf calculation."""
        workchain = self.ctx.workchain_scf

        if not workchain.is_finished_ok:
            self.report(
                f'PwBaseWorkChain for pressure evaluation failed with exit status {workchain.exit_status}'
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_SCF

        output_trajectory = workchain.outputs.output_trajectory
        output_parameters = workchain.outputs.output_parameters

        # Return the output parameters of current workchain
        output_parameters = helper_get_hydrostatic_stress(
            output_trajectory, output_parameters)
        self.out('output_parameters', output_parameters)
