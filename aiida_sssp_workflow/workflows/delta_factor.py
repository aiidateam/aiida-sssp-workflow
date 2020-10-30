# -*- coding: utf-8 -*-
"""Workchain to calculate delta factor of specific psp"""
from aiida import orm
from aiida.engine import WorkChain, ToContext, workfunction
from aiida.plugins import WorkflowFactory, CalculationFactory

birch_murnaghan_fit = CalculationFactory('sssp_workflow.birch_murnaghan_fit')
calculate_delta = CalculationFactory('sssp_workflow.calculate_delta')
EquationOfStateWorkChain = WorkflowFactory('sssp_workflow.eos')


class DeltaFactorWorkchain(WorkChain):
    """Workchain to calculate delta factor of specific psp"""

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        super().difine(spec)
        spec.input('code')
        spec.input('pseudo', valid_type=orm.UpfData, help='Pseudopotential to be verified')
        spec.input('structure', valid_type=orm.StructureData,
                   help='Ground state structure (a family?) which the verification perform')
        # TODO clean_workdir
        spec.outline(
            cls.setup,
            cls.validate_structure_and_pseudo,
            cls.run_eos,
            cls.inspect_eos,
            cls.run_delta_calc,
            cls.results,
        )
        spec.output('delta_factor', valid_type=orm.Float, required=True,
                 help='The delta factor of the pseudopotential.')
        # TODO delta prime out

    def setup(self):
        """Input validation"""
        pass

    def validate_structure_and_pseudo(self):
        """validate structure"""
        self.ctx.element = self.inputs.pseudo.element

    def run_eos(self):
        """run eos workchain"""
        inputs = AttributeDict({
            'structure': structure,
            'scale_count': orm.Int(7),
            'scf': {
                'pw': {
                    'code': load_code('qe-6.6-pw@daint-mc'),
                    'pseudos': {self.ctx.element: self.inputs.pseudo},
                    'parameters': orm.Dict(dict={
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
                    }),
                    'metadata': {
                        'options': {
                            'resources': {'num_machines': 1},
                            'max_wallclock_seconds': 1800,
                            'withmpi': True,
                        },
                    },
                },
                'kpoints_distance': orm.Float(0.1),
            }
        })
        running = self.submit(EquationOfStateWorkChain, **inputs)

        self.report(f'launching EquationOfStateWorkChain<{running.pk}>')

        return ToContext(workchain_eos=running)

    def inspect_eos(self):
        """Inspect the results of EquationOfStateWorkChain
        and run the Birch-Murnaghan fit"""
        workchain = self.ctx.workchain_eos

        if not workchain.is_finished_ok:
            self.report(f'EquationOfStateWorkChain failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_EOS

        volume_energy = workchain.outputs.output_parameters # This keep the provenance
        self.ctx.birch_murnaghan_fit_result = birch_murnaghan_fit(volume_energy)
        # TODO report result and output it

    def run_delta_calc(self):
        """calculate the delta factor"""
        res = self.ctx.birch_murnaghan_fit_result
        inputs = {
            'element': self.ctx.element,
            'v0': res['volume0'],
            'b0': res['bulk_modulus0'],
            'bp': res['bulk_deriv0'],
        }
        self.ctx.delta_factor = calculate_delta(**inputs)
        # TODO report

    def results(self):
        """Attach the output parameters to the outputs."""
        self.out('delta_factor', self.ctx.delta_factor)