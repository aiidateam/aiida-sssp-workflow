# -*- coding: utf-8 -*-
"""Workchain to calculate delta factor of specific psp"""
from aiida import orm
from aiida.engine import WorkChain, ToContext
from aiida.plugins import WorkflowFactory

EquationOfStateWorkChain = WorkflowFactory('sssp_workflow.eos')


class DeltaFactorWorkchain(WorkChain):
    """Workchain to calculate delta factor of specific psp"""

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        super().difine(spec)
        spec.input('pseudo', valid_type=orm.SinglefileData, help='Pseudopotential to be verified')
        spec.input('structure', valid_type=orm.StructureData,
                   help='Ground state structure (a family?) which the verification perform')
        # TODO clean_workdir
        spec.outline(
            cls.setup,
            cls.validate_structure,
            cls.run_eos,
            cls.inspect_eos,
            cls.run_delta_calc,
            cls.results,
        )
        spec.out('output_parameters', valid_type=orm.Dict, required=True,
                 help='The result of delta factor of pseudopotential')
        # TODO delta prime out

    def setup(self):
        """Input validation"""
        pass

    def validate_structure(self):
        """validate structure"""
        pass

    def run_eos(self):
        """run eos workchain"""
        inputs = AttributeDict({
            'structure': structure,
            'scale_count': orm.Int(7),
            'scf': {
                'pw': {
                    'code': load_code('qe-6.6-pw@daint-mc'),
                    'pseudos': {upf.element: upf},
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
        pass

    def run_delta_calc(self):
        """calculate the delta factor"""
        pass

    def results(self):
        """Attach the output parameters to the outputs."""
        pass