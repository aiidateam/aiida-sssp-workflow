# -*- coding: utf-8 -*-
"""Workchain to calculate delta factor of specific psp"""
from aiida import orm
from aiida.common import AttributeDict
from aiida.engine import WorkChain, ToContext, calcfunction
from aiida.plugins import WorkflowFactory, CalculationFactory

birch_murnaghan_fit = CalculationFactory('sssp_workflow.birch_murnaghan_fit')
calculate_delta = CalculationFactory('sssp_workflow.calculate_delta')
EquationOfStateWorkChain = WorkflowFactory('sssp_workflow.eos')

@calcfunction
def helper_parse_upf(upf):
    return orm.Str(upf.element)

PW_PARAS = lambda: orm.Dict(dict={
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

class DeltaFactorWorkChain(WorkChain):
    """Workchain to calculate delta factor of specific pseudopotential"""

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        super().define(spec)
        spec.input('code', valid_type=orm.Code,
                   help='The `pw.x` code use for the `PwCalculation`.')
        spec.input('pseudo', valid_type=orm.UpfData, required=True, help='Pseudopotential to be verified')
        spec.input('structure', valid_type=orm.StructureData, required=False,
                   help='Ground state structure which the verification perform')
        spec.input('options', valid_type=orm.Dict, required=False,
            help='Optional `options` to use for the `PwCalculations`.')
        spec.input_namespace('parameters', help='Para')
        spec.input('parameters.pw', valid_type=orm.Dict, default=PW_PARAS, help='parameters for pwscf.')
        spec.input('parameters.kpoints_distance', valid_type=orm.Float, default=lambda: orm.Float(0.1),
                   help='Global kpoints setting.')
        spec.input('parameters.scale_count', valid_type=orm.Int, default=lambda: orm.Int(7),
                   help='Numbers of scale points in eos step.')
        spec.input('parameters.scale_increment', valid_type=orm.Float, default=lambda: orm.Float(0.02),
                   help='The scale increment in eos step.')
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
        spec.output('element', valid_type=orm.Str, required=True,
                    help='The element of the pseudopotential.')
        # TODO delta prime out
        spec.exit_code(201, 'ERROR_SUB_PROCESS_FAILED_EOS',
                       message='The `EquationOfStateWorkChain` sub process failed.')

    def setup(self):
        """Input validation"""
        # TODO set ecutwfc and ecutrho according to certain protocol
        self.ctx.pw_parameters = self.inputs.parameters.pw
        self.ctx.kpoints_distance = self.inputs.parameters.kpoints_distance

    def validate_structure_and_pseudo(self):
        """validate structure"""
        self.ctx.element = helper_parse_upf(self.inputs.pseudo)

        if not 'structure' in self.inputs:
            import importlib_resources

            element = self.ctx.element.value
            fpath = importlib_resources.path('aiida_sssp_workflow.CIFs', f'{element}.cif')
            with fpath as path:
                cif_data = orm.CifData.get_or_create(path)
                self.ctx.structure = cif_data[0].get_structure()
        else:
            self.ctx.structure = self.inputs.structure

    def run_eos(self):
        """run eos workchain"""
        inputs = AttributeDict({
            'structure': self.ctx.structure,
            'scale_count': self.inputs.parameters.scale_count,
            'scale_increment': self.inputs.parameters.scale_increment,
            'scf': {
                'pw': {
                    'code': self.inputs.code,
                    'pseudos': {self.ctx.element.value: self.inputs.pseudo},
                    'parameters': self.ctx.pw_parameters,
                    'metadata': {},
                },
                'kpoints_distance': self.ctx.kpoints_distance,
            }
        })

        if 'options' in self.inputs:
            inputs.scf.pw.metadata.options = self.inputs.options.get_dict()
        else:
            from aiida_quantumespresso.utils.resources import get_default_options

            inputs.scf.pw.metadata.options = get_default_options(with_mpi=True)

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
        self.out('element', self.ctx.element)
        # TODO output the parameters used for eos and pw.