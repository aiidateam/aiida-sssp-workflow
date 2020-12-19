# -*- coding: utf-8 -*-
"""
All in one verification workchain
"""
from aiida import orm
from aiida.engine import WorkChain

from aiida.plugins import WorkflowFactory

DeltaFactorWorkChain = WorkflowFactory('sssp_workflow.delta_factor')
ConvergenceCohesiveEnergy = WorkflowFactory(
    'sssp_workflow.convergence.cohesive_energy')
ConvergenceBandsWorkChain = WorkflowFactory('sssp_workflow.convergence.bands')
ConvergencePhononFrequencies = WorkflowFactory(
    'sssp_workflow.convergence.phonon_frequencies')
ConvergencePressureWorkChain = WorkflowFactory(
    'sssp_workflow.convergence.pressure')

PARA_ECUTWFC_LIST = lambda: orm.List(list=[
    20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 110,
    120, 130, 140, 160, 180, 200
])

PARA_ECUTRHO_LIST = lambda: orm.List(list=[
    160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680, 720,
    760, 800, 880, 960, 1040, 1120, 1280, 1440, 1600
])


class VerificationWorkChain(WorkChain):
    """doc"""

    _MAX_ECUTWFC = 200

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
        spec.input_namespace('parameters', help='Para')
        spec.input('parameters.ecutrho_list',
                   valid_type=orm.List,
                   default=lambda: PARA_ECUTRHO_LIST,
                   help='dual value for ecutrho list.')
        spec.input('parameters.ecutwfc_list',
                   valid_type=orm.List,
                   default=PARA_ECUTWFC_LIST,
                   help='list of ecutwfc evaluate list.')
        spec.input('parameters.dual',
                   valid_type=orm.Float,
                   required=True,
                   help='dual')
        spec.outline(
            cls.setup,
            cls.run_stage1,
            cls.run_stage2,
            cls.results,
        )
        spec.output('output_delta_factor',
                    valid_type=orm.Dict,
                    help='delta factor outputs')
        spec.output('output_convergence',
                    valid_type=orm.Dict,
                    help='results of every convergence test.')

    def setup(self):
        self.ctx.ecutwfc_list = self.inputs.parameters.ecutwfc_list
        self.ctx.ecutrho_list = self.inputs.parameters.ecutrho_list
        self.ctx.pseudo = self.inputs.pseudo
        self.ctx.dual = self.inputs.parameters.dual.value
        if not len(self.ctx.ecutwfc_list) == len(self.ctx.ecutrho_list):
            return self.exit_codes.ERROR_DIFFERENT_SIZE_OF_ECUTOFF_PAIRS

    def run_stage1(self):
        # run delta factor
        ecutwfc = 200.0
        ecutrho = ecutwfc * self.ctx.dual
        inputs = {
            'code':
            self.inputs.pw_code,
            'pseudo':
            self.ctx.pseudo,
            'options':
            orm.Dict(
                dict={
                    'resources': {
                        'num_machines': 1
                    },
                    'max_wallclock_seconds': 1800 * 3,
                    'withmpi': True,
                }),
            'parameters': {
                'pw':
                orm.Dict(dict={
                    'SYSTEM': {
                        'ecutwfc': ecutwfc,
                        'ecutrho': ecutrho,
                    },
                })
            },
        }

        running_delta_factor = self.submit(DeltaFactorWorkChain, **inputs)
        self.report(
            f'submit workchain delta factor pk={running_delta_factor.pk}')
        self.to_context(workchain_delta_factor=running_delta_factor)

        # -----------------------------------------
        if self.ctx.pseudo.element == 'fe':
            dual = 12.0
        else:
            dual = self.ctx.dual
        inputs = {
            'code': self.inputs.pw_code,
            'pseudo': self.ctx.pseudo,
            'parameters': {
                'ecutwfc_list':
                self.ctx.ecutwfc_list,
                'ecutrho_list':
                self.ctx.ecutrho_list,
                'ref_cutoff_pair':
                orm.List(list=[self._MAX_ECUTWFC, self._MAX_ECUTWFC * dual])
            },
        }

        # running cohesive energy convergence
        running_cohesive_convergence = self.submit(ConvergenceCohesiveEnergy,
                                                   **inputs)
        self.report(
            f'submit workchain cohesive energy convergence pk={running_cohesive_convergence.pk}'
        )
        self.to_context(
            workchain_cohesive_convergence=running_cohesive_convergence)

        # running bands convergence
        running_bands_convergence = self.submit(ConvergenceBandsWorkChain,
                                                **inputs)
        self.report(
            f'submit workchain bands convergence pk={running_bands_convergence.pk}'
        )
        self.to_context(workchain_bands_convergence=running_bands_convergence)

    def run_stage2(self):
        # TODO check previous workchain finish_ok

        # run phonon convergence test
        if self.ctx.pseudo.element == 'fe':
            dual = 12.0
        else:
            dual = self.ctx.dual
        inputs = {
            'pw_code': self.inputs.pw_code,
            'ph_code': self.inputs.ph_code,
            'pseudo': self.ctx.pseudo,
            'parameters': {
                'ecutwfc_list':
                self.ctx.ecutwfc_list,
                'ecutrho_list':
                self.ctx.ecutrho_list,
                'ref_cutoff_pair':
                orm.List(list=[self._MAX_ECUTWFC, self._MAX_ECUTWFC * dual])
            },
        }

        # running phonon convergence
        running_phonon_convergence = self.submit(ConvergencePhononFrequencies,
                                                 **inputs)
        self.report(
            f'submit workchain phonon convergence pk={running_phonon_convergence.pk}'
        )
        self.to_context(
            workchain_phonon_convergence=running_phonon_convergence)

        workchain_delta_factor = self.ctx.workchain_delta_factor
        res = workchain_delta_factor.outputs.output_birch_murnaghan_fit
        v0_b0_b1 = {
            'V0': res['v0'],
            'B0': res['b0'],
            'B1': res['bp'],
        }
        self.out('output_delta_factor',
                 workchain_delta_factor.outputs.output_parameters)

        inputs = {
            'code': self.inputs.pw_code,
            'pseudo': self.ctx.pseudo,
            'parameters': {
                'ecutwfc_list':
                self.ctx.ecutwfc_list,
                'ecutrho_list':
                self.ctx.ecutrho_list,
                'ref_cutoff_pair':
                orm.List(list=[self._MAX_ECUTWFC, self._MAX_ECUTWFC * dual]),
                'v0_b0_b1':
                orm.Dict(dict=v0_b0_b1),
            },
        }
        # running pressure convergence
        running_pressure_convergence = self.submit(
            ConvergencePressureWorkChain, **inputs)
        self.report(
            f'submit workchain pressure convergence pk={running_pressure_convergence.pk}'
        )
        self.to_context(
            workchain_pressure_convergence=running_pressure_convergence)

    def results(self):
        output_convergence = {}
        workchain_cohesive_convergence = self.ctx.workchain_cohesive_convergence
        output_convergence[
            'cohesive_energy'] = workchain_cohesive_convergence.outputs.output_parameters.get_dict(
            )

        workchain_bands_convergence = self.ctx.workchain_bands_convergence
        output_convergence[
            'bands'] = workchain_bands_convergence.outputs.output_parameters.get_dict(
            )

        workchain_phonon_convergence = self.ctx.workchain_phonon_convergence
        output_convergence[
            'phonon'] = workchain_phonon_convergence.outputs.output_parameters.get_dict(
            )

        workchain_pressure_convergence = self.ctx.workchain_pressure_convergence
        output_convergence[
            'pressure'] = workchain_pressure_convergence.outputs.output_parameters.get_dict(
            )

        self.out('output_convergence',
                 orm.Dict(dict=output_convergence).store())
        self.report(
            f'verification of pseudopotential pk={self.ctx.pseudo.pk} finished'
        )
