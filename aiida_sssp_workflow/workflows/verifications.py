# -*- coding: utf-8 -*-
"""
All in one verification workchain
"""
from aiida import orm
from aiida.engine import WorkChain

from aiida.plugins import WorkflowFactory

from aiida_sssp_workflow.helpers import get_pw_inputs_from_pseudo

BandsWorkChain = WorkflowFactory('sssp_workflow.evaluation.bands')
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
        spec.input('options',
                   valid_type=orm.Dict,
                   required=False,
                   help='Optional `options` to use for the `PwCalculations`.')
        spec.input('protocol',
                   valid_type=orm.Str,
                   default=lambda: orm.Str('efficiency'),
                   help='The protocol to use for the workchain.')
        spec.input_namespace('parameters', help='Para')
        spec.input('parameters.ecutrho_list',
                   valid_type=orm.List,
                   default=lambda: PARA_ECUTRHO_LIST,
                   help='dual value for ecutrho list.')
        spec.input('parameters.ecutwfc_list',
                   valid_type=orm.List,
                   default=PARA_ECUTWFC_LIST,
                   help='list of ecutwfc evaluate list.')
        spec.input('parameters.ref_cutoff_pair',
                   valid_type=orm.List,
                   required=True,
                   default=lambda: orm.List(list=[200, 1600]),
                   help='ecutwfc/ecutrho pair for reference calculation.')
        spec.input(
            'clean_workdir',
            valid_type=orm.Bool,
            default=lambda: orm.Bool(False),
            help=
            'If `True`, work directories of all called calculation will be cleaned at the end of execution.'
        )
        spec.outline(
            cls.setup,
            cls.run_stage1,
            cls.run_stage2,
            cls.results,
        )
        spec.output_namespace('delta_factor',
                              dynamic=True,
                              help='results of delta factor calculation.')
        spec.output_namespace(
            'convergence_cohesive_energy',
            dynamic=True,
            help='results of convergence cohesive energy calculation.')
        spec.output_namespace(
            'convergence_phonon_frequencies',
            dynamic=True,
            help='results of convergence phonon_frequencies calculation.')
        spec.output_namespace(
            'convergence_pressure',
            dynamic=True,
            help='results of convergence pressure calculation.')
        spec.output_namespace('convergence_bands',
                              dynamic=True,
                              help='results of convergence bands calculation.')
        spec.output_namespace('band_structure',
                              dynamic=True,
                              help='results of band structure calculation.')

        spec.exit_code(
            811,
            'WARNING_NOT_ALL_SUB_WORKFLOW_OK',
            message='The sub-workflows {processes} is not finished ok.')

    def setup(self):
        self.ctx.ecutwfc_list = self.inputs.parameters.ecutwfc_list
        self.ctx.ecutrho_list = self.inputs.parameters.ecutrho_list
        self.ctx.pseudo = self.inputs.pseudo
        if not len(self.ctx.ecutwfc_list) == len(self.ctx.ecutrho_list):
            return self.exit_codes.ERROR_DIFFERENT_SIZE_OF_ECUTOFF_PAIRS

        # to collect workchains in a dict
        self.ctx.workchains = {}

    def run_stage1(self):
        """
        In this stage run:
        1) delta factor calculation, the results also used for pressure convergence
        2) phonon frequencies convergence since ph.x calculation depend on the workdir
        3) bands distance convergence.
        4) band_structure evaluation to get seekpath band structure
        """
        # run delta factor
        inputs = {
            'code': self.inputs.pw_code,
            'pseudo': self.ctx.pseudo,
            'protocol': self.inputs.protocol,
        }
        if 'options' in self.inputs:
            inputs['options'] = self.inputs.options

        running = self.submit(DeltaFactorWorkChain, **inputs)
        self.report(f'submit workchain delta factor pk={running}')
        self.to_context(w00=running)
        self.ctx.workchains['delta_factor'] = running

        # -----------------------------------------
        # run phonon convergence test
        inputs = {
            'pw_code': self.inputs.pw_code,
            'ph_code': self.inputs.ph_code,
            'pseudo': self.ctx.pseudo,
            'protocol': self.inputs.protocol,
            'parameters': {
                'ecutwfc_list': self.ctx.ecutwfc_list,
                'ecutrho_list': self.ctx.ecutrho_list,
                'ref_cutoff_pair': self.inputs.parameters.ref_cutoff_pair,
            },
        }
        if 'options' in self.inputs:
            inputs['options'] = self.inputs.options

        # running phonon convergence
        running = self.submit(ConvergencePhononFrequencies, **inputs)
        self.report(f'submit workchain phonon convergence pk={running.pk}')
        self.to_context(w01=running)
        self.ctx.workchains['convergence_phonon_frequencies'] = running

        # running bands convergence
        inputs = {
            'code': self.inputs.pw_code,
            'pseudo': self.ctx.pseudo,
            'protocol': self.inputs.protocol,
            'parameters': {
                'ecutwfc_list': self.ctx.ecutwfc_list,
                'ecutrho_list': self.ctx.ecutrho_list,
                'ref_cutoff_pair': self.inputs.parameters.ref_cutoff_pair,
            },
        }
        if 'options' in self.inputs:
            inputs['options'] = self.inputs.options

        running = self.submit(ConvergenceBandsWorkChain, **inputs)
        self.report(f'submit workchain bands convergence pk={running.pk}')
        self.to_context(w02=running)
        self.ctx.workchains['convergence_bands'] = running

        # running band structure
        res = get_pw_inputs_from_pseudo(pseudo=self.ctx.pseudo)

        structure = res['structure']
        pseudos = res['pseudos']

        cutoff_pair = self.inputs.parameters.ref_cutoff_pair.get_list()
        ecutwfc = cutoff_pair[0]
        ecutrho = cutoff_pair[1]

        inputs = {
            'code': self.inputs.pw_code,
            'pseudos': pseudos,
            'structure': structure,
            'parameters': {
                'ecutwfc': orm.Float(ecutwfc),
                'ecutrho': orm.Float(ecutrho),
                'run_band_structure': orm.Bool(True),
                'nbands_factor': orm.Float(2)
            },
        }
        running = self.submit(BandsWorkChain, **inputs)
        self.report(f'submit workchain band structure pk={running.pk}')
        self.to_context(w_band_structure=running)
        self.ctx.workchains['band_structure'] = running

    def run_stage2(self):
        """
        The stage2 runs
        1) Cohesive energy convergence
        2) pressure convergence which use V0_B0_B1 from delta_factor calculation as inputs
        """
        # -----------------------------------
        # Cohesive energy
        inputs = {
            'code': self.inputs.pw_code,
            'pseudo': self.ctx.pseudo,
            'protocol': self.inputs.protocol,
            'parameters': {
                'ecutwfc_list': self.ctx.ecutwfc_list,
                'ecutrho_list': self.ctx.ecutrho_list,
                'ref_cutoff_pair': self.inputs.parameters.ref_cutoff_pair,
            },
        }
        if 'options' in self.inputs:
            inputs['options'] = self.inputs.options

        # running cohesive energy convergence
        running = self.submit(ConvergenceCohesiveEnergy, **inputs)
        self.report(
            f'submit workchain cohesive energy convergence pk={running.pk}')
        self.to_context(w03=running)
        self.ctx.workchains['convergence_cohesive_energy'] = running

        # -----------------------------------------
        # Pressure
        workchain = self.ctx.workchains['delta_factor']

        if workchain.is_finished_ok:
            for label in workchain.outputs:
                self.out(f'delta_factor.{label}', workchain.outputs[label])

            res = workchain.outputs.output_birch_murnaghan_fit
            v0_b0_b1 = {
                'V0': res['V0'],
                'B0': res['B0'],
                'B1': res['B1'],
            }

            inputs = {
                'code': self.inputs.pw_code,
                'pseudo': self.ctx.pseudo,
                'parameters': {
                    'ecutwfc_list': self.ctx.ecutwfc_list,
                    'ecutrho_list': self.ctx.ecutrho_list,
                    'ref_cutoff_pair': self.inputs.parameters.ref_cutoff_pair,
                    'v0_b0_b1': orm.Dict(dict=v0_b0_b1),
                },
            }
            if 'options' in self.inputs:
                inputs['options'] = self.inputs.options
            # running pressure convergence
            running = self.submit(ConvergencePressureWorkChain, **inputs)
            self.report(
                f'submit workchain pressure convergence pk={running.pk}')
            self.to_context(w04=running)
            self.ctx.workchains['convergence_pressure'] = running

    def results(self):
        not_finished_ok = {}
        for wname, workchain in self.ctx.workchains.items():
            if workchain.is_finished_ok:
                for label in workchain.outputs:
                    self.out(f'{wname}.{label}', workchain.outputs[label])
            else:
                self.report(
                    f'The sub-workflow {wname} pk={workchain.pk} not finished ok.'
                )
                not_finished_ok[wname] = workchain.pk

        if not_finished_ok:
            return self.exit_codes.WARNING_NOT_ALL_SUB_WORKFLOW_OK.format(
                processes=not_finished_ok)

    def on_terminated(self):
        """Clean the working directories of all child calculations if `clean_workdir=True` in the inputs."""
        super().on_terminated()

        if self.inputs.clean_workdir.value is False:
            self.report('remote folders will not be cleaned')
            return

        cleaned_calcs = []

        for called_descendant in self.node.called_descendants:
            if isinstance(called_descendant, orm.CalcJobNode):
                try:
                    called_descendant.outputs.remote_folder._clean()  # pylint: disable=protected-access
                    cleaned_calcs.append(called_descendant.pk)
                except (IOError, OSError, KeyError):
                    pass

        if cleaned_calcs:
            self.report(
                f"cleaned remote folders of calculations: {' '.join(map(str, cleaned_calcs))}"
            )
