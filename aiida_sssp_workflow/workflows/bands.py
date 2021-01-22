# -*- coding: utf-8 -*-
"""
WorkChain calculate the bands for certain pseudopotential
"""
import numpy as np

from aiida import orm
from aiida.engine import WorkChain, ToContext, while_, if_
from aiida.plugins import WorkflowFactory, CalculationFactory

from aiida_sssp_workflow.utils import update_dict

PwBandsWorkChain = WorkflowFactory('quantumespresso.pw.bands')
create_kpoints_from_distance = CalculationFactory(
    'quantumespresso.create_kpoints_from_distance')

PW_PARAS = lambda: orm.Dict(
    dict={
        'SYSTEM': {
            'ecutrho': 800,
            'ecutwfc': 200,
        },
        'ELECTRONS': {
            'conv_thr': 1e-10,
        },
    })


class BandsWorkChain(WorkChain):
    """WorkChain calculate the bands for certain pseudopotential"""

    _SCF_PARAMETERS = {
        'SYSTEM': {
            'degauss': 0.00735,
            'occupations': 'smearing',
            'smearing': 'marzari-vanderbilt',
        },
        'ELECTRONS': {
            'conv_thr': 1e-10,
        },
    }

    _BAND_PARAMETERS = {
        'SYSTEM': {
            'degauss': 0.00735,
            'occupations': 'smearing',
            'smearing': 'marzari-vanderbilt',
            "noinv": True,
            "nosym": True,
        },
        'ELECTRONS': {
            'conv_thr': 1e-10,
        },
    }

    _BANDS_SHIFT = 10.5
    _SEEKPATH_DISTANCE = 0.1
    _SCF_CMDLINE_SETTING = {'CMDLINE': ['-ndiag', '1', '-nk', '4']}
    _BAND_CMDLINE_SETTING = {'CMDLINE': ['-nk', '4']}
    _MAX_WALLCLOCK_SECONDS = 1800 * 3

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
        spec.input(
            'parameters.pw',
            valid_type=orm.Dict,
            required=False,
            help=
            'parameters for pw.x, if not set use the default hard code one.')
        spec.input(
            'parameters.run_band_structure',
            default=lambda: orm.Bool(False),
            help='If True, run to get refined band structure along path.')
        spec.input(
            'parameters.ecutwfc',
            valid_type=(orm.Float, orm.Int),
            required=False,
            help=
            'The ecutwfc set for both atom and bulk calculation. Please also set ecutrho if ecutwfc is set.'
        )
        spec.input(
            'parameters.ecutrho',
            valid_type=(orm.Float, orm.Int),
            required=False,
            help=
            'The ecutrho set for both atom and bulk calculation.  Please also set ecutwfc if ecutrho is set.'
        )
        spec.input('parameters.scf_kpoints_distance',
                   valid_type=orm.Float,
                   default=lambda: orm.Float(0.1),
                   help='Kpoints distance setting for scf calculation.')
        spec.input('parameters.nbands_factor',
                   valid_type=orm.Float,
                   default=lambda: orm.Float(1.0),
                   help='Bands number factor in bands calculation.')
        spec.input('parameters.bands_kpoints_distance',
                   valid_type=orm.Float,
                   default=lambda: orm.Float(0.15),
                   help='Kpoints distance setting for bands nscf calculation.')
        spec.outline(
            cls.setup,
            cls.validate_structure,
            while_(cls.not_enough_bands)(
                cls.run_bands,
                cls.inspect_bands,
            ),
            if_(cls.should_band_structure)(
                cls.run_band_structure,
                cls.inspect_band_structure,
            ),
            cls.results,
        )
        spec.output_namespace('seekpath_band_structure',
                              dynamic=True,
                              help='output of band structure along seekpath.')
        spec.output('scf_parameters',
                    valid_type=orm.Dict,
                    help='The output parameters of the SCF `PwBaseWorkChain`.')
        spec.output(
            'band_parameters',
            valid_type=orm.Dict,
            help='The output parameters of the BANDS `PwBaseWorkChain`.')
        spec.output('band_structure',
                    valid_type=orm.BandsData,
                    help='The computed band structure.')
        spec.output('nbands_factor',
                    valid_type=orm.Float,
                    help='The nbands factor of final bands run.')
        spec.exit_code(201,
                       'ERROR_SUB_PROCESS_FAILED_BANDS',
                       message='The `PwBandsWorkChain` sub process failed.')

    def setup(self):
        """Input validation"""
        scf_parameters = self._SCF_PARAMETERS
        bands_parameters = self._BAND_PARAMETERS

        if 'pw' in self.inputs.parameters:
            self.ctx.pw_parameters = self.inputs.parameters.pw.get_dict()
        else:
            self.ctx.pw_parameters = {}

        pw_scf_parameters = update_dict(scf_parameters, self.ctx.pw_parameters)
        pw_bands_parameters = update_dict(bands_parameters,
                                          self.ctx.pw_parameters)

        # nbnd can not sit with nband_factor and nbnd might be set somewhere
        pw_bands_parameters['SYSTEM'].pop('nbnd', None)

        if self.inputs.parameters.ecutwfc and self.inputs.parameters.ecutrho:
            parameters = {
                'SYSTEM': {
                    'ecutwfc': self.inputs.parameters.ecutwfc,
                    'ecutrho': self.inputs.parameters.ecutrho,
                },
            }
            pw_scf_parameters = update_dict(pw_scf_parameters, parameters)
            pw_bands_parameters = update_dict(pw_bands_parameters, parameters)

        self.ctx.pw_scf_parameters = orm.Dict(dict=pw_scf_parameters)
        self.ctx.pw_bands_parameters = orm.Dict(dict=pw_bands_parameters)

        self.ctx.scf_kpoints_distance = self.inputs.parameters.scf_kpoints_distance

        # set initial lowest highest bands eigenvalue - fermi_energy equals to 0.0
        self.ctx.nbands_factor = self.inputs.parameters.nbands_factor
        self.ctx.lowest_highest_eigenvalue = 0.0

        bands_kpoints_distance = self.inputs.parameters.bands_kpoints_distance
        self.ctx.bands_kpoints = create_kpoints_from_distance(
            self.inputs.structure, bands_kpoints_distance, orm.Bool(False))

    def validate_structure(self):
        """Create isolate atom and validate structure"""
        self.ctx.pseudos = self.inputs.pseudos

    def not_enough_bands(self):
        """Check if the number of bands enough for shift 10eV"""
        bands_shift = self._BANDS_SHIFT  # hard code for the time being
        return self.ctx.lowest_highest_eigenvalue < bands_shift

    def _get_band_inputs(self):
        if 'options' in self.inputs:
            options = self.inputs.options.get_dict()
        else:
            from aiida_quantumespresso.utils.resources import get_default_options

            # Too many kpoints may go beyond 1800s
            options = get_default_options(
                with_mpi=True,
                max_wallclock_seconds=self._MAX_WALLCLOCK_SECONDS)

        inputs = {
            'structure': self.inputs.structure,
            'scf': {
                'pw': {
                    'code': self.inputs.code,
                    'pseudos': self.ctx.pseudos,
                    'parameters': self.ctx.pw_scf_parameters,
                    'settings': orm.Dict(dict=self._SCF_CMDLINE_SETTING),
                    'metadata': {
                        'options': options
                    },
                },
                'kpoints_distance': self.ctx.scf_kpoints_distance,
            },
            'bands': {
                'pw': {
                    'code': self.inputs.code,
                    'pseudos': self.ctx.pseudos,
                    'parameters': self.ctx.pw_bands_parameters,
                    'settings': orm.Dict(dict=self._BAND_CMDLINE_SETTING),
                    'metadata': {
                        'options': options
                    },
                },
            },
        }

        return inputs

    def run_bands(self):
        inputs = self._get_band_inputs()
        inputs['nbands_factor'] = self.ctx.nbands_factor
        inputs['bands_kpoints'] = self.ctx.bands_kpoints

        running = self.submit(PwBandsWorkChain, **inputs)
        self.report(f'Running pw bands calculation pk={running.pk}')
        return ToContext(workchain_bands=running)

    def inspect_bands(self):
        """inspect the result of bands calculation."""
        workchain = self.ctx.workchain_bands

        if not workchain.is_finished_ok:
            self.report(
                f'PwBandsWorkChain for bands evaluation failed with exit status {workchain.exit_status}'
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_BANDS

        fermi_energy = workchain.outputs.band_parameters['fermi_energy']
        bands = workchain.outputs.band_structure.get_array(
            'bands') - fermi_energy
        eigv = float(np.amin(bands[:, -1]))
        self.ctx.lowest_highest_eigenvalue = eigv

        self.ctx.nbands_factor += 1.0
        self.ctx.output_nbands_factor = self.ctx.nbands_factor - 1.0

    def should_band_structure(self):
        return self.inputs.parameters.run_band_structure.value

    def run_band_structure(self):
        inputs = self._get_band_inputs()
        inputs['nbands_factor'] = self.ctx.output_nbands_factor
        inputs['bands_kpoints_distance'] = orm.Float(self._SEEKPATH_DISTANCE)

        running = self.submit(PwBandsWorkChain, **inputs)
        self.report(f'Running pw band structure calculation pk={running.pk}')
        return ToContext(workchain_band_structure=running)

    def inspect_band_structure(self):
        workchain = self.ctx.workchain_band_structure

        if not workchain.is_finished_ok:
            self.report(
                f'PwBandsWorkChain for bands structure failed with exit status {workchain.exit_status}'
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_BANDS

        self.report('pw band structure workchain successfully completed')
        self.out('seekpath_band_structure.scf_parameters',
                 workchain.outputs.scf_parameters)
        self.out('seekpath_band_structure.band_parameters',
                 workchain.outputs.band_parameters)
        self.out('seekpath_band_structure.band_structure',
                 workchain.outputs.band_structure)

    def results(self):
        self.report('pw bands workchain successfully completed')
        self.out('scf_parameters',
                 self.ctx.workchain_bands.outputs.scf_parameters)
        self.out('band_parameters',
                 self.ctx.workchain_bands.outputs.band_parameters)
        self.out('band_structure',
                 self.ctx.workchain_bands.outputs.band_structure)
        self.out('nbands_factor',
                 orm.Float(self.ctx.output_nbands_factor).store())
