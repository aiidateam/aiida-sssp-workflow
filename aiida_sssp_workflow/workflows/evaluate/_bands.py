# -*- coding: utf-8 -*-
"""
WorkChain calculate the bands for certain pseudopotential
"""
import numpy as np

from aiida import orm
from aiida.engine import WorkChain, ToContext, while_, if_, calcfunction
from aiida.plugins import WorkflowFactory, DataFactory

from aiida_sssp_workflow.utils import update_dict

PwBandsWorkChain = WorkflowFactory('quantumespresso.pw.bands')
UpfData = DataFactory('pseudo.upf')


@calcfunction
def create_kpoints_from_distance(structure, distance, force_parity):
    """Generate a uniformly spaced kpoint mesh for a given structure.
    The spacing between kpoints in reciprocal space is guaranteed to be at least the defined distance.
    :param structure: the StructureData to which the mesh should apply
    :param distance: a Float with the desired distance between kpoints in reciprocal space
    :param force_parity: a Bool to specify whether the generated mesh should maintain parity
    :returns: a KpointsData with the generated mesh
    """
    from numpy import linalg
    from aiida.orm import KpointsData

    epsilon = 1E-5

    kpoints = KpointsData()
    kpoints.set_cell_from_structure(structure)
    kpoints.set_kpoints_mesh_from_density(distance.value,
                                          force_parity=force_parity.value)

    lengths_vector = [linalg.norm(vector) for vector in structure.cell]
    lengths_kpoint = kpoints.get_kpoints_mesh()[0]

    is_symmetric_cell = all(
        abs(length - lengths_vector[0]) < epsilon for length in lengths_vector)
    is_symmetric_mesh = all(length == lengths_kpoint[0]
                            for length in lengths_kpoint)

    # If the vectors of the cell all have the same length, the kpoint mesh should be isotropic as well
    if is_symmetric_cell and not is_symmetric_mesh:
        nkpoints = max(lengths_kpoint)
        kpoints.set_kpoints_mesh([nkpoints, nkpoints, nkpoints])

    return kpoints


class BandsWorkChain(WorkChain):
    """WorkChain calculate the bands for certain pseudopotential"""

    _BANDS_SHIFT = 10.05
    _SEEKPATH_DISTANCE = 0.1

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
                    help='parameters for pwscf of calculation.')
        spec.input('ecutwfc', valid_type=orm.Float,
                    help='The ecutwfc set for both atom and bulk calculation. Please also set ecutrho if ecutwfc is set.')
        spec.input('ecutrho', valid_type=orm.Float,
                    help='The ecutrho set for both atom and bulk calculation.  Please also set ecutwfc if ecutrho is set.')
        spec.input('kpoints_distance', valid_type=orm.Float,
                    help='Kpoints distance setting for bulk energy calculation.')
        spec.input('init_nbands_factor', valid_type=orm.Float,
                    help='initial nbands factor.')
        spec.input('should_run_bands_structure', valid_type=orm.Bool, default=lambda: orm.Bool(False),
                    help='if True, run final bands structure calculation on seekpath kpath.')
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

            cls.run_bands,
            while_(cls.not_enough_bands)(
                cls.increase_nbands,
                cls.run_bands,
            ),

            if_(cls.should_run_bands_structure)(
                cls.run_band_structure,
                cls.inspect_band_structure,
            ),

            cls.results,
        )
        spec.output_namespace('seekpath_band_structure', dynamic=True,
                                help='output of band structure along seekpath.')
        spec.output('output_scf_parameters', valid_type=orm.Dict,
                    help='The output parameters of the SCF `PwBaseWorkChain`.')
        spec.output('output_bands_parameters', valid_type=orm.Dict,
                    help='The output parameters of the BANDS `PwBaseWorkChain`.')
        spec.output('output_bands_structure', valid_type=orm.BandsData,
                    help='The computed band structure.')
        spec.output('output_nbands_factor', valid_type=orm.Float,
                    help='The nbands factor of final bands run.')
        spec.exit_code(201, 'ERROR_SUB_PROCESS_FAILED_BANDS',
                    message='The `PwBandsWorkChain` sub process failed.')
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

        pw_scf_parameters = update_dict(pw_parameters, parameters)

        parameters = {
            'SYSTEM': {
                'ecutwfc': self.inputs.ecutwfc,
                'ecutrho': self.inputs.ecutrho,
                'noinv': True,
                'nosym': True,
            },
        }

        pw_bands_parameters = update_dict(pw_parameters, parameters)

        self.ctx.pw_scf_parameters = pw_scf_parameters
        self.ctx.pw_bands_parameters = pw_bands_parameters

        self.ctx.kpoints_distance = self.inputs.kpoints_distance

        # set initial lowest highest bands eigenvalue - fermi_energy equals to 0.0
        self.ctx.nbands_factor = self.inputs.init_nbands_factor
        self.ctx.lowest_highest_eigenvalue = 0.0

        self.ctx.bands_kpoints = create_kpoints_from_distance(
            self.inputs.structure, self.ctx.kpoints_distance, orm.Bool(False))

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

    def _get_base_bands_inputs(self):
        """
        get the inputs for raw band workflow
        """
        inputs = {
            'structure': self.inputs.structure,
            'scf': {
                'pw': {
                    'code': self.inputs.code,
                    'pseudos': self.ctx.pseudos,
                    'parameters': orm.Dict(dict=self.ctx.pw_scf_parameters),
                    'metadata': {
                        'options': self.ctx.options,
                    },
                    'parallelization': orm.Dict(dict=self.ctx.parallelization),
                },
                'kpoints_distance': self.ctx.kpoints_distance,
            },
            'bands': {
                'pw': {
                    'code': self.inputs.code,
                    'pseudos': self.ctx.pseudos,
                    'parameters': orm.Dict(dict=self.ctx.pw_bands_parameters),
                    'metadata': {
                        'options': self.ctx.options,
                    },
                    'parallelization': orm.Dict(dict=self.ctx.parallelization),
                },
            },
        }

        return inputs

    def run_bands(self):
        """run bands calculation"""
        inputs = self._get_base_bands_inputs()
        inputs['nbands_factor'] = self.ctx.nbands_factor
        inputs['bands_kpoints'] = self.ctx.bands_kpoints

        running = self.submit(PwBandsWorkChain, **inputs)
        self.report(f'Running pw bands calculation pk={running.pk}')
        return ToContext(workchain_bands=running)

    def not_enough_bands(self):
        """inspect and check if the number of bands enough for shift 10eV (_BANDS_SHIFT)"""
        workchain = self.ctx.workchain_bands

        if not workchain.is_finished_ok:
            self.report(
                f'PwBandsWorkChain for bands evaluation failed with exit status {workchain.exit_status}'
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_BANDS

        fermi_energy = workchain.outputs.band_parameters['fermi_energy']
        bands = workchain.outputs.band_structure.get_array('bands')
        self.ctx.highest_band = float(np.amin(bands[:, -1]))

        return self.ctx.highest_band - fermi_energy < self._BANDS_SHIFT

    def increase_nbands(self):
        """inspect the result of bands calculation."""
        self.ctx.nbands_factor += 1.0

    def should_run_bands_structure(self):
        """whether run band structure"""
        return self.inputs.should_run_bands_structure.value

    def run_band_structure(self):
        """run band structure calculation"""
        inputs = self._get_base_bands_inputs()
        inputs['nbands_factor'] = self.ctx.nbands_factor
        inputs['bands_kpoints_distance'] = orm.Float(self._SEEKPATH_DISTANCE)

        running = self.submit(PwBandsWorkChain, **inputs)
        self.report(f'Running pw band structure calculation pk={running.pk}')
        return ToContext(workchain_band_structure=running)

    def inspect_band_structure(self):
        """inspect band structure"""
        workchain = self.ctx.workchain_band_structure

        if not workchain.is_finished_ok:
            self.report(
                f'PwBandsWorkChain for bands structure failed with exit status {workchain.exit_status}'
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_BANDS

        self.report('pw band structure workchain successfully completed')
        self.out('seekpath_band_structure.output_scf_parameters',
                 workchain.outputs.scf_parameters)
        self.out('seekpath_band_structure.output_bands_parameters',
                 workchain.outputs.band_parameters)
        self.out('seekpath_band_structure.output_bands_structure',
                 workchain.outputs.band_structure)

    def results(self):
        """result"""
        self.report('pw bands workchain successfully completed')
        self.out('output_scf_parameters',
                 self.ctx.workchain_bands.outputs.scf_parameters)
        self.out('output_bands_parameters',
                 self.ctx.workchain_bands.outputs.band_parameters)
        self.out('output_bands_structure',
                 self.ctx.workchain_bands.outputs.band_structure)
        self.out('output_nbands_factor',
                 orm.Float(self.ctx.nbands_factor).store())
