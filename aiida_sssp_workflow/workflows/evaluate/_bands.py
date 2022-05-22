# -*- coding: utf-8 -*-
"""
WorkChain calculate the bands for certain pseudopotential
"""
import numpy as np
from aiida import orm
from aiida.engine import ToContext, calcfunction, if_, while_
from aiida.plugins import DataFactory, WorkflowFactory

from . import _BaseEvaluateWorkChain

PwBandsWorkChain = WorkflowFactory("quantumespresso.pw.bands")
UpfData = DataFactory("pseudo.upf")


@calcfunction
def create_kpoints_from_distance(structure, distance, force_parity):
    """Generate a uniformly spaced kpoint mesh for a given structure.
    The spacing between kpoints in reciprocal space is guaranteed to be at least the defined distance.
    :param structure: the StructureData to which the mesh should apply
    :param distance: a Float with the desired distance between kpoints in reciprocal space
    :param force_parity: a Bool to specify whether the generated mesh should maintain parity
    :returns: a KpointsData with the generated mesh
    """
    from aiida.orm import KpointsData
    from numpy import linalg

    epsilon = 1e-5

    kpoints = KpointsData()
    kpoints.set_cell_from_structure(structure)
    kpoints.set_kpoints_mesh_from_density(
        distance.value, force_parity=force_parity.value
    )

    lengths_vector = [linalg.norm(vector) for vector in structure.cell]
    lengths_kpoint = kpoints.get_kpoints_mesh()[0]

    is_symmetric_cell = all(
        abs(length - lengths_vector[0]) < epsilon for length in lengths_vector
    )
    is_symmetric_mesh = all(length == lengths_kpoint[0] for length in lengths_kpoint)

    # If the vectors of the cell all have the same length, the kpoint mesh should be isotropic as well
    if is_symmetric_cell and not is_symmetric_mesh:
        nkpoints = max(lengths_kpoint)
        kpoints.set_kpoints_mesh([nkpoints, nkpoints, nkpoints])

    return kpoints


def validate_inputs(inputs, ctx=None):
    """Validate the inputs of the entire input namespace"""
    if inputs["run_bands_structure"] and (
        "kpoints_distance_band_structure" not in inputs
    ):
        return (
            BandsWorkChain.exit_codes.ERROR_KPOINTS_DISTANCE_BAND_STRUCTURE_NOT_SET.message
        )


class BandsWorkChain(_BaseEvaluateWorkChain):
    """WorkChain calculate the bands for certain pseudopotential
    Can choose only run bands or only on bandstructure"""

    # maximum number of bands factor increase loop
    # to prevent the infinite loop in bands evaluation
    _MAX_NUM_BANDS_FACTOR = 5

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        # yapf: disable
        super().define(spec)
        spec.expose_inputs(PwBandsWorkChain, include=['scf', 'bands', 'structure'])
        spec.input('kpoints_distance_bands', valid_type=orm.Float,
                    help='Kpoints distance setting for bulk energy bands calculation.')
        spec.input('init_nbands_factor', valid_type=orm.Float,
                    help='initial nbands factor.')
        spec.input('fermi_shift', valid_type=orm.Float, default=lambda: orm.Float(10.0),
                    help='The uplimit of energy to check the bands diff, control the number of bands.')
        spec.input('run_bands_structure', valid_type=orm.Bool, default=lambda: orm.Bool(False),
                    help='if True, run bands structure calculation on seekpath kpath.')
        spec.input('kpoints_distance_band_structure', valid_type=orm.Float, required=False,
                    help='Kpoints distance setting for band structure calculation.')
        spec.inputs.validator = validate_inputs

        spec.outline(
            cls.setup,

            cls.run_bands,
            while_(cls.not_enough_bands)(
                cls.increase_nbands,
                cls.run_bands,
            ),

            if_(cls.run_bands_structure)(
                cls.run_band_structure,
                cls.inspect_band_structure,
            ),
            cls.finalize,
        )

        spec.expose_outputs(PwBandsWorkChain, namespace='band_structure',
                            namespace_options={'dynamic': True, 'required': False})
        spec.expose_outputs(PwBandsWorkChain, namespace='bands')
        spec.exit_code(201, 'ERROR_SUB_PROCESS_FAILED_BANDS',
                    message='The `PwBandsWorkChain` sub process failed.')
        spec.exit_code(203, 'ERROR_REACH_MAX_BANDS_FACTOR_INCREASE_LOOP',
                    message=f'The maximum number={cls._MAX_NUM_BANDS_FACTOR} of bands factor'
                            'increase loop reached.')
        spec.exit_code(204, 'ERROR_KPOINTS_DISTANCE_BAND_STRUCTURE_NOT_SET',
                    message=f'kpoints distance is not set in inputs.')
        # yapf: enable

    def setup(self):
        """Input validation"""
        # set initial lowest highest bands eigenvalue - fermi_energy equals to 0.0
        self.ctx.nbands_factor = self.inputs.init_nbands_factor
        self.ctx.lowest_highest_eigenvalue = 0.0

        # For qe PwBandsWorkChain if `bands_kpoints` not set, the seekpath will run
        # to give a seekpath along the recommonded path.
        # There for I need to explicitly set the bands_kpoints and apply it
        # to the bands workchain.
        # While for the band_structure sub-workchain, the seekpath will run and
        # give band structure along the path.
        self.ctx.bands_kpoints = create_kpoints_from_distance(
            self.inputs.structure, self.inputs.kpoints_distance_bands, orm.Bool(False)
        )

        # break_increase_nbands set to True will force the while_ loop to stop.
        self.ctx.break_increase_nbands = False

    def run_bands(self):
        """run bands calculation"""
        inputs = self.exposed_inputs(PwBandsWorkChain)
        inputs["nbands_factor"] = self.ctx.nbands_factor
        inputs["bands_kpoints"] = self.ctx.bands_kpoints

        running = self.submit(PwBandsWorkChain, **inputs)
        self.report(f"Running pw bands calculation pk={running.pk}")
        return ToContext(workchain_bands=running)

    def _inspect_bands(self):
        """this is called from `not_enough_bands`
        separated for the name is more readable.

        Return workchain if it is finished_ok
        """
        workchain = self.ctx.workchain_bands

        if not workchain.is_finished_ok:
            self.logger.warning(
                f"PwBandsWorkChain for bands evaluation failed with exit status {workchain.exit_status}"
            )
            self.ctx.break_increase_nbands = True

            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_BANDS

        self.ctx.ecutwfc = workchain.inputs.scf.pw.parameters["SYSTEM"]["ecutwfc"]
        self.ctx.ecutrho = workchain.inputs.scf.pw.parameters["SYSTEM"]["ecutrho"]

        return workchain

    def not_enough_bands(self):
        """inspect and check if the number of bands enough for fermi shift (_FERMI_SHIFT)
        this is a if-else statement"""
        if self.ctx.break_increase_nbands:
            return False

        workchain = self._inspect_bands()

        fermi_energy = workchain.outputs.band_parameters["fermi_energy"]
        bands = workchain.outputs.band_structure.get_array("bands")
        # -1 colume for the highest eigenvalue of every kpoints
        # which might not be belong to one band if there are degeneracy
        # not enough until eigenvalues of all kpoints are greater than shift value.
        highest_band = bands[:, -1]

        return np.all(highest_band < fermi_energy + self.inputs.fermi_shift.value)

    def increase_nbands(self):
        """inspect the result of bands calculation."""
        # It is here not only because we don't want to stop bands number increase,
        # but also because of a bug that return exit in `while_` not terminate the workflow.
        #
        # If the bands evaluation failed it will keeps on increasing the `nbands_factor`
        # which lead to the infinite loop to this work chain.
        # Here I work around it by giving maximum nbands_factor loop to _MAX_NUM_BANDS_FACTOR=5.
        # And add a `break_increase_nbands` to break and get out from `not_enough_bands` immediately.
        if (
            self.ctx.nbands_factor
            > self.inputs.init_nbands_factor + self._MAX_NUM_BANDS_FACTOR
        ):
            return self.exit_codes.ERROR_REACH_MAX_BANDS_FACTOR_INCREASE_LOOP

        self.ctx.nbands_factor += 1.0

    def run_bands_structure(self):
        """whether run band structure"""
        return self.inputs.run_bands_structure.value

    def run_band_structure(self):
        """run band structure calculation"""
        inputs = self.exposed_inputs(PwBandsWorkChain)
        inputs["nbands_factor"] = self.ctx.nbands_factor
        inputs["bands_kpoints_distance"] = orm.Float(
            self.inputs.kpoints_distance_band_structure
        )

        running = self.submit(PwBandsWorkChain, **inputs)
        self.report(f"Running pw band structure calculation pk={running.pk}")
        return ToContext(workchain_band_structure=running)

    def inspect_band_structure(self):
        """inspect band structure and dump all its output"""
        self._inspect_workchain(
            namespace="band_structure", workchain=self.ctx.workchain_band_structure
        )

    def finalize(self):
        """inspect band and dump all its output"""
        self._inspect_workchain(namespace="bands", workchain=self.ctx.workchain_bands)

        self.out("ecutwfc", orm.Int(self.ctx.ecutwfc).store())
        self.out("ecutrho", orm.Int(self.ctx.ecutrho).store())

    def _inspect_workchain(self, namespace, workchain):
        if not workchain.is_finished_ok:
            self.logger.warning(
                f"PwBandsWorkChain uuid={workchain.uuid} failed with exit status {workchain.exit_status}"
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_BANDS

        self.out_many(
            self.exposed_outputs(workchain, PwBandsWorkChain, namespace=namespace)
        )
