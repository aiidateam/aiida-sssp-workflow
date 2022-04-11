# -*- coding: utf-8 -*-
"""
WorkChain calculate the bands for certain pseudopotential
"""
import numpy as np
from aiida import orm
from aiida.engine import ToContext, WorkChain, calcfunction, if_, while_
from aiida.plugins import DataFactory, WorkflowFactory

from aiida_sssp_workflow.utils import update_dict

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


class BandsWorkChain(WorkChain):
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
        spec.input('kpoints_distance_scf', valid_type=orm.Float,
                    help='Kpoints distance setting for bulk energy scf calculation.')
        spec.input('kpoints_distance_bands', valid_type=orm.Float,
                    help='Kpoints distance setting for bulk energy bands calculation.')
        spec.input('kpoints_distance_band_structure', valid_type=orm.Float, required=False,
                    help='Kpoints distance setting for band structure calculation.')
        spec.input('init_nbands_factor', valid_type=orm.Float,
                    help='initial nbands factor.')
        spec.input('fermi_shift', valid_type=orm.Float, default=lambda: orm.Float(10.0),
                    help='The uplimit of energy to check the bands diff, control the number of bands.')
        spec.input('should_run_bands_structure', valid_type=orm.Bool, default=lambda: orm.Bool(False),
                    help='if True, run bands structure calculation on seekpath kpath.')
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
        # yapf: enable

    def setup_base_parameters(self):
        """Input validation"""
        pw_parameters = self.inputs.pw_base_parameters.get_dict()

        parameters = {
            "SYSTEM": {
                "ecutwfc": self.inputs.ecutwfc,
                "ecutrho": self.inputs.ecutrho,
            },
        }

        pw_scf_parameters = update_dict(pw_parameters, parameters)

        parameters = {
            "SYSTEM": {
                "ecutwfc": self.inputs.ecutwfc,
                "ecutrho": self.inputs.ecutrho,
                "nosym": True,
            },
        }

        pw_bands_parameters = update_dict(pw_parameters, parameters)
        # if nbnd set from pw_base_parametres (lanthanoids case) remove the key `nbnd`
        pw_bands_parameters["SYSTEM"].pop("nbnd", None)

        self.ctx.pw_scf_parameters = pw_scf_parameters
        self.ctx.pw_bands_parameters = pw_bands_parameters

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

    def validate_structure(self):
        """doc"""
        self.ctx.pseudos = self.inputs.pseudos

    def setup_code_resource_options(self):
        """
        setup resource options and parallelization for `PwCalculation` from inputs
        """
        if "options" in self.inputs:
            self.ctx.options = self.inputs.options.get_dict()
        else:
            from aiida_sssp_workflow.utils import get_default_options

            self.ctx.options = get_default_options(with_mpi=True)

        if "parallelization" in self.inputs:
            self.ctx.parallelization = self.inputs.parallelization.get_dict()
        else:
            self.ctx.parallelization = {}

        self.report(f"resource options set to {self.ctx.options}")
        self.report(f"parallelization options set to {self.ctx.parallelization}")

    def _get_base_bands_inputs(self):
        """
        get the inputs for raw band workflow
        """
        inputs = {
            "structure": self.inputs.structure,
            "scf": {
                "pw": {
                    "code": self.inputs.code,
                    "pseudos": self.ctx.pseudos,
                    "parameters": orm.Dict(dict=self.ctx.pw_scf_parameters),
                    "metadata": {
                        "options": self.ctx.options,
                    },
                    "parallelization": orm.Dict(dict=self.ctx.parallelization),
                },
                "kpoints_distance": self.inputs.kpoints_distance_scf,
            },
            "bands": {
                "pw": {
                    "code": self.inputs.code,
                    "pseudos": self.ctx.pseudos,
                    "parameters": orm.Dict(dict=self.ctx.pw_bands_parameters),
                    "metadata": {
                        "options": self.ctx.options,
                    },
                    "parallelization": orm.Dict(dict=self.ctx.parallelization),
                },
            },
        }

        return inputs

    def run_bands(self):
        """run bands calculation"""
        inputs = self._get_base_bands_inputs()
        inputs["nbands_factor"] = self.ctx.nbands_factor
        inputs["bands_kpoints"] = self.ctx.bands_kpoints

        running = self.submit(PwBandsWorkChain, **inputs)
        self.report(f"Running pw bands calculation pk={running.pk}")
        return ToContext(workchain_bands=running)

    def not_enough_bands(self):
        """inspect and check if the number of bands enough for fermi shift (_FERMI_SHIFT)"""
        workchain = self.ctx.workchain_bands

        if not workchain.is_finished_ok:
            self.report(
                f"PwBandsWorkChain for bands evaluation failed with exit status {workchain.exit_status}"
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_BANDS

        fermi_energy = workchain.outputs.band_parameters["fermi_energy"]
        bands = workchain.outputs.band_structure.get_array("bands")
        # -1 colume for the highest eigenvalue of every kpoints
        # which might not be belong to one band if there are degeneracy
        # not enough until eigenvalues of all kpoints are greater than shift value.
        highest_band = bands[:, -1]

        return np.all(highest_band < fermi_energy + self.inputs.fermi_shift.value)

    def increase_nbands(self):
        """inspect the result of bands calculation."""
        # It is here because of a bug that return exit in `while_` not terminate the workflow.
        ###
        # If the bands evaluation failed it will keeps on increasing the `nbands_factor`
        # which lead to the infinite loop to this work chain.
        # Here I work around it by giving maximum nbands_factor loop to _MAX_NUM_BANDS_FACTOR=5.
        if (
            self.ctx.nbands_factor
            > self.inputs.init_nbands_factor + self._MAX_NUM_BANDS_FACTOR
        ):
            return self.exit_codes.ERROR_REACH_MAX_BANDS_FACTOR_INCREASE_LOOP

        self.ctx.nbands_factor += 1.0

    def should_run_bands_structure(self):
        """whether run band structure"""
        return self.inputs.should_run_bands_structure.value

    def run_band_structure(self):
        """run band structure calculation"""
        inputs = self._get_base_bands_inputs()
        inputs["nbands_factor"] = self.ctx.nbands_factor

        # since

        if "kpoints_distance_band_structure" in self.inputs:
            inputs["bands_kpoints_distance"] = orm.Float(
                self.inputs.kpoints_distance_band_structure
            )
        else:
            # TODO: logger warning for using scf k-distance
            inputs["bands_kpoints_distance"] = orm.Float(
                self.inputs.kpoints_distance_scf
            )

        running = self.submit(PwBandsWorkChain, **inputs)
        self.report(f"Running pw band structure calculation pk={running.pk}")
        return ToContext(workchain_band_structure=running)

    def inspect_band_structure(self):
        """inspect band structure"""
        self._inspect_workchain(
            namespace="band_structure", workchain=self.ctx.workchain_band_structure
        )

    def finalize(self):
        """inspect band"""
        self._inspect_workchain(namespace="bands", workchain=self.ctx.workchain_bands)

    def _inspect_workchain(self, namespace, workchain):
        if not workchain.is_finished_ok:
            self.report(
                f"PwBandsWorkChain uuid={workchain.uuid} failed with exit status {workchain.exit_status}"
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_BANDS

        self.report("pw band structure workchain successfully completed")
        self.out_many(
            self.exposed_outputs(workchain, PwBandsWorkChain, namespace=namespace)
        )
