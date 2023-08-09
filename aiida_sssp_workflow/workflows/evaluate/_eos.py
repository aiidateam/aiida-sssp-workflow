# -*- coding: utf-8 -*-
"""Equation of state workflow that can use any code plugin implementing the common relax workflow."""
from typing import List

from aiida import orm
from aiida.engine import WorkChain, append_, calcfunction, run_get_node
from aiida.plugins import CalculationFactory, WorkflowFactory

PwBaseWorkChain = WorkflowFactory("quantumespresso.pw.base")
birch_murnaghan_fit = CalculationFactory("sssp_workflow.birch_murnaghan_fit")


@calcfunction
def scale_structure(
    structure: orm.StructureData, scale_factor: orm.Float
) -> orm.StructureData:
    """Scale the structure with the given scaling factor."""
    ase = structure.get_ase().copy()
    ase.set_cell(ase.get_cell() * float(scale_factor) ** (1 / 3), scale_atoms=True)
    return orm.StructureData(ase=ase)


class _EquationOfStateWorkChain(WorkChain):
    """Workflow to compute the equation of state for a given crystal structure."""

    @classmethod
    def define(cls, spec):
        # yapf: disable
        super().define(spec)
        spec.input('structure', valid_type=orm.StructureData, help='The structure at equilibrium volume.')
        spec.input('kpoints_distance', valid_type=orm.Float, required=True,
            help='The kpoints distance used in generating the kmesh of unscaled structure then for all scaled structures')
        spec.input('scale_factors', valid_type=orm.List, required=False,
            help='The list of scale factors at which the volume and total energy of the structure should be computed.')
        spec.input('scale_count', valid_type=orm.Int, default=lambda: orm.Int(7),
            help='The number of points to compute for the equation of state.')
        spec.input('scale_increment', valid_type=orm.Float, default=lambda: orm.Float(0.02),
            help='The relative difference between consecutive scaling factors.')
        spec.expose_inputs(PwBaseWorkChain,
            exclude=('kpoints', 'pw.structure', 'pw.kpoints_distance'),
            namespace_options={'help': 'Inputs for the `PwBaseWorkChain` for the SCF calculation.'})
        spec.outline(
            cls.run_init,
            cls.run_eos,
            cls.inspect_eos,
        )
        spec.output('output_volume_energy', valid_type=orm.Dict,
            help='Results volumes and energise.')
        spec.output('output_birch_murnaghan_fit', valid_type=orm.Dict, required=False,
            help='Result of birch murnaghan fitting.')
        spec.exit_code(400, 'ERROR_SUB_PROCESS_FAILED',
            message='At least one of the `{cls}` sub processes did not finish successfully.')
        spec.exit_code(500, 'ERROR_BIRCH_MURNAGHAN_FIT_FAILED',
            message='The birch murnaghan fit failed with exit code={code}.')
        # yapf: enable

    def get_scale_factors(self) -> List[float]:
        """Return the list of scale factors.
        The points are averagely distributed from the minimal volume.
        The scale equal to 1 will not returned.

        Return: list
        """
        if "scale_factors" in self.inputs:
            return self.inputs.scale_factors.get_list()

        count = self.inputs.scale_count.value
        increment = self.inputs.scale_increment.value
        return [
            round(1 + (2 * i - count + 1) / 2 * increment, 3)
            for i in range(count)
            if (2 * i - count + 1 != 0)
        ]

    def _get_inputs(self, scale_factor):
        """Inputs for pw calculation of every scale increment."""
        inputs = self.exposed_inputs(PwBaseWorkChain)

        # structure scaled may lead to kpoints mesh change if it is generated from kpoint distance,
        # therefore the kpoints mesh is fixed for all scaled structures and from the unscaled one.
        kpoints = orm.KpointsData()
        kpoints.set_cell_from_structure(self.inputs.structure)
        kpoints.set_kpoints_mesh_from_density(
            distance=self.inputs.kpoints_distance.value
        )

        inputs["metadata"] = {"call_link_label": "EOS_scf"}
        inputs["kpoints"] = kpoints
        inputs["pw"]["structure"] = scale_structure(
            self.inputs.structure, orm.Float(scale_factor)
        )
        inputs.pop("kpoints_distance", None)

        return inputs

    def run_init(self):
        """Run the first sub-workchain, if this failed the whole workchain break."""
        inputs = self._get_inputs(scale_factor=1)  # inputs for unscaled structure
        self.report(f"submitting precheck calculation for unscaled structure.")
        self.ctx.init_workchain = self.submit(PwBaseWorkChain, **inputs)
        self.to_context(children=append_(self.ctx.init_workchain))

    def run_eos(self):
        """Run the sub process at each scale factor to compute the structure volume and total energy."""
        workchain = self.ctx.children[0]

        if not workchain.is_finished_ok:
            self.logger.warning(
                f"PwBaseWorkChain pk={workchain.pk} for first scale structure run is failed."
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED.format(cls=PwBaseWorkChain)

        for scale_factor in self.get_scale_factors():
            inputs = self._get_inputs(scale_factor)
            self.report(f"submitting scale_factor=`{scale_factor}`")
            self.to_context(children=append_(self.submit(PwBaseWorkChain, **inputs)))

    def inspect_eos(self):
        """Inspect all children workflows to make sure they finished successfully."""
        if any(not child.is_finished_ok for child in self.ctx.children):
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED.format(cls=PwBaseWorkChain)

        volume_energy = {
            "num_of_atoms": sum(self.inputs.structure.get_composition().values()),
            "volume_unit": "A^3",
            "energy_unit": "eV",
        }
        volumes = []
        energies = []
        for child in self.ctx.children:
            # free energy E-TS (metal)
            energies.append(child.outputs.output_parameters["energy"])
            volumes.append(child.outputs.output_parameters["volume"])

        self.logger.info(f"volumes={volumes}, energies={energies}")

        volume_energy["volumes"] = volumes
        volume_energy["energies"] = energies

        output_volume_energy = orm.Dict(dict=volume_energy).store()
        self.out("output_volume_energy", output_volume_energy)

        output_birch_murnaghan_fit, node = run_get_node(
            birch_murnaghan_fit, output_volume_energy
        )

        if not node.is_finished_ok:
            self.logger.warning(f"The birch murnaghan fit failed for node pk={node.pk}")
            return self.exit_codes.ERROR_BIRCH_MURNAGHAN_FIT_FAILED.format(
                code=node.exit_status
            )
        else:
            self.report(
                f"The birch murnaghan fitting results are: {output_birch_murnaghan_fit.get_dict()}"
            )
            self.out("output_birch_murnaghan_fit", output_birch_murnaghan_fit)
