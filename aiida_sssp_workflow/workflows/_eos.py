# -*- coding: utf-8 -*-
"""Equation of state workflow that can use any code plugin implementing the common relax workflow."""
from aiida import orm
from aiida.common import AttributeDict
from aiida.engine import WorkChain, append_, calcfunction
from aiida.plugins import WorkflowFactory, CalculationFactory

PwBaseWorkChain = WorkflowFactory('quantumespresso.pw.base')
birch_murnaghan_fit = CalculationFactory('sssp_workflow.birch_murnaghan_fit')


def validate_inputs(value, _):
    """Validate the entire input namespace."""
    if 'scale_factors' not in value and ('scale_count' not in value
                                         and 'scale_count' not in value):
        return 'neither `scale_factors` nor the pair of `scale_count` and `scale_increment` were defined.'


def validate_scale_factors(value, _):
    """Validate the `validate_scale_factors` input."""
    if value and len(value) < 3:
        return 'need at least 3 scaling factors.'


def validate_scale_count(value, _):
    """Validate the `scale_count` input."""
    if value is not None and value < 3:
        return 'need at least 3 scaling factors.'


def validate_scale_increment(value, _):
    """Validate the `scale_increment` input."""
    if value is not None and not 0 < value < 1:
        return 'scale increment needs to be between 0 and 1.'


@calcfunction
def scale_structure(structure: orm.StructureData,
                    scale_factor: orm.Float) -> orm.StructureData:
    """Scale the structure with the given scaling factor."""
    ase = structure.get_ase().copy()
    ase.set_cell(ase.get_cell() * float(scale_factor)**(1 / 3),
                 scale_atoms=True)
    return orm.StructureData(ase=ase)


class _EquationOfStateWorkChain(WorkChain):
    """Workflow to compute the equation of state for a given crystal structure."""
    @classmethod
    def define(cls, spec):
        # yapf: disable
        super().define(spec)
        spec.expose_inputs(PwBaseWorkChain, namespace='scf',
            exclude=('clean_workdir', 'pw.structure', 'pw.kpoints', 'pw.kpoints_distance'),
            namespace_options={'help': 'Inputs for the `PwBaseWorkChain` for the SCF calculation.'})
        spec.input('structure', valid_type=orm.StructureData, help='The structure at equilibrium volume.')
        spec.input('kpoints_distance', valid_type=orm.Float, required=True,
                   help='The kpoints distance used in generating the kmesh of '
                        'unscaled structure then for all scaled structures')
        spec.input('scale_factors', valid_type=orm.List, required=False, validator=validate_scale_factors,
            help='The list of scale factors at which the volume and total energy of the structure should be computed.')
        spec.input('scale_count', valid_type=orm.Int, default=lambda: orm.Int(7), validator=validate_scale_count,
            help='The number of points to compute for the equation of state.')
        spec.input('scale_increment', valid_type=orm.Float, default=lambda: orm.Float(0.02),
            validator=validate_scale_increment,
            help='The relative difference between consecutive scaling factors.')
        spec.inputs.validator = validate_inputs
        spec.outline(
            cls.run_init,
            cls.run_eos,
            cls.inspect_eos,
        )
        spec.output('output_volume_energy', valid_type=orm.Dict,
                    help='Results volumes and energise.')
        spec.output('output_birch_murnaghan_fit', valid_type=orm.Dict,
                    help='Result of birch murnaghan fitting.')
        spec.exit_code(400, 'ERROR_SUB_PROCESS_FAILED',
            message='At least one of the `{cls}` sub processes did not finish successfully.')
        # yapf: enable

    def get_scale_factors(self):
        """Return the list of scale factors."""
        if 'scale_factors' in self.inputs:
            return self.inputs.scale_factors

        count = self.inputs.scale_count.value
        increment = self.inputs.scale_increment.value
        return [
            orm.Float(1 + i * increment - (count - 1) * increment / 2)
            for i in range(count)
        ]

    def get_sub_workchain_builder(self, scale_factor):
        """Return the builder for the relax workchain."""
        process_class = PwBaseWorkChain
        structure = scale_structure(self.inputs.structure, scale_factor)
        unscaled_structure = self.inputs.structure

        inputs = AttributeDict(
            self.exposed_inputs(PwBaseWorkChain, namespace='scf'))

        parameters = inputs.pw.parameters.get_dict()
        parameters.setdefault('CONTROL', {})['calculation'] = 'scf'

        kpoints = orm.KpointsData()
        kpoints.set_cell_from_structure(unscaled_structure)
        kpoints.set_kpoints_mesh_from_density(
            distance=self.inputs.kpoints_distance.value)

        inputs.metadata.call_link_label = 'scf'
        inputs.kpoints = kpoints.store()
        inputs.pw.structure = structure
        inputs.pw.parameters = orm.Dict(dict=parameters)
        if 'options' in self.inputs:
            inputs.pw.metadata.options = self.inputs.options.get_dict()
        else:
            from aiida_sssp_workflow.utils import get_default_options

            inputs.pw.metadata.options = get_default_options(with_mpi=True)

        builder = process_class.get_builder()

        builder.update(**inputs)

        return builder

    def run_init(self):
        """Run the first sub-workchain, if this failed the whole workchain break."""
        scale_factor = self.get_scale_factors()[0]
        builder = self.get_sub_workchain_builder(scale_factor)
        self.report(
            f'submitting `{builder.process_class.__name__}` for scale_factor `{scale_factor}`'
        )
        self.ctx.previous_workchain = self.submit(builder)
        self.to_context(children=append_(self.ctx.previous_workchain))

    def run_eos(self):
        """Run the sub process at each scale factor to compute the structure volume and total energy."""
        workchain = self.ctx.children[0]

        if not workchain.is_finished_ok:
            self.report(
                f'PwBaseWorkChain pk={workchain.pk} for first scale structure run is failed.'
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED.format(
                cls=PwBaseWorkChain)

        for scale_factor in self.get_scale_factors()[1:]:
            builder = self.get_sub_workchain_builder(scale_factor)
            self.report(
                f'submitting `{builder.process_class.__name__}` for scale_factor `{scale_factor}`'
            )
            self.to_context(children=append_(self.submit(builder)))

    def inspect_eos(self):
        """Inspect all children workflows to make sure they finished successfully."""
        if any(not child.is_finished_ok for child in self.ctx.children):
            process_class = PwBaseWorkChain
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED.format(
                cls=process_class)

        volume_energy = {
            'volumes': {},
            'energies': {},
            'volume_unit': 'A^3/atom',
            'energy_unit': 'eV/atom',
        }
        for index, child in enumerate(self.ctx.children):
            volume = child.outputs.output_parameters['volume']
            energy = child.outputs.output_parameters[
                'energy']  # Already the free energy E-TS (metal)
            num_of_atoms = child.outputs.output_parameters['number_of_atoms']
            self.report(
                f'Image {index}: volume={volume}, total energy={energy}')
            volume_energy['volumes'][index] = volume / num_of_atoms
            volume_energy['energies'][index] = energy / num_of_atoms

        output_volume_energy = orm.Dict(dict=volume_energy).store()
        output_birch_murnaghan_fit = birch_murnaghan_fit(output_volume_energy)

        self.report(
            f'The birch murnaghan fitting results are: {output_birch_murnaghan_fit.get_dict()}'
        )

        self.out('output_volume_energy', output_volume_energy)
        self.out('output_birch_murnaghan_fit', output_birch_murnaghan_fit)
