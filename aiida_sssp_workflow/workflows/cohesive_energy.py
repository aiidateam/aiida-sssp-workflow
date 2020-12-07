# -*- coding: utf-8 -*-
"""
A calcfunctian create_isolate_atom
Create the structure of isolate atom
"""
from aiida import orm
from aiida.engine import calcfunction, WorkChain
from aiida.common import AttributeDict
from aiida.plugins import WorkflowFactory

from aiida_sssp_workflow.utils import update_dict

PwBaseWorkflow = WorkflowFactory('quantumespresso.pw.base')


@calcfunction
def helper_analyze_cohesive_results(out_bulk_calc, out_atom_calc):
    bulk_energy = out_bulk_calc[
        'energy']  # Already the free energy E-TS (metal)
    num_of_atoms = out_bulk_calc['number_of_atoms']
    bulk_energy_per_atom = bulk_energy / num_of_atoms

    # TODO take care of the unphysical entropy caused by smearing, should be zero for isolated atom
    # TODO report if it is not zero
    assert out_atom_calc['number_of_atoms'] == 1
    atom_free_energy = out_atom_calc['energy']
    try:
        # calculation with smearing, aka the internal energy of pwscf
        atom_smearing_energy = out_atom_calc['energy_smearing']
        atom_energy = atom_free_energy - atom_smearing_energy
    except KeyError:
        atom_energy = atom_free_energy
        atom_smearing_energy = 'None'

    cohesive_energy = bulk_energy_per_atom - atom_energy

    return orm.Dict(
        dict={
            'cohesive_energy': cohesive_energy,
            'bulk_energy': bulk_energy,
            'bulk_energy_per_atom': bulk_energy_per_atom,
            'number_of_atoms_in_bluk': num_of_atoms,
            'isolate_atom_energy': atom_energy,
            'atom_smearing_energy': atom_smearing_energy,
            'energy_units': 'eV',
        })


@calcfunction
def create_isolate_atom(
    element: orm.Str, length=lambda: orm.Float(12.0)) -> orm.StructureData:
    """
    create a cubic cell with length and the element kind. The atom is locate at the origin point.
    """
    from ase import Atoms
    L = length.value  # pylint: disable=invalid-name
    atom = Atoms(symbols=element.value,
                 pbc=[True, True, True],
                 cell=[L, L, L],
                 positions=[(
                     0.,
                     0.,
                     0,
                 )])
    structure = orm.StructureData(ase=atom)
    return structure


@calcfunction
def helper_parse_upf(upf):
    return orm.Str(upf.element)


PW_PARAS = lambda: orm.Dict(dict={
    'SYSTEM': {
        'ecutrho': 1600,
        'ecutwfc': 200,
    },
})


class CohesiveEnergyWorkChain(WorkChain):
    """WorkChain to calculate cohisive energy of input structure"""
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input('code',
                   valid_type=orm.Code,
                   help='The `pw.x` code use for the `PwCalculation`.')
        spec.input('pseudo',
                   valid_type=orm.UpfData,
                   required=True,
                   help='Pseudopotential to be verified')
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
        spec.input('parameters.pw_bulk',
                   valid_type=orm.Dict,
                   default=PW_PARAS,
                   help='parameters for pwscf of bulk calculation.')
        spec.input('parameters.pw_atom',
                   valid_type=orm.Dict,
                   default=PW_PARAS,
                   help='parameters for pwscf of atom calculation.')
        spec.input(
            'parameters.kpoints_distance',
            valid_type=orm.Float,
            default=lambda: orm.Float(0.1),
            help='Kpoints distance setting for bulk energy calculation.')
        spec.input(
            'parameters.vacuum_length',
            valid_type=orm.Float,
            default=lambda: orm.Float(12.0),
            help='The length of cubic cell in isolate atom calculation.')
        spec.outline(
            cls.setup,
            cls.validate_structure,
            cls.run_energy,
            cls.inspect_energy,
            cls.results,
        )
        spec.output(
            'output_parameters',
            valid_type=orm.Dict,
            required=True,
            help=
            'The output parameters include cohesive energy of the structure.')
        spec.exit_code(
            211,
            'ERROR_SUB_PROCESS_FAILED_ATOM_ENERGY',
            message='PwBaseWorkChain of atom energy evaluation failed.')
        spec.exit_code(
            212,
            'ERROR_SUB_PROCESS_FAILED_BULK_ENERGY',
            message=
            'PwBaseWorkChain of bulk structure energy evaluation failed with exit status.'
        )

    def setup(self):
        """Input validation"""
        bulk_parameters = {
            'SYSTEM': {
                'degauss': 0.02,
                'occupations': 'smearing',
                'smearing': 'marzari-vanderbilt',
            },
            'ELECTRONS': {
                'conv_thr': 1e-10,
            },
        }
        atom_parameters = {
            'SYSTEM': {
                'degauss': 0.02,
                'occupations': 'smearing',
                'smearing': 'gaussian',
            },
            'ELECTRONS': {
                'conv_thr': 1e-10,
            },
        }
        pw_bulk_parameters = update_dict(
            bulk_parameters, self.inputs.parameters.pw_bulk.get_dict())
        pw_atom_parameters = update_dict(
            atom_parameters, self.inputs.parameters.pw_atom.get_dict())
        self.ctx.pw_bulk_parameters = orm.Dict(dict=pw_bulk_parameters)
        self.ctx.pw_atom_parameters = orm.Dict(dict=pw_atom_parameters)

        self.ctx.kpoints_distance = self.inputs.parameters.kpoints_distance

    def validate_structure(self):
        """Create isolate atom and validate structure"""
        # create isolate atom structure
        self.ctx.element = helper_parse_upf(self.inputs.pseudo)
        self.ctx.pseudos = {self.ctx.element.value: self.inputs.pseudo}
        self.ctx.atom_structure = create_isolate_atom(
            self.ctx.element, self.inputs.parameters.vacuum_length)

    def run_energy(self):
        """set the inputs and submit atom/bulk energy evaluation parallel"""
        # TODO tstress to cache the calculation for pressure convergence
        bulk_inputs = AttributeDict({
            'metadata': {
                'call_link_label': 'bulk_scf'
            },
            'pw': {
                'structure': self.inputs.structure,
                'code': self.inputs.code,
                'pseudos': self.ctx.pseudos,
                'parameters': self.ctx.pw_bulk_parameters,
                'settings': orm.Dict(dict={'CMDLINE': ['-ndiag', '1']}),
                'metadata': {},
            },
            'kpoints_distance':
            self.ctx.kpoints_distance,
        })

        atom_kpoints = orm.KpointsData()
        atom_kpoints.set_kpoints_mesh([1, 1, 1])
        atom_inputs = AttributeDict({
            'metadata': {
                'call_link_label': 'atom_scf'
            },
            'pw': {
                'structure': self.ctx.atom_structure,
                'code': self.inputs.code,
                'pseudos': self.ctx.pseudos,
                'parameters': self.ctx.pw_atom_parameters,
                'metadata': {},
            },
            'kpoints': atom_kpoints,
        })

        if 'options' in self.inputs:
            options = self.inputs.options.get_dict()
        else:
            from aiida_quantumespresso.utils.resources import get_default_options

            options = get_default_options(with_mpi=True)

        bulk_inputs.pw.metadata.options = options
        atom_inputs.pw.metadata.options = options

        running_bulk_energy = self.submit(PwBaseWorkflow, **bulk_inputs)
        running_atom_energy = self.submit(PwBaseWorkflow, **atom_inputs)
        self.to_context(
            **{
                'workchain_bulk_energy': running_bulk_energy,
                'workchain_atom_energy': running_atom_energy,
            })

    def inspect_energy(self):
        """inspect the result of energy calculation."""
        workchain_atom_energy = self.ctx['workchain_atom_energy']
        workchain_bulk_energy = self.ctx['workchain_bulk_energy']

        if not workchain_atom_energy.is_finished_ok:
            self.report(
                f'PwBaseWorkChain of atom energy evaluation failed with exit status {workchain_atom_energy.exit_status}'
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_ATOM_ENERGY
        if not workchain_bulk_energy.is_finished_ok:
            self.report(
                f'PwBaseWorkChain of bulk energy evaluation failed with exit status {workchain_atom_energy.exit_status}'
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_BULK_ENERGY

        out_bulk_calc = workchain_bulk_energy.outputs.output_parameters
        out_atom_calc = workchain_atom_energy.outputs.output_parameters

        output_parameters = helper_analyze_cohesive_results(
            out_bulk_calc, out_atom_calc)

        self.report(f'output_parameters node<{output_parameters.pk}>')
        self.out('output_parameters', output_parameters)

    def results(self):
        pass
