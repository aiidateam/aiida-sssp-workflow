# -*- coding: utf-8 -*-
"""
A calcfunctian create_isolate_atom
Create the structure of isolate atom
"""
from aiida import orm
from aiida.engine import calcfunction, WorkChain, append_
from aiida.common import AttributeDict
from aiida.plugins import WorkflowFactory

from aiida_sssp_workflow.utils import helper_parse_upf, update_dict, RARE_EARTH_ELEMENTS

PwBaseWorkflow = WorkflowFactory('quantumespresso.pw.base')


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
                     L / 2,
                     L / 2,
                     L / 2,
                 )])
    structure = orm.StructureData(ase=atom)
    return structure


PW_PARAS = lambda: orm.Dict(dict={
    'SYSTEM': {
        'ecutrho': 1600,
        'ecutwfc': 200,
    },
})


class CohesiveEnergyWorkChain(WorkChain):
    """WorkChain to calculate cohisive energy of input structure"""

    _BULK_PARAMETERS = {
        'SYSTEM': {
            'degauss': 0.02,
            'occupations': 'smearing',
            'smearing': 'marzari-vanderbilt',
        },
        'ELECTRONS': {
            'conv_thr': 1e-10,
        },
        'CONTROL': {
            'calculation': 'scf',
            'wf_collect': True,
            'tstress': True,
        },
    }

    _ATOM_PARAMETERS = {
        'SYSTEM': {
            'degauss': 0.02,
            'occupations': 'smearing',
            'smearing': 'gaussian',
        },
        'ELECTRONS': {
            'conv_thr': 1e-10,
        },
    }

    _BULK_CMDLINE_SETTING = {'CMDLINE': ['-ndiag', '1', '-nk', '4']}
    _ATOM_CMDLINE_SETTING = {
        'CMDLINE': ['-ndiag', '1']
    }  # TODO use the same one above

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
            required=False,
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
        bulk_parameters = self._BULK_PARAMETERS
        atom_parameters = self._ATOM_PARAMETERS
        pw_bulk_parameters = update_dict(
            bulk_parameters, self.inputs.parameters.pw_bulk.get_dict())
        pw_atom_parameters = update_dict(
            atom_parameters, self.inputs.parameters.pw_atom.get_dict())

        if self.inputs.parameters.ecutwfc and self.inputs.parameters.ecutrho:
            parameters = {
                'SYSTEM': {
                    'ecutwfc': self.inputs.parameters.ecutwfc,
                    'ecutrho': self.inputs.parameters.ecutrho,
                },
            }
            pw_bulk_parameters = update_dict(pw_bulk_parameters, parameters)
            pw_atom_parameters = update_dict(pw_atom_parameters, parameters)

        self.ctx.pw_bulk_parameters = pw_bulk_parameters
        self.ctx.pw_atom_parameters = pw_atom_parameters

        self.ctx.kpoints_distance = self.inputs.parameters.kpoints_distance

    def validate_structure(self):
        """Create isolate atom and validate structure"""
        # create isolate atom structure
        elements = self.inputs.structure.get_kind_names()
        formula_list = self.inputs.structure.get_ase().get_chemical_symbols()
        element_number = {
            element: formula_list.count(element)
            for element in formula_list
        }

        assert len(self.inputs.pseudos) == len(element_number)
        element_structure = {}
        for element in elements:
            atom_structure = create_isolate_atom(
                orm.Str(element), self.inputs.parameters.vacuum_length)
            element_structure[element] = atom_structure

        self.ctx.pseudos = self.inputs.pseudos
        self.ctx.element_structure = element_structure
        self.ctx.element_number = element_number

    def run_energy(self):
        """set the inputs and submit atom/bulk energy evaluation parallel"""
        bulk_inputs = AttributeDict({
            'metadata': {
                'call_link_label': 'bulk_scf'
            },
            'pw': {
                'structure': self.inputs.structure,
                'code': self.inputs.code,
                'pseudos': self.ctx.pseudos,
                'parameters': orm.Dict(dict=self.ctx.pw_bulk_parameters),
                'settings': orm.Dict(dict=self._BULK_CMDLINE_SETTING),
                'metadata': {},
            },
            'kpoints_distance':
            self.ctx.kpoints_distance,
        })

        if 'options' in self.inputs:
            options = self.inputs.options.get_dict()
        else:
            from aiida_quantumespresso.utils.resources import get_default_options

            options = get_default_options(with_mpi=True)

        bulk_inputs.pw.metadata.options = options

        running_bulk_energy = self.submit(PwBaseWorkflow, **bulk_inputs)
        self.to_context(workchain_bulk_energy=running_bulk_energy)

        for element, pseudo in self.ctx.pseudos.items():
            atom_kpoints = orm.KpointsData()
            atom_kpoints.set_kpoints_mesh([1, 1, 1])

            header = helper_parse_upf(pseudo)
            z_valence = header['z_valence']

            parameters = {}
            if element in RARE_EARTH_ELEMENTS:
                # And might be a small mixing is needed for electrons convergence
                parameters = {
                    'SYSTEM': {
                        'nbnd': int(z_valence * 3),
                    }
                }

            atom_pw_parameters = update_dict(self.ctx.pw_atom_parameters,
                                             parameters)

            atom_inputs = AttributeDict({
                'metadata': {
                    'call_link_label': 'atom_scf'
                },
                'pw': {
                    'structure': self.ctx.element_structure[element],
                    'code': self.inputs.code,
                    'pseudos': {
                        element: pseudo
                    },
                    'parameters': orm.Dict(dict=atom_pw_parameters),
                    'settings': orm.Dict(dict=self._ATOM_CMDLINE_SETTING),
                    'metadata': {},
                },
                'kpoints': atom_kpoints,
            })
            atom_inputs.pw.metadata.options = options

            # TODO n_machine = 4 mandatory for lanthanides.
            if self.inputs.parameters.ecutwfc.value > 100:
                self.report(
                    'High ecutwfc may require more RAM. Simply increase the node.'
                )
                atom_inputs.pw.metadata.options['resources'][
                    'num_machines'] = 2

            running_atom_energy = self.submit(PwBaseWorkflow, **atom_inputs)
            self.to_context(
                workchain_atom_children=append_(running_atom_energy))

    def inspect_energy(self):
        """inspect the result of energy calculation."""

        workchain_bulk_energy = self.ctx['workchain_bulk_energy']
        if not workchain_bulk_energy.is_finished_ok:
            self.report(
                f'PwBaseWorkChain of bulk energy evaluation failed'
                f' with exit status {workchain_bulk_energy.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_BULK_ENERGY

        self.ctx.bulk_energy = workchain_bulk_energy.outputs.output_parameters[
            'energy']

        element_energy = {}
        for child in self.ctx.workchain_atom_children:
            element = child.inputs.pw__structure.get_kind_names()[0]
            if not child.is_finished_ok:
                self.report(
                    f'PwBaseWorkChain of element={element} atom energy evaluation failed'
                    f' with exit status {child.exit_status}')
                return self.exit_codes.ERROR_SUB_PROCESS_FAILED_ATOM_ENERGY

            output_parameters = child.outputs.output_parameters

            atom_free_energy = output_parameters['energy']
            atom_smearing_energy = output_parameters['energy_smearing']
            atom_energy = atom_free_energy - atom_smearing_energy
            element_energy[element] = atom_energy

        self.ctx.element_energy = element_energy

    def results(self):
        num_of_atoms = sum(self.ctx.element_number.values())
        cohesive_energy = self.ctx.bulk_energy
        element_energy = {
        }  # dict to be output for every element isolate energy
        for element, energy in self.ctx.element_energy.items():
            element_energy[f'isolate_atom_energy_{element}'] = energy
            cohesive_energy -= energy * self.ctx.element_number[element]

        cohesive_energy_per_atom = cohesive_energy / num_of_atoms

        parameters_dict = {
            'cohesive_energy': cohesive_energy,
            'cohesive_energy_per_atom': cohesive_energy_per_atom,
            'bulk_energy': self.ctx.bulk_energy,
            'structure_formula': self.inputs.structure.get_formula(),
            'energy_unit': 'eV',
            'energy_per_atom_unit': 'eV/atom',
        }
        parameters_dict.update(element_energy)
        output_parameters = orm.Dict(dict=parameters_dict)

        self.out('output_parameters', output_parameters.store())
        self.report(f'output_parameters node<{output_parameters.pk}>')
