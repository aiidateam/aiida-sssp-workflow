# -*- coding: utf-8 -*-
"""
A calcfunctian create_isolate_atom
Create the structure of isolate atom
"""
from aiida import orm
from aiida.engine import calcfunction, WorkChain, append_
from aiida.plugins import WorkflowFactory, DataFactory

from aiida_sssp_workflow.utils import update_dict

PwBaseWorkflow = WorkflowFactory('quantumespresso.pw.base')
UpfData = DataFactory('pseudo.upf')


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


class CohesiveEnergyWorkChain(WorkChain):
    """WorkChain to calculate cohisive energy of input structure"""
    _MAX_WALLCLOCK_SECONDS = 3600

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
        spec.input('bulk_parameters', valid_type=orm.Dict,
                    help='parameters for pwscf of bulk calculation.')
        spec.input('atom_parameters', valid_type=orm.Dict,
                    help='parameters for pwscf of atom calculation.')
        spec.input('ecutwfc', valid_type=orm.Float,
                    help='The ecutwfc set for both atom and bulk calculation. Please also set ecutrho if ecutwfc is set.')
        spec.input('ecutrho', valid_type=orm.Float,
                    help='The ecutrho set for both atom and bulk calculation.  Please also set ecutwfc if ecutrho is set.')
        spec.input('kpoints_distance', valid_type=orm.Float,
                    help='Kpoints distance setting for bulk energy calculation.')
        spec.input('vacuum_length', valid_type=orm.Float,
                    help='The length of cubic cell in isolate atom calculation.')
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
            cls.run_energy,
            cls.inspect_energy,
            cls.results,
        )
        spec.output('output_parameters', valid_type=orm.Dict, required=True,
                    help='The output parameters include cohesive energy of the structure.')
        spec.exit_code(211, 'ERROR_SUB_PROCESS_FAILED_ATOM_ENERGY',
                    message='PwBaseWorkChain of atom energy evaluation failed.')
        spec.exit_code(212, 'ERROR_SUB_PROCESS_FAILED_BULK_ENERGY',
                    message='PwBaseWorkChain of bulk structure energy evaluation failed with exit status.')
        # yapf: enable

    def setup_base_parameters(self):
        """Input validation"""
        bulk_parameters = self.inputs.bulk_parameters.get_dict()
        atom_parameters = self.inputs.atom_parameters.get_dict()

        parameters = {
            'SYSTEM': {
                'ecutwfc': self.inputs.ecutwfc,
                'ecutrho': self.inputs.ecutrho,
            },
        }
        bulk_parameters = update_dict(bulk_parameters, parameters)
        atom_parameters = update_dict(atom_parameters, parameters)

        self.ctx.bulk_parameters = bulk_parameters
        self.ctx.atom_parameters = atom_parameters

        self.ctx.kpoints_distance = self.inputs.kpoints_distance

    def validate_structure(self):
        """Create isolate atom and validate structure"""
        # create isolate atom structure
        formula_list = self.inputs.structure.get_ase().get_chemical_symbols()
        elements = list(dict.fromkeys(formula_list))
        dict_element_and_count = {
            element: formula_list.count(element)
            for element in formula_list
        }

        # assert len(self.inputs.pseudos) == len(dict_element_and_count)
        dict_element_and_structure = {}
        for element in elements:
            atom_structure = create_isolate_atom(orm.Str(element),
                                                 self.inputs.vacuum_length)
            dict_element_and_structure[element] = atom_structure

        self.ctx.pseudos = self.inputs.pseudos
        self.ctx.d_element_structure = dict_element_and_structure
        self.ctx.d_element_count = dict_element_and_count

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

    @staticmethod
    def _get_pseudo(element, pseudos):
        """
        get the pseudo by element from input pseudos dict
        the tricky is for the element name with number like in mag structure
        the pseudo get from the first found.
        """
        try:
            pseudo = pseudos[element]
        except KeyError:
            key = f'{element}1'
            pseudo = pseudos[key]

        return pseudo

    def run_energy(self):
        """set the inputs and submit atom/bulk energy evaluation parallel"""
        bulk_inputs = {
            'metadata': {
                'call_link_label': 'bulk_scf'
            },
            'pw': {
                'structure': self.inputs.structure,
                'code': self.inputs.code,
                'pseudos': self.ctx.pseudos,
                'parameters': orm.Dict(dict=self.ctx.bulk_parameters),
                'metadata': {
                    'options': self.ctx.options,
                },
                'parallelization': orm.Dict(dict=self.ctx.parallelization),
            },
            'kpoints_distance': self.ctx.kpoints_distance,
        }

        running_bulk_energy = self.submit(PwBaseWorkflow, **bulk_inputs)
        self.report(
            f'Submit SCF calculation of bulk {self.inputs.structure.get_description()}'
        )
        self.to_context(workchain_bulk_energy=running_bulk_energy)

        for element, structure in self.ctx.d_element_structure.items():
            atom_kpoints = orm.KpointsData()
            atom_kpoints.set_kpoints_mesh([1, 1, 1])

            atom_inputs = {
                'metadata': {
                    'call_link_label': 'atom_scf'
                },
                'pw': {
                    'structure': structure,
                    'code': self.inputs.code,
                    'pseudos': {
                        element: self._get_pseudo(element,
                                                  self.inputs.pseudos),
                    },
                    'parameters': orm.Dict(dict=self.ctx.atom_parameters),
                    'metadata': {
                        'options': self.ctx.options,
                    },
                    'parallelization': orm.Dict(dict=self.ctx.parallelization),
                },
                'kpoints': atom_kpoints,
            }

            # resources = {
            #     'num_machines': 1,
            #     'num_mpiprocs_per_machine': 1,
            # }
            # atom_inputs['pw']['metadata']['options'].setdefault(
            #     'resources', resources)

            running_atom_energy = self.submit(PwBaseWorkflow, **atom_inputs)
            self.report(f'Submit atomic SCF of {element}.')
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
        calc_time = workchain_bulk_energy.outputs.output_parameters[
            'wall_time_seconds']

        element_energy = {}
        for child in self.ctx.workchain_atom_children:
            element = child.inputs.pw.structure.get_kind_names()[0]
            if not child.is_finished_ok and child.exit_status < 700:
                # exit_status > 700 for all the warnings
                self.report(
                    f'PwBaseWorkChain of element={element} atom energy evaluation failed'
                    f' with exit status {child.exit_status}')
                return self.exit_codes.ERROR_SUB_PROCESS_FAILED_ATOM_ENERGY

            if child.exit_status > 700:
                self.report(
                    f'atom calculation [{child.pk}] finished[{child.exit_status}]: '
                    f'{child.exit_message}')

            output_parameters = child.outputs.output_parameters

            atom_free_energy = output_parameters['energy']
            atom_smearing_energy = output_parameters['energy_smearing']
            atom_energy = atom_free_energy - atom_smearing_energy
            element_energy[element] = atom_energy
            calc_time += output_parameters['wall_time_seconds']

        self.ctx.calc_time = calc_time
        self.ctx.element_energy = element_energy

    def results(self):
        """result"""
        num_of_atoms = sum(self.ctx.d_element_count.values())
        cohesive_energy = self.ctx.bulk_energy
        element_energy = {
        }  # dict to be output for every element isolate energy
        for element, energy in self.ctx.element_energy.items():
            element_energy[f'isolate_atom_energy_{element}'] = energy
            cohesive_energy -= energy * self.ctx.d_element_count[element]

        cohesive_energy_per_atom = cohesive_energy / num_of_atoms

        parameters_dict = {
            'cohesive_energy': cohesive_energy,
            'cohesive_energy_per_atom': cohesive_energy_per_atom,
            'bulk_energy': self.ctx.bulk_energy,
            'structure_formula': self.inputs.structure.get_formula(),
            'energy_unit': 'eV',
            'energy_per_atom_unit': 'eV/atom',
            'total_calc_time': self.ctx.calc_time,
            'time_unit': 's',
        }
        parameters_dict.update(element_energy)
        output_parameters = orm.Dict(dict=parameters_dict)

        self.out('output_parameters', output_parameters.store())
        self.report(
            f'output_parameters node<{output_parameters.pk}> with: {output_parameters.get_dict()}'
        )

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
