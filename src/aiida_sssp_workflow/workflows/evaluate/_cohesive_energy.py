# -*- coding: utf-8 -*-
"""
A calcfunctian create_isolate_atom
Create the structure of isolate atom
"""

from aiida import orm
from aiida.engine import (
    CalcJob,
    ProcessHandlerReport,
    append_,
    calcfunction,
    process_handler,
)
from aiida.plugins import DataFactory
from aiida_quantumespresso.common.types import RestartType
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain

from . import _BaseEvaluateWorkChain

UpfData = DataFactory("pseudo.upf")


@calcfunction
def create_isolate_atom(
    element: orm.Str, length=lambda: orm.Float(12.0)
) -> orm.StructureData:
    """
    create a cubic cell with length and the element kind. The atom is locate at the origin point.
    """
    from ase import Atoms

    L = length.value  # pylint: disable=invalid-name
    atom = Atoms(
        symbols=element.value,
        pbc=[True, True, True],
        cell=[L, L, L],
        positions=[
            (
                L / 2,
                L / 2,
                L / 2,
            )
        ],
    )
    structure = orm.StructureData(ase=atom)
    return structure


class PwBaseWorkChainWithMemoryHandler(PwBaseWorkChain):
    """Add memory handler to PwBaseWorkChain to use large memory resource"""

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input(
            "pw_code_large_memory",
            valid_type=orm.Code,
            required=False,
            help="The `pw.x` code use for the `PwCalculation` with large memory resource.",
        )

    @process_handler(
        priority=601,
        exit_codes=[
            CalcJob.exit_codes.ERROR_SCHEDULER_OUT_OF_MEMORY,
        ],
    )
    def handle_out_of_memory(self, calculation):
        """Handle out of memory error by using the code with large memory resource if provided"""
        if "pw_code_large_memory" in self.inputs:
            # use code with large memory resource
            pw_code_large_memory = self.inputs.pw_code_large_memory
            self.ctx.inputs.code = pw_code_large_memory

            action = f"Use code {self.inputs.pw_code_large_memory} with large memory resource"

            self.set_restart_type(RestartType.FROM_SCRATCH)
            self.report_error_handled(calculation, action)
            return ProcessHandlerReport(True)
        else:
            self.ctx.current_num_machines = self.ctx.inputs.metadata.options.get(
                "resources", {}
            ).get("num_machines", 1)

            if self.ctx.current_num_machines > 4:
                self.report(
                    "The number of machines is larger than 4, the calculation will be terminated."
                )
                return ProcessHandlerReport(
                    False, CalcJob.exit_codes.ERROR_SCHEDULER_OUT_OF_MEMORY
                )

            action = f"Increase the number of machines from {self.ctx.current_num_machines} to {self.ctx.current_num_machines + 1}"
            self.ctx.inputs.metadata.options["resources"]["num_machines"] = (
                self.ctx.current_num_machines + 1
            )
            # for atomic calculation, the num_mpiprocs_per_machine is set, but increase the number of machines
            # will cause too many mpi processes, so pop the num_mpiprocs_per_machine and use the `tot_num_mpiprocs`.
            num_mpiprocs_per_machine = self.ctx.inputs.metadata.options[
                "resources"
            ].pop("num_mpiprocs_per_machine")
            if num_mpiprocs_per_machine:
                self.ctx.inputs.metadata.options["resources"]["tot_num_mpiprocs"] = (
                    num_mpiprocs_per_machine
                )

            self.set_restart_type(RestartType.FROM_SCRATCH)
            self.report_error_handled(calculation, action)

            return ProcessHandlerReport(True)


class CohesiveEnergyWorkChain(_BaseEvaluateWorkChain):
    """WorkChain to calculate cohisive energy of input structure"""

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        # yapf: disable
        super().define(spec)
        spec.input('structure', valid_type=orm.StructureData,
                    help='Ground state structure which the verification perform')
        spec.input_namespace('pseudos', valid_type=UpfData, dynamic=True,
                    help='A mapping of `UpfData` nodes onto the kind name to which they should apply.')
        spec.input('vacuum_length', valid_type=orm.Float,
                    help='The length of cubic cell in isolate atom calculation.')
        spec.expose_inputs(PwBaseWorkChain, namespace="bulk", exclude=["pw.structure", "pw.pseudos"])
        spec.expose_inputs(PwBaseWorkChainWithMemoryHandler, namespace="atom", exclude=["pw.structure", "pw.pseudos"])

        spec.outline(
            cls.validate_structure,
            cls.run_energy,
            cls.inspect_energy,
            cls.finalize,
        )

        spec.output('output_parameters', valid_type=orm.Dict, required=True,
                    help='The output parameters include cohesive energy of the structure.')
        spec.exit_code(211, 'ERROR_SUB_PROCESS_FAILED_ATOM_ENERGY',
                    message='PwBaseWorkChain of atom energy evaluation failed.')
        spec.exit_code(212, 'ERROR_SUB_PROCESS_FAILED_BULK_ENERGY',
                    message='PwBaseWorkChain of bulk structure energy evaluation failed with exit status.')
        # yapf: enable

    def validate_structure(self):
        """Create isolate atom and validate structure"""
        # create isolate atom structure
        formula_list = self.inputs.structure.get_ase().get_chemical_symbols()
        elements = list(dict.fromkeys(formula_list))
        dict_element_and_count = {
            element: formula_list.count(element) for element in formula_list
        }

        # assert len(self.inputs.pseudos) == len(dict_element_and_count)
        dict_element_and_structure = {}
        for element in elements:
            atom_structure = create_isolate_atom(
                orm.Str(element), self.inputs.vacuum_length
            )
            dict_element_and_structure[element] = atom_structure

        self.ctx.pseudos = self.inputs.pseudos
        self.ctx.d_element_structure = dict_element_and_structure
        self.ctx.d_element_count = dict_element_and_count

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
            key = f"{element}1"
            pseudo = pseudos[key]

        return pseudo

    def run_energy(self):
        """set the inputs and submit atom/bulk energy evaluation parallel"""
        bulk_inputs = self.exposed_inputs(PwBaseWorkChain, namespace="bulk")
        bulk_inputs["pw"]["structure"] = self.inputs.structure
        bulk_inputs["pw"]["pseudos"] = self.inputs.pseudos

        running_bulk_energy = self.submit(PwBaseWorkChain, **bulk_inputs)
        self.report(
            f"Submit SCF calculation of bulk {self.inputs.structure.get_description()}"
        )
        self.to_context(workchain_bulk_energy=running_bulk_energy)

        for element, structure in self.ctx.d_element_structure.items():
            atom_inputs = self.exposed_inputs(
                PwBaseWorkChainWithMemoryHandler, namespace="atom"
            )
            atom_inputs["pw"]["structure"] = structure
            atom_inputs["pw"]["pseudos"] = {
                element: self._get_pseudo(element, self.inputs.pseudos),
            }

            running_atom_energy = self.submit(
                PwBaseWorkChainWithMemoryHandler, **atom_inputs
            )
            self.logger.info(f"Submit atomic SCF of {element}.")
            self.to_context(workchain_atom_children=append_(running_atom_energy))

    def inspect_energy(self):
        """inspect the result of energy calculation."""

        workchain_bulk_energy = self.ctx["workchain_bulk_energy"]
        if not workchain_bulk_energy.is_finished_ok:
            self.logger.warning(
                f"PwBaseWorkChain of bulk energy evaluation failed"
                f" with exit status {workchain_bulk_energy.exit_status}"
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_BULK_ENERGY

        self._disable_cache(workchain_bulk_energy)

        self.ctx.bulk_energy = workchain_bulk_energy.outputs.output_parameters["energy"]
        calc_time = workchain_bulk_energy.outputs.output_parameters["wall_time_seconds"]

        self.ctx.ecutwfc = workchain_bulk_energy.inputs.pw.parameters["SYSTEM"][
            "ecutwfc"
        ]
        self.ctx.ecutrho = workchain_bulk_energy.inputs.pw.parameters["SYSTEM"][
            "ecutrho"
        ]

        element_energy = {}
        for child in self.ctx.workchain_atom_children:
            element = child.inputs.pw.structure.get_kind_names()[0]
            if not child.is_finished_ok:
                self.logger.warning(
                    f"PwBaseWorkChain of element={element} atom energy evaluation failed"
                    f" with exit status {child.exit_status}"
                )
                return self.exit_codes.ERROR_SUB_PROCESS_FAILED_ATOM_ENERGY

            output_parameters = child.outputs.output_parameters

            atom_free_energy = output_parameters["energy"]
            atom_smearing_energy = output_parameters["energy_smearing"]
            atom_energy = atom_free_energy - atom_smearing_energy
            element_energy[element] = atom_energy
            calc_time += output_parameters["wall_time_seconds"]

        self.ctx.calc_time = calc_time
        self.ctx.element_energy = element_energy

    def finalize(self):
        num_of_atoms = sum(self.ctx.d_element_count.values())
        cohesive_energy = self.ctx.bulk_energy
        element_energy = {}  # dict to be output for every element isolate energy
        for element, energy in self.ctx.element_energy.items():
            element_energy[f"isolate_atom_energy_{element}"] = energy
            cohesive_energy -= energy * self.ctx.d_element_count[element]

        cohesive_energy_per_atom = cohesive_energy / num_of_atoms

        parameters_dict = {
            "cohesive_energy": cohesive_energy,
            "cohesive_energy_per_atom": cohesive_energy_per_atom,
            "bulk_energy": self.ctx.bulk_energy,
            "structure_formula": self.inputs.structure.get_formula(),
            "energy_unit": "eV",
            "energy_per_atom_unit": "eV/atom",
            "total_calc_time": self.ctx.calc_time,
            "time_unit": "s",
        }
        parameters_dict.update(element_energy)
        output_parameters = orm.Dict(dict=parameters_dict)

        self.out("output_parameters", output_parameters.store())
