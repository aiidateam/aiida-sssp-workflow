# -*- coding: utf-8 -*-
"""
Convergence test on cohesive energy of a given pseudopotential
"""

from aiida import orm
from aiida.engine import calcfunction
from aiida.plugins import DataFactory

from aiida_sssp_workflow.utils import RARE_EARTH_ELEMENTS, update_dict
from aiida_sssp_workflow.workflows.convergence._base import _BaseConvergenceWorkChain
from aiida_sssp_workflow.workflows.evaluate._cohesive_energy import (
    CohesiveEnergyWorkChain,
)

UpfData = DataFactory("pseudo.upf")


@calcfunction
def helper_cohesive_energy_difference(
    input_parameters: orm.Dict, ref_parameters: orm.Dict
) -> orm.Dict:
    """calculate the cohesive energy difference from parameters"""
    res_energy = input_parameters["cohesive_energy_per_atom"]
    ref_energy = ref_parameters["cohesive_energy_per_atom"]
    absolute_diff = abs(res_energy - ref_energy) * 1000.0
    relative_diff = abs((res_energy - ref_energy) / ref_energy) * 100

    res = {
        "cohesive_energy_per_atom": res_energy,
        "absolute_diff": absolute_diff,
        "relative_diff": relative_diff,
        "absolute_unit": "meV/atom",
        "relative_unit": "%",
    }

    return orm.Dict(dict=res)


class ConvergenceCohesiveEnergyWorkChain(_BaseConvergenceWorkChain):
    """WorkChain to converge test on cohisive energy of input structure"""

    # pylint: disable=too-many-instance-attributes

    _PROPERTY_NAME = "cohesive_energy"
    _EVALUATE_WORKCHAIN = CohesiveEnergyWorkChain
    _MEASURE_OUT_PROPERTY = "absolute_diff"

    def init_setup(self):
        super().init_setup()
        self.ctx.extra_pw_parameters = {}
        self.ctx.extra_pw_parameters_for_atom = {}

    def extra_setup_for_magnetic_element(self):
        """Extra setup for magnetic element, for atom especially"""
        super().extra_setup_for_magnetic_element()
        self.ctx.extra_pw_parameters_for_atom = {
            self.ctx.element: {
                "SYSTEM": {
                    "nspin": 2,
                    "starting_magnetization": {
                        self.ctx.element: 0.5,
                    },
                },
                "ELECTRONS": {
                    "diagonalization": "cg",
                    "mixing_beta": 0.5,
                    "electron_maxstep": 200,
                },
            }
        }

    def extra_setup_for_rare_earth_element(self):
        """Extra setup for rare earth element, for atom especially"""
        super().extra_setup_for_rare_earth_element()
        self.ctx.extra_pw_parameters_for_atom = {
            self.ctx.element: {
                "SYSTEM": {
                    "nspin": 2,
                    "starting_magnetization": {
                        self.ctx.element: 0.5,
                    },
                    # Need high number of bands to make atom calculation of lanthanoids
                    # converged.
                    "nbnd": int(self.inputs.pseudo.z_valence * 3),
                },
                "ELECTRONS": {
                    "diagonalization": "cg",
                    "mixing_beta": 0.3,  # even small mixing_beta value
                    "electron_maxstep": 200,
                },
            },
        }

    def setup_code_parameters_from_protocol(self):
        """Input validation"""
        # pylint: disable=invalid-name, attribute-defined-outside-init

        # Read from protocol if parameters not set from inputs
        super().setup_code_parameters_from_protocol()

        # parse protocol
        protocol = self.ctx.protocol
        self._DEGAUSS = protocol["degauss"]
        self._OCCUPATIONS = protocol["occupations"]
        self._BULK_SMEARING = protocol["smearing"]
        self._ATOM_SMEARING = protocol["atom_smearing"]
        self._CONV_THR = protocol["electron_conv_thr"]
        self.ctx.kpoints_distance = self._KDISTANCE = protocol["kpoints_distance"]
        self.ctx.vacuum_length = self._VACUUM_LENGTH = protocol["vacuum_length"]

        # Set context parameters
        self.ctx.bulk_parameters = super()._get_pw_base_parameters(
            self._DEGAUSS, self._OCCUPATIONS, self._BULK_SMEARING, self._CONV_THR
        )
        base_atom_pw_parameters = {
            "SYSTEM": {
                "degauss": self._DEGAUSS,
                "occupations": self._OCCUPATIONS,
                "smearing": self._ATOM_SMEARING,
            },
            "ELECTRONS": {
                "conv_thr": self._CONV_THR,
            },
        }

        self.ctx.atom_parameters = {}
        for element in self.ctx.structure.get_symbols_set():
            self.ctx.atom_parameters[element] = base_atom_pw_parameters

        self.ctx.atom_parameters = update_dict(
            self.ctx.atom_parameters, self.ctx.extra_pw_parameters_for_atom
        )

    def _get_inputs(self, ecutwfc, ecutrho):
        """
        get inputs for the evaluation CohesiveWorkChain by provide ecutwfc and ecutrho,
        all other parameters are fixed for the following steps
        """
        update_parameters = {
            "SYSTEM": {
                "ecutwfc": ecutwfc,
                "ecutrho": ecutrho,
            }
        }
        bulk_parameters = update_dict(self.ctx.bulk_parameters, update_parameters)

        atom_kpoints = orm.KpointsData()
        atom_kpoints.set_kpoints_mesh([1, 1, 1])

        # atomic parallelization always set npool to 1 since only one kpoints
        # requires no k parallel
        atomic_parallelization = update_dict(self.ctx.parallelization, {})
        atomic_parallelization.pop("npool", None)
        atomic_parallelization.pop("ndiag", None)
        atomic_parallelization = update_dict(atomic_parallelization, {"npool": 1})
        atomic_parallelization = update_dict(atomic_parallelization, {"ndiag": 1})

        # atomic option if mpiprocs too many confine it too no larger than 32 procs
        atomic_options = update_dict(self.ctx.options, {})
        if atomic_options["resources"]["num_mpiprocs_per_machine"] > 32:
            # copy is a shallow copy, so using update_dict.
            # if simply assign the value will change also the original dict
            atomic_options = update_dict(atomic_options, {"resources": {"num_mpiprocs_per_machine": 32}})

        # atomic calculation for lanthanides require more time to finish.
        if self.ctx.element in RARE_EARTH_ELEMENTS:
            pw_max_walltime = self.ctx.options.get("max_wallclock_seconds", None)
            if pw_max_walltime:
                atomic_options["max_wallclock_seconds"] = pw_max_walltime * 4

        # atom_parameters update with ecutwfc and ecutrho
        atom_parameters = update_dict(self.ctx.atom_parameters, {})
        for element in atom_parameters.keys():
            atom_parameters[element] = update_dict(atom_parameters[element], update_parameters)

        inputs = {
            "pseudos": self.ctx.pseudos,
            "structure": self.ctx.structure,
            "atom_parameters": orm.Dict(dict=atom_parameters),
            "vacuum_length": orm.Float(self.ctx.vacuum_length),
            "bulk": {
                "metadata": {"call_link_label": "bulk_scf"},
                "pw": {
                    "code": self.inputs.code,
                    "parameters": orm.Dict(dict=bulk_parameters),
                    "metadata": {
                        "options": self.ctx.options,
                    },
                    "parallelization": orm.Dict(dict=self.ctx.parallelization),
                },
                "kpoints_distance": orm.Float(self.ctx.kpoints_distance),
            },
            "atom": {
                "metadata": {"call_link_label": "atom_scf"},
                "pw": {
                    "code": self.inputs.code,
                    "parameters": orm.Dict(dict={}),
                    "metadata": {
                        "options": atomic_options,
                    },
                    "parallelization": orm.Dict(dict=atomic_parallelization),
                },
                "kpoints": atom_kpoints,
            },
            "clean_workchain": self.inputs.clean_workchain,
        }

        return inputs

    def helper_compare_result_extract_fun(self, sample_node, reference_node, **kwargs):
        """extract"""
        sample_output = sample_node.outputs.output_parameters
        reference_output = reference_node.outputs.output_parameters

        res = helper_cohesive_energy_difference(
            sample_output, reference_output
        ).get_dict()

        return res

    def get_result_metadata(self):
        return {
            "absolute_unit": "meV/atom",
            "relative_unit": "%",
        }
