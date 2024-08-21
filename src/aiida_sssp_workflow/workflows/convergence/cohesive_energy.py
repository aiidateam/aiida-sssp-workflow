# -*- coding: utf-8 -*-
"""
Convergence test on cohesive energy of a given pseudopotential
"""

from pathlib import Path
from typing import Union, Any

from aiida import orm
from aiida.engine import ProcessBuilder
from aiida_pseudo.data.pseudo import UpfData

from aiida_sssp_workflow.utils import get_default_mpi_options
from aiida_sssp_workflow.utils.element import ACTINIDE_ELEMENTS, LANTHANIDE_ELEMENTS
from aiida_sssp_workflow.workflows.convergence.report import ConvergenceReport
from aiida_sssp_workflow.workflows.convergence._base import _BaseConvergenceWorkChain
from aiida_sssp_workflow.workflows.evaluate._cohesive_energy import (
    CohesiveEnergyWorkChain,
)


class ConvergenceCohesiveEnergyWorkChain(_BaseConvergenceWorkChain):
    """WorkChain to converge test on cohisive energy of input structure"""

    _PROPERTY_NAME = "cohesive_energy"
    _EVALUATE_WORKCHAIN = CohesiveEnergyWorkChain

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input(
            "code",
            valid_type=orm.AbstractCode,
            help="The `pw.x` code use for the `PwCalculation`.",
        )
        spec.input(
            "bulk_parallelization",
            valid_type=orm.Dict,
            required=False,
            help="The parallelization settings for the `PwCalculation` of bulk calculation.",
        )
        spec.input(
            "bulk_mpi_options",
            valid_type=orm.Dict,
            required=False,
            help="The MPI options for the `PwCalculation` of bulk calculation.",
        )
        spec.input(
            "atom_parallelization",
            valid_type=orm.Dict,
            required=False,
            help="The parallelization settings for the `PwCalculation` of bulk calculation.",
        )
        spec.input(
            "atom_mpi_options",
            valid_type=orm.Dict,
            required=False,
            help="The MPI options for the `PwCalculation` of bulk calculation.",
        )

    @classmethod
    def get_builder(
        cls,
        pseudo: Union[Path, UpfData],
        protocol: str,
        cutoff_list: list,
        code: orm.AbstractCode,
        configuration: str | None = None,
        bulk_parallelization: dict | None = None,
        bulk_mpi_options: dict | None = None,
        atom_parallelization: dict | None = None,
        atom_mpi_options: dict | None = None,
        clean_workdir: bool = True,  # clean workdir by default
    ) -> ProcessBuilder:
        """Return a builder to run this EOS convergence workchain"""
        builder = super().get_builder(pseudo, protocol, cutoff_list, configuration)

        builder.metadata.call_link_label = "convergence_cohesive_energy"
        builder.clean_workdir = orm.Bool(clean_workdir)
        builder.code = code

        if bulk_parallelization:
            builder.bulk_parallelization = orm.Dict(bulk_parallelization)
        else:
            builder.bulk_parallelization = orm.Dict()

        if bulk_mpi_options:
            builder.bulk_mpi_options = orm.Dict(bulk_mpi_options)
        else:
            builder.bulk_mpi_options = orm.Dict(get_default_mpi_options())

        if atom_parallelization:
            builder.atom_parallelization = orm.Dict(atom_parallelization)
        else:
            builder.atom_parallelization = orm.Dict()

        if atom_mpi_options:
            builder.atom_mpi_options = orm.Dict(atom_mpi_options)
        else:
            builder.atom_mpi_options = orm.Dict(get_default_mpi_options())

        return builder

    def prepare_evaluate_builder(self, ecutwfc, ecutrho) -> ProcessBuilder:
        """Input builder for running the inner EOS evaluation workchain"""
        protocol = self.protocol
        natoms = len(self.structure.sites)

        builder = self._EVALUATE_WORKCHAIN.get_builder()

        builder.clean_workdir = (
            self.inputs.clean_workdir
        )  # sync with the main workchain
        builder.pseudos = self.pseudos
        builder.structure = self.structure
        builder.vacuum_length = orm.Float(protocol["vacuum_length"])

        # bulk
        bulk_pw_parameters = {
            "SYSTEM": {
                "degauss": protocol["degauss"],
                "occupations": protocol["occupations"],
                "smearing": protocol["smearing"],
                "ecutwfc": ecutwfc,  # <-- Here set the ecutwfc
                "ecutrho": ecutrho,  # <-- Here set the ecutrho
            },
            "ELECTRONS": {
                "conv_thr": protocol["conv_thr_per_atom"] * natoms,
                "mixing_beta": protocol["mixing_beta"],
            },
            "CONTROL": {
                "calculation": "scf",
                "disk_io": "nowf",  # not store wavefunction file to save inodes
            },
        }

        builder.bulk.kpoints_distance = orm.Float(protocol["kpoints_distance"])
        builder.bulk.metadata.call_link_label = "cohesive_bulk_scf"
        builder.bulk.pw["code"] = self.inputs.code
        builder.bulk.pw["parameters"] = orm.Dict(dict=bulk_pw_parameters)
        builder.bulk.pw["parallelization"] = self.inputs.bulk_parallelization
        builder.bulk.pw["metadata"]["options"] = self.inputs.bulk_mpi_options.get_dict()

        # atom
        atom_pw_parameters = {
            "SYSTEM": {
                "degauss": protocol["degauss"],
                "occupations": protocol["occupations"],
                "smearing": protocol["atom_smearing"],
                "ecutwfc": ecutwfc,  # <-- Here set the ecutwfc
                "ecutrho": ecutrho,  # <-- Here set the ecutrho
                "nosym": True,  # this is enssential for getting a converged isolated atom calculation.
            },
            "ELECTRONS": {
                "conv_thr": protocol["conv_thr_per_atom"] * natoms,
                "mixing_beta": protocol["mixing_beta"],
            },
            "CONTROL": {
                "calculation": "scf",
                "disk_io": "nowf",  # not store wavefunction file to save inodes
            },
        }

        # XXX: This I have to add here, although not best option to keep
        # an condition inside the generic workflow. But for lanthanoids and actinoids
        # using large nbnd is the only way to make it converge. Set it to 5 * Z
        if self.element in LANTHANIDE_ELEMENTS + ACTINIDE_ELEMENTS:
            atom_pw_parameters["SYSTEM"]["nbnd"] = 5 * self.inputs.pseudo.z_valence

        atom_kpoints = orm.KpointsData()
        atom_kpoints.set_kpoints_mesh([1, 1, 1])

        builder.atom.kpoints = atom_kpoints
        builder.atom.metadata.call_link_label = "cohesive_atom_scf"
        builder.atom.pw["code"] = self.inputs.code
        builder.atom.pw["parameters"] = orm.Dict(dict=atom_pw_parameters)
        builder.atom.pw["parallelization"] = self.inputs.atom_parallelization
        builder.atom.pw["metadata"]["options"] = self.inputs.atom_mpi_options.get_dict()

        return builder

def compute_xy(
    node: orm.Node,
) -> dict[str, Any]:
    """From report calculate the xy data, xs are cutoffs and ys are cohesive energy diff from reference"""
    report_dict = node.outputs.report.get_dict()
    report = ConvergenceReport.construct(**report_dict)

    reference_node = orm.load_node(report.reference.uuid)
    output_parameters_r: orm.Dict = reference_node.outputs.output_parameters
    y_ref = output_parameters_r['cohesive_energy_per_atom']

    xs = []
    ys = []
    ys_cohesive_energy_per_atom = []
    for node_point in report.convergence_list:
        if node_point.exit_status != 0:
            # TODO: log to a warning file for where the node is not finished_okay
            continue
        
        x = node_point.wavefunction_cutoff
        xs.append(x)

        node = orm.load_node(node_point.uuid)
        output_parameters_p: orm.Dict = node.outputs.output_parameters

        y = (output_parameters_p['cohesive_energy_per_atom'] - y_ref) / y_ref * 100
        ys.append(y)
        ys_cohesive_energy_per_atom.append(output_parameters_p['cohesive_energy_per_atom'])

    return {
        'xs': xs,
        'ys': ys,
        'ys_relative_diff': ys,
        'ys_cohesive_energy_per_atom': ys_cohesive_energy_per_atom,
        'metadata': {
            'unit': '%',
        }
    }

