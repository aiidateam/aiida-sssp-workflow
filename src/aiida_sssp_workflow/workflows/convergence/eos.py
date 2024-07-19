# -*- coding: utf-8 -*-
"""
Convergence test on cohesive energy of a given pseudopotential
"""

from pathlib import Path
from typing import Union, Any

from aiida import orm
from aiida.engine import ProcessBuilder
from aiida_pseudo.data.pseudo import UpfData

from aiida_sssp_workflow.calculations.calculate_metric import rel_errors_vec_length
from aiida_sssp_workflow.utils import get_default_mpi_options
from aiida_sssp_workflow.workflows.convergence._base import _BaseConvergenceWorkChain
from aiida_sssp_workflow.workflows.convergence.report import ConvergenceReport
from aiida_sssp_workflow.workflows.evaluate._metric import MetricWorkChain


class ConvergenceEOSWorkChain(_BaseConvergenceWorkChain):
    """WorkChain to converge test on delta factor of input structure"""

    _PROPERTY_NAME = "eos"
    _EVALUATE_WORKCHAIN = MetricWorkChain

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input(
            "code",
            valid_type=orm.AbstractCode,
            help="The `pw.x` code use for the `PwCalculation`.",
        )
        spec.input(
            "parallelization",
            valid_type=orm.Dict,
            required=False,
            help="The parallelization settings for the `PwCalculation`.",
        )
        spec.input(
            "mpi_options",
            valid_type=orm.Dict,
            required=False,
            help="The MPI options for the `PwCalculation`.",
        )

    @classmethod
    def get_builder(
        cls,
        pseudo: Union[Path, UpfData],
        protocol: str,
        cutoff_list: list,
        code: orm.AbstractCode,
        configuration: str | None = None,
        parallelization: dict | None = None,
        mpi_options: dict | None = None,
        clean_workdir: bool = True,  # default to clean workdir
    ) -> ProcessBuilder:
        """Return a builder to run this EOS convergence workchain"""
        builder = super().get_builder(pseudo, protocol, cutoff_list, configuration)

        builder.metadata.call_link_label = "convergence_eos"
        builder.clean_workdir = orm.Bool(clean_workdir)
        builder.code = code

        if parallelization:
            builder.parallelization = orm.Dict(parallelization)
        else:
            builder.parallelization = orm.Dict()

        if mpi_options:
            builder.mpi_options = orm.Dict(mpi_options)
        else:
            builder.mpi_options = orm.Dict(get_default_mpi_options())

        return builder

    def prepare_evaluate_builder(self, ecutwfc, ecutrho) -> ProcessBuilder:
        """Input builder for running the inner EOS evaluation workchain"""
        protocol = self.protocol
        natoms = len(self.structure.sites)

        pw_parameters = {
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

        ## For lanthanides, use sparse kpoints and tetrahedra occupation
        ## TODO: TBD, high possibility to remove this
        # if self.ctx.element in LANTHANIDE_ELEMENTS:
        #    self.ctx.kpoints_distance = self._KDISTANCE + 0.05
        #    pw_parameters["SYSTEM"].pop("smearing", None)
        #    pw_parameters["SYSTEM"].pop("degauss", None)
        #    pw_parameters["SYSTEM"]["occupations"] = "tetrahedra"

        builder = self._EVALUATE_WORKCHAIN.get_builder()

        builder.clean_workdir = (
            self.inputs.clean_workdir
        )  # sync with the main workchain

        builder.element = orm.Str(self.element)
        builder.configuration = self.configuration

        builder.eos.metadata.call_link_label = "EOS"
        builder.eos.structure = self.structure
        builder.eos.kpoints_distance = orm.Float(protocol["kpoints_distance"])
        builder.eos.scale_count = orm.Int(protocol["scale_count"])
        builder.eos.scale_increment = orm.Float(protocol["scale_increment"])

        # pw
        builder.eos.pw["code"] = self.inputs.code
        builder.eos.pw["pseudos"] = self.pseudos
        builder.eos.pw["parameters"] = orm.Dict(dict=pw_parameters)
        builder.eos.pw["parallelization"] = self.inputs.parallelization
        builder.eos.pw["metadata"]["options"] = self.inputs.mpi_options.get_dict()

        return builder


def compute_xy(
    node: orm.Node,
) -> dict[str, Any]:
    """From report calculate the xy data, xs are cutoffs and ys are eos from reference"""
    report_dict = node.outputs.report.get_dict()
    report = ConvergenceReport.construct(**report_dict)

    reference_node = orm.load_node(report.reference.uuid)
    output_parameters_r: orm.Dict = reference_node.outputs.output_parameters
    ref_V0, ref_B0, ref_B1 = output_parameters_r["birch_murnaghan_results"]

    xs = []
    ys_nu = []
    for node_point in report.convergence_list:
        if node_point.exit_status != 0:
            # TODO: log to a warning file for where the node is not finished_okay
            continue

        x = node_point.wavefunction_cutoff
        xs.append(x)

        node = orm.load_node(node_point.uuid)
        output_parameters_p: orm.Dict = node.outputs.output_parameters

        V0, B0, B1 = output_parameters_p["birch_murnaghan_results"]

        y_nu = rel_errors_vec_length(ref_V0, ref_B0, ref_B1, V0, B0, B1)

        ys_nu.append(y_nu)

    return {
        "xs": xs,
        "ys": ys_nu,
        "metadata": {
            "unit": "n/a",
        },
    }


# def compute_xy_epsilon(
#     report: ConvergenceReport,
# ) -> dict[str, Any]:
#     sample_node = orm.load_node(sample_uuid)
#     ref_node = orm.load_node(ref_uuid)
#
#     arr_sample = np.array(sample_node.outputs.eos.output_volume_energy.get_dict()["energies"])
#     arr_ref = np.array(ref_node.outputs.eos.output_volume_energy.get_dict()["energies"])
#
#     avg_sample = np.average(arr_sample)
#     avg_ref = np.average(arr_ref)
#
#     # eq.6 of nat.rev.phys
#     A = np.sum(np.square(arr_sample - arr_ref))
#     B = np.sum(np.square(arr_sample - avg_sample))
#     C = np.sum(np.square(arr_ref - avg_ref))
#
#     epsilon = np.sqrt(A / (np.sqrt(B * C)))
