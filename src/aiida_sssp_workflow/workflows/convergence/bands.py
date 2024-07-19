# -*- coding: utf-8 -*-
"""
Convergence test on bands of a given pseudopotential
"""

from pathlib import Path
from typing import Union, Any
import copy

from aiida import orm
from aiida.engine import ProcessBuilder
from aiida_pseudo.data.pseudo import UpfData

from aiida_sssp_workflow.utils import get_default_mpi_options
from aiida_sssp_workflow.calculations.calculate_bands_distance import get_bands_distance
from aiida_sssp_workflow.workflows.convergence._base import _BaseConvergenceWorkChain
from aiida_sssp_workflow.workflows.evaluate._bands import BandsWorkChain
from aiida_sssp_workflow.workflows.convergence.report import ConvergenceReport


class ConvergenceBandsWorkChain(_BaseConvergenceWorkChain):
    """WorkChain to converge test on cohisive energy of input structure"""

    # pylint: disable=too-many-instance-attributes

    _PROPERTY_NAME = "bands"
    _EVALUATE_WORKCHAIN = BandsWorkChain

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

    def prepare_evaluate_builder(self, ecutwfc, ecutrho):
        """Input builder for running the inner bands/bands-structure evation workchain"""
        protocol = self.protocol
        natoms = len(self.structure.sites)

        builder = self._EVALUATE_WORKCHAIN.get_builder()

        builder.clean_workdir = (
            self.inputs.clean_workdir
        )  # sync with the main workchain

        builder.metadata.call_link_label = "convergence_bands"

        # ports from PwBandsWorkChain: scf/bands namespace and structure port
        builder.structure = self.structure

        # For SCF pw calculation
        scf_pw_parameters = {
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
            },
        }

        builder.scf.pw["code"] = self.inputs.code
        builder.scf.pw["pseudos"] = self.pseudos
        builder.scf.pw["parameters"] = orm.Dict(scf_pw_parameters)
        builder.scf.pw["parallelization"] = self.inputs.parallelization
        builder.scf.pw["metadata"]["options"] = self.inputs.mpi_options.get_dict()
        builder.scf.kpoints_distance = orm.Float(protocol["kpoints_distance"])

        # For band pw calculation
        bands_pw_parameters = {
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
                "calculation": "bands",
            },
        }

        builder.bands.pw["code"] = self.inputs.code
        builder.bands.pw["pseudos"] = self.pseudos
        builder.bands.pw["parameters"] = orm.Dict(bands_pw_parameters)
        builder.bands.pw["parallelization"] = self.inputs.parallelization
        builder.bands.pw["metadata"]["options"] = self.inputs.mpi_options.get_dict()

        # Generic
        builder.kpoints_distance_bands = orm.Float(protocol["kpoints_distance"])
        builder.init_nbands_factor = orm.Int(protocol["init_nbands_factor"])
        builder.fermi_shift = orm.Float(protocol["fermi_shift"])
        builder.run_band_structure = orm.Bool(False)

        return builder


def compute_xy(
    node: orm.Node,
) -> dict[str, Any]:
    """From report calculate the xy data, xs are cutoffs and ys are band distance from reference"""
    report_dict = node.outputs.report.get_dict()
    report = ConvergenceReport.construct(**report_dict)

    reference_node = orm.load_node(report.reference.uuid)
    band_structure_r: orm.BandsData = reference_node.outputs.bands.band_structure
    band_parameters_r: orm.Dict = reference_node.outputs.bands.band_parameters

    bandsdata_r = {
        "number_of_electrons": band_parameters_r["number_of_electrons"],
        "number_of_bands": band_parameters_r["number_of_bands"],
        "fermi_level": band_parameters_r["fermi_energy"],
        "bands": band_structure_r.get_bands(),
        "kpoints": band_structure_r.get_kpoints(),
        "weights": band_structure_r.get_array("weights"),
    }

    # smearing width is from degauss
    smearing = reference_node.inputs.bands.pw.parameters.get_dict()["SYSTEM"]["degauss"]
    fermi_shift = reference_node.inputs.fermi_shift.value

    # always do smearing on high bands and not include the spin since we didn't turn on the spin for all
    # convergence test, but this may change in the future.
    # The `get_bands_distance` function can deal with mag bands with spin_up and spin_down
    spin = False
    do_smearing = True

    xs = []
    ys_eta_c = []
    ys_max_diff_c = []
    for node_point in report.convergence_list:
        if node_point.exit_status != 0:
            # TODO: log to a warning file for where the node is not finished_okay
            continue

        x = node_point.wavefunction_cutoff
        xs.append(x)

        node = orm.load_node(node_point.uuid)
        band_structure_p: orm.BandsData = node.outputs.bands.band_structure
        band_parameters_p: orm.Dict = node.outputs.bands.band_parameters

        # The raw implementation of `get_bands_distance` is in `aiida_sssp_workflow/calculations/bands_distance.py`
        bandsdata_p = {
            "number_of_electrons": band_parameters_p["number_of_electrons"],
            "number_of_bands": band_parameters_p["number_of_bands"],
            "fermi_level": band_parameters_p["fermi_energy"],
            "bands": band_structure_p.get_bands(),
            "kpoints": band_structure_p.get_kpoints(),
            "weights": band_structure_p.get_array("weights"),
        }
        res = get_bands_distance(
            copy.deepcopy(bandsdata_r),
            copy.deepcopy(bandsdata_p),
            smearing=smearing,
            fermi_shift=fermi_shift,
            do_smearing=do_smearing,
            spin=spin,
        )
        eta_c = res.get("eta_c", None)
        shift_c = res.get("shift_c", None)
        max_diff_c = res.get("max_diff_c", None)
        unit = res.get("unit", None)

        # eta_c is the y, others are write into as metadata
        ys_eta_c.append(eta_c)
        ys_max_diff_c.append(max_diff_c)

    return {
        "xs": xs,
        "ys": ys_eta_c,
        "ys_eta_c": ys_eta_c,
        "ys_max_diff_c": ys_max_diff_c,
        "metadata": {
            "unit": unit,
        },
    }
