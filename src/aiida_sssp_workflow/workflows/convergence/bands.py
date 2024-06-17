# -*- coding: utf-8 -*-
"""
Convergence test on bands of a given pseudopotential
"""

from pathlib import Path
from typing import Union

from aiida import orm
from aiida.engine import ProcessBuilder
from aiida_pseudo.data.pseudo import UpfData

from aiida_sssp_workflow.utils import get_default_mpi_options
from aiida_sssp_workflow.workflows.convergence._base import _BaseConvergenceWorkChain
from aiida_sssp_workflow.workflows.evaluate._bands import BandsWorkChain


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
        configuration: str,
        code: orm.AbstractCode,
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
