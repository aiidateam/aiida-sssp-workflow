# -*- coding: utf-8 -*-
"""
Convergence test on phonon frequencies of a given pseudopotential
"""

from typing import Union
from pathlib import Path

from aiida import orm
from aiida.engine import ProcessBuilder
from aiida_pseudo.data.pseudo import UpfData
from aiida_sssp_workflow.workflows.convergence._base import _BaseConvergenceWorkChain
from aiida_sssp_workflow.workflows.evaluate._phonon_frequencies import (
    PhononFrequenciesWorkChain,
)
from aiida_sssp_workflow.utils import get_default_mpi_options


class ConvergencePhononFrequenciesWorkChain(_BaseConvergenceWorkChain):
    """WorkChain to converge test on cohisive energy of input structure"""

    # pylint: disable=too-many-instance-attributes

    _PROPERTY_NAME = "phonon_frequencies"
    _EVALUATE_WORKCHAIN = PhononFrequenciesWorkChain

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input(
            "pw_code",
            valid_type=orm.AbstractCode,
            help="The `pw.x` code use for the `PwCalculation`.",
        )
        spec.input(
            "ph_code",
            valid_type=orm.AbstractCode,
            help="The `ph.x` code  use for the `PhCalculation`.",
        )
        spec.input(
            "pw_parallelization",
            valid_type=orm.Dict,
            required=False,
            help="The parallelization settings for the `PwCalculation`",
        )
        spec.input(
            "pw_mpi_options",
            valid_type=orm.Dict,
            required=False,
            help="The MPI options for the `PwCalculation`",
        )
        spec.input(
            "ph_mpi_options",
            valid_type=orm.Dict,
            required=False,
            help="The MPI options for the `PhCalculation`",
        )
        spec.input(
            "ph_settings",
            valid_type=orm.Dict,
            required=False,
            help="The settings for ph calculation, npool is currently set here.",
        )

    @classmethod
    def get_builder(
        cls,
        pseudo: Union[Path, UpfData],
        protocol: str,
        cutoff_list: list,
        pw_code: orm.AbstractCode,
        ph_code: orm.AbstractCode,
        configuration: str | None = None,
        pw_parallelization: dict | None = None,
        pw_mpi_options: dict | None = None,
        ph_settings: dict | None = None,
        ph_mpi_options: dict | None = None,
        clean_workdir: bool = True,  # default to clean workdir
    ) -> ProcessBuilder:
        """Return a builder to run this EOS convergence workchain"""
        builder = super().get_builder(pseudo, protocol, cutoff_list, configuration)

        builder.metadata.call_link_label = "convergence_phonon_frequencies"
        builder.clean_workdir = orm.Bool(clean_workdir)
        builder.pw_code = pw_code
        builder.ph_code = ph_code

        if pw_parallelization:
            builder.pw_parallelization = orm.Dict(pw_parallelization)
        else:
            builder.pw_parallelization = orm.Dict()

        if ph_settings:
            builder.ph_settings = orm.Dict(ph_settings)
        else:
            builder.ph_settings = orm.Dict()

        if pw_mpi_options:
            builder.pw_mpi_options = orm.Dict(pw_mpi_options)
        else:
            builder.pw_mpi_options = orm.Dict(get_default_mpi_options())

        if ph_mpi_options:
            builder.ph_mpi_options = orm.Dict(ph_mpi_options)
        else:
            builder.ph_mpi_options = orm.Dict(get_default_mpi_options())

        return builder

    def prepare_evaluate_builder(self, ecutwfc, ecutrho):
        """Input builder for running the inner phonon frequencies evaluation workchain"""
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
            },
        }

        builder = self._EVALUATE_WORKCHAIN.get_builder()

        builder.clean_workdir = (
            self.inputs.clean_workdir
        )  # sync with the main workchain

        builder.structure = self.structure

        # pw calculation
        builder.scf.kpoints_distance = orm.Float(protocol["kpoints_distance"])
        builder.scf.metadata.call_link_label = "scf"
        builder.scf.pw["code"] = self.inputs.pw_code
        builder.scf.pw["pseudos"] = self.pseudos
        builder.scf.pw["parameters"] = orm.Dict(dict=pw_parameters)
        builder.scf.pw["parallelization"] = self.inputs.pw_parallelization
        builder.scf.pw["metadata"]["options"] = self.inputs.pw_mpi_options.get_dict()

        # ph calculation
        qpoints = orm.KpointsData()
        qpoints.set_cell_from_structure(self.structure)
        qpoints.set_kpoints(protocol["qpoints_list"])

        ph_parameters = {
            "INPUTPH": {
                "tr2_ph": protocol["tr2_ph"],
                "epsil": protocol["epsilon"],
                "diagonalization": protocol["diagonalization"],
            }
        }

        builder.phonon.metadata.call_link_label = "ph"
        builder.phonon["qpoints"] = qpoints
        builder.phonon.ph["code"] = self.inputs.ph_code
        builder.phonon.ph["parameters"] = orm.Dict(dict=ph_parameters)
        builder.phonon.ph["settings"] = self.inputs.ph_settings
        builder.phonon.ph["metadata"]["options"] = self.inputs.ph_mpi_options.get_dict()

        return builder
