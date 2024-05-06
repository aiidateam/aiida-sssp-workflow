# -*- coding: utf-8 -*-
"""
Convergence test on cohesive energy of a given pseudopotential
"""

from pathlib import Path

from aiida import orm
from aiida.engine import ProcessBuilder

from aiida_sssp_workflow.utils import get_default_mpi_options
from aiida_sssp_workflow.workflows.convergence._base import _BaseConvergenceWorkChain
from aiida_sssp_workflow.workflows.evaluate._metric import MetricWorkChain


class ConvergenceEOSWorkChain(_BaseConvergenceWorkChain):
    """WorkChain to converge test on delta factor of input structure"""

    # pylint: disable=too-many-instance-attributes

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
        pseudo: Path,
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

        builder.metadata.call_link_label = "caching"
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
        builder.configuration = self.inputs.configuration

        builder.eos.metadata.call_link_label = "EOS"
        builder.eos.structure = self.structure
        builder.eos.kpoints_distance = orm.Float(protocol["kpoints_distance"])
        builder.eos.scale_count = orm.Int(protocol["scale_count"])
        builder.eos.scale_increment = orm.Float(protocol["scale_increment"])

        # pw
        builder.eos.pw["code"] = self.inputs.code
        builder.eos.pw["pseudos"] = self.ctx.pseudos
        builder.eos.pw["parameters"] = orm.Dict(dict=pw_parameters)
        builder.eos.pw["parallelization"] = self.inputs.parallelization
        builder.eos.pw["metadata"]["options"] = self.inputs.mpi_options.get_dict()

        return builder
