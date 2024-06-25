from pathlib import Path

from aiida import orm
from aiida.engine import ProcessBuilder

from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain

from aiida_sssp_workflow.workflows.convergence._base import _BaseConvergenceWorkChain
from aiida_sssp_workflow.utils import get_default_mpi_options


class _CachingConvergenceWorkChain(_BaseConvergenceWorkChain):
    """Convergence caching workflow

    this workflow will only run in verification workflow
    when there are at least two convergence workflows are order to run.

    It also require that the caching machenism of aiida is on.
    The purpose of this workflow is to run a set of common SCF calculations
    with the same input parameters in reference calculation and wavefunction
    cutoff test calculations.
    In order to save the time and resource for the following convergence test."""

    _PROPERTY_NAME = None  # will only use convergence/base protocol
    _EVALUATE_WORKCHAIN = PwBaseWorkChain

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input(
            "code",
            valid_type=orm.AbstractCode,
            required=True,
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
        code: orm.AbstractCode,
        configuration: str | None = None,
        parallelization: dict | None = None,
        mpi_options: dict | None = None,
        clean_workdir: bool = False,
    ) -> ProcessBuilder:
        """Return the builder for the convergence workchain"""
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
        """Input builder for running a dummy SCF for caching as the inner evaluation workchain"""
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
                "tstress": True,  # for pressue evaluation to use _caching node directly.
            },
        }

        builder = self._EVALUATE_WORKCHAIN.get_builder()
        builder.metadata.call_link_label = "SCF_for_cache"

        builder.pw["structure"] = self.structure
        builder.pw["pseudos"] = self.pseudos
        builder.pw["code"] = self.inputs.code
        builder.pw["parameters"] = orm.Dict(dict=pw_parameters)
        builder.pw["parallelization"] = self.inputs.parallelization
        builder.pw["metadata"]["options"] = self.inputs.mpi_options.get_dict()

        builder.kpoints_distance = orm.Float(protocol["kpoints_distance"])

        return builder
