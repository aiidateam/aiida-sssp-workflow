"""Base Measure Workchain since bands measure and precision measure share same input ports
This class makes it able to expose precision measure and bands measure inputs to verification
workchain.
"""

from typing import Tuple
from pathlib import Path

from aiida import orm
from aiida_pseudo.data.pseudo import UpfData

from aiida_sssp_workflow.utils import get_default_mpi_options, get_default_dual
from aiida_sssp_workflow.workflows import SelfCleanWorkChain


class _BaseMeasureWorkChain(SelfCleanWorkChain):
    """Base Measure Workchain since bands measure and precision measure share same input ports"""

    _DEFAULT_WAVEFUNCTION_CUTOFF = 200

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        # yapf: disable
        super().define(spec)
        spec.input('code', valid_type=orm.AbstractCode,
                    help='The `pw.x` code use for the `PwCalculation`.')
        spec.input('pseudo', valid_type=UpfData, required=True,
                    help='Pseudopotential to be verified')
        spec.input('protocol', valid_type=orm.Str, required=True,
                    help='The protocol which define input calculation parameters.')
        spec.input('wavefunction_cutoff', valid_type=orm.Int, required=True, help='The wavefunction cutoff.')
        spec.input('charge_density_cutoff', valid_type=orm.Int, required=True, help='The charge density cutoff.')
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
        code: orm.AbstractCode,
        pseudo: Path | UpfData,
        protocol: str,
        cutoffs: Tuple[float, float] | None = None,
        parallelization: dict | None = None,
        mpi_options: dict | None = None,
        clean_workdir: bool = True,  # default to clean workdir
    ):
        builder = super().get_builder()
        builder.code = code
        builder.protocol = orm.Str(protocol)

        if isinstance(pseudo, Path):
            builder.pseudo = UpfData.get_or_create(pseudo)
        else:
            builder.pseudo = pseudo

        if cutoffs is None:
            ecutwfc = cls._DEFAULT_WAVEFUNCTION_CUTOFF
            ecutrho = cls._DEFAULT_WAVEFUNCTION_CUTOFF * get_default_dual(
                builder.pseudo
            )
        else:
            ecutwfc = cutoffs[0]
            ecutrho = cutoffs[1]

        builder.wavefunction_cutoff, builder.charge_density_cutoff = (
            orm.Int(round(ecutwfc)),
            orm.Int(round(ecutrho)),
        )

        if parallelization:
            builder.parallelization = orm.Dict(parallelization)
        else:
            builder.parallelization = orm.Dict()

        if mpi_options:
            builder.mpi_options = orm.Dict(mpi_options)
        else:
            builder.mpi_options = orm.Dict(get_default_mpi_options())

        builder.clean_workdir = orm.Bool(clean_workdir)

        return builder

    def _get_pw_cutoff(
        self, structure: orm.StructureData, ecutwfc: float, ecutrho: float
    ):
        """Get cutoff pair, if strcture contains oxygen or nitrogen, need
        to use the max between pseudo cutoff and the O/N cutoff.
        """
        o_ecutwfc = self.inputs.oxygen_ecutwfc.value
        o_ecutrho = self.inputs.oxygen_ecutrho.value

        elements = set(structure.get_symbols_set())
        if "O" in elements:
            ecutwfc = max(ecutwfc, o_ecutwfc)
            ecutrho = max(ecutrho, o_ecutrho)

        if "N" in elements:
            ecutwfc = max(ecutwfc, self._N_ECUTWFC)
            ecutrho = max(ecutrho, self._N_ECUTRHO)

        return ecutwfc, ecutrho
