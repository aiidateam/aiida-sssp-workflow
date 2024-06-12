"""Base Measure Workchain since bands measure and precision measure share same input ports
This class makes it able to expose precision measure and bands measure inputs to verification
workchain.
"""

from aiida import orm
from aiida.plugins import DataFactory

from aiida_sssp_workflow.workflows import SelfCleanWorkChain

UpfData = DataFactory("pseudo.upf")


class _BaseMeasureWorkChain(SelfCleanWorkChain):
    """Base Measure Workchain since bands measure and precision measure share same input ports"""

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        # yapf: disable
        super().define(spec)
        spec.input('code', valid_type=orm.AbstractCode,
                    help='The `pw.x` code use for the `PwCalculation`.')
        spec.input('pseudo', valid_type=UpfData, required=True,
                    help='Pseudopotential to be verified')
        spec.input('oxygen_pseudo', valid_type=UpfData, required=True)
        spec.input('oxygen_ecutwfc', valid_type=orm.Float, required=True)
        spec.input('oxygen_ecutrho', valid_type=orm.Float, required=True)
        spec.input('protocol', valid_type=orm.Str, required=True,
                    help='The protocol which define input calculation parameters.')
        spec.input('configurations', valid_type=orm.List, required=False)
        spec.input('wavefunction_cutoff', valid_type=orm.Float, required=True, help='The wavefunction cutoff.')
        spec.input('charge_density_cutoff', valid_type=orm.Float, required=True, help='The charge density cutoff.')
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
