"""Base Measure Workchain since bands measure and precision measure share same input ports
This class makes it able to expose precision measure and bands measure inputs to verification
workchain.
"""
from aiida import orm
from aiida.plugins import DataFactory

from aiida_sssp_workflow.workflows import SelfCleanWorkChain

UpfData = DataFactory("pseudo.upf")


class _BaseMeasureWorkChain(SelfCleanWorkChain):

    # ECUT for oxygen, remember to update this if the oxygen pseudo is changed
    _O_ECUTWFC = 75.0
    _O_ECUTRHO = 600.0

    # ECUT for nitrogen, remember to update this if the nitrogen pseudo is changed
    _N_ECUTWFC = 80.0
    _N_ECUTRHO = 320.0

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
        spec.input('wavefunction_cutoff', valid_type=orm.Float, required=True, help='The wavefunction cutoff.')
        spec.input('charge_density_cutoff', valid_type=orm.Float, required=True, help='The charge density cutoff.')
        spec.input('options', valid_type=orm.Dict, required=True,
                    help='Optional `options` to use for the `PwCalculations`.')
        spec.input('parallelization', valid_type=orm.Dict, required=True,
                    help='Parallelization options for the `PwCalculations`.')
