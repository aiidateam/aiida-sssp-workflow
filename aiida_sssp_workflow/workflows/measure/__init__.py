"""Base Measure Workchain since bands measure and delta measure share same input ports
This class makes it able to expose delta measure and bands measure inputs to verification
workchain.
"""
from aiida import orm
from aiida.plugins import DataFactory

from aiida_sssp_workflow.workflows import SelfCleanWorkChain

UpfData = DataFactory("pseudo.upf")


class _BaseMeasureWorkChain(SelfCleanWorkChain):
    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        # yapf: disable
        super().define(spec)
        spec.input('code', valid_type=orm.Code,
                    help='The `pw.x` code use for the `PwCalculation`.')
        spec.input('pseudo', valid_type=UpfData, required=True,
                    help='Pseudopotential to be verified')
        spec.input('protocol', valid_type=orm.Str, required=True,
                    help='The protocol which define input calculation parameters.')
        spec.input('cutoff_control', valid_type=orm.Str, default=lambda: orm.Str('standard'),
                    help='The control protocol where define max_wfc.')
        spec.input('options', valid_type=orm.Dict, required=True,
                    help='Optional `options` to use for the `PwCalculations`.')
        spec.input('parallelization', valid_type=orm.Dict, required=True,
                    help='Parallelization options for the `PwCalculations`.')
