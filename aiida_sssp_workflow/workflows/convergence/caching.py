from aiida_sssp_workflow.workflows.convergence.pressure import (
    ConvergencePressureWorkChain,
)


class _CachingConvergenceWorkChain(ConvergencePressureWorkChain):
    """Convergence caching"""
