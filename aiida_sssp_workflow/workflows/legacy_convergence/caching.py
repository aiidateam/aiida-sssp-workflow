from aiida_sssp_workflow.workflows.legacy_convergence.pressure import (
    ConvergencePressureWorkChain,
)


class _CachingConvergenceWorkChain(ConvergencePressureWorkChain):
    """Convergence caching"""
