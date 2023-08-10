from aiida import orm
from aiida.plugins import WorkflowFactory

from aiida_sssp_workflow.utils import update_dict
from aiida_sssp_workflow.workflows.convergence._base import _BaseConvergenceWorkChain

PwBaseWorkflow = WorkflowFactory("quantumespresso.pw.base")


class _CachingConvergenceWorkChain(_BaseConvergenceWorkChain):
    """Convergence caching workflow
    this workflow will only run in verification workflow
    when there are at least two convergence workflows are order to run.
    It also require that the caching machenism of aiida is on.
    The purpose of this workflow is to run a set of common SCF calculations
    with the same input parameters in reference calculation and wavefunction
    cutoff test calculations. In order to save the time and resource for
    the following convergence test."""

    _PROPERTY_NAME = None  # will only use convergence/base protocol
    _EVALUATE_WORKCHAIN = PwBaseWorkflow
    _MEASURE_OUT_PROPERTY = None

    _RUN_WFC_TEST = True
    _RUN_RHO_TEST = False  # will not run charge density cutoff test

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input(
            "clean_workdir",
            valid_type=orm.Bool,
            default=lambda: orm.Bool(False),
            help="If `True`, work directories of all called calculation will be cleaned at the end of execution.",
        )

    def init_setup(self):
        super().init_setup()
        self.ctx.extra_pw_parameters = {
            "CONTROL": {
                "disk_io": "low",
            },
        }

    def inspect_wfc_convergence_test(self):
        """Override this step to do nothing to parse wavefunction
        cutoff test results but only run it."""
        return None

    def setup_criteria_parameters_from_protocol(self):
        """Override this step to do nothing, since it is not
        used for caching run."""
        return None

    def get_result_metadata(self):
        """No need to actual implemented for caching workchain"""
        return None

    def helper_compare_result_extract_fun(
        self, sample_node, reference_node, **kwargs
    ) -> dict:
        """No need to actual implemented for caching workchain"""
        return None

    def setup_code_parameters_from_protocol(self):
        """Input validation"""
        # pylint: disable=invalid-name, attribute-defined-outside-init

        # Read from protocol if parameters not set from inputs
        super().setup_code_parameters_from_protocol()

        protocol = self.ctx.protocol
        self._DEGAUSS = protocol["degauss"]
        self._OCCUPATIONS = protocol["occupations"]
        self._SMEARING = protocol["smearing"]
        self._CONV_THR_PER_ATOM = protocol["conv_thr_per_atom"]
        self.ctx.kpoints_distance = self._KDISTANCE = protocol["kpoints_distance"]

        self.ctx.pw_base_parameters = super()._get_pw_base_parameters(
            self._DEGAUSS,
            self._OCCUPATIONS,
            self._SMEARING,
            self._CONV_THR_PER_ATOM,
        )

    def _get_inputs(self, ecutwfc, ecutrho) -> dict:
        """inputs for running a dummy SCF for caching"""
        pw_parameters = update_dict(self.ctx.pw_base_parameters, {})
        pw_parameters["SYSTEM"]["ecutwfc"] = ecutwfc
        pw_parameters["SYSTEM"]["ecutrho"] = ecutrho

        inputs = {
            "metadata": {"call_link_label": "SCF_for_cache"},
            "pw": {
                "structure": self.ctx.structure,
                "code": self.inputs.code,
                "pseudos": self.ctx.pseudos,
                "parameters": orm.Dict(dict=pw_parameters),
                "metadata": {
                    "options": self.ctx.options,
                },
                "parallelization": orm.Dict(dict=self.ctx.parallelization),
            },
            "kpoints_distance": orm.Float(self.ctx.kpoints_distance),
        }

        return inputs
