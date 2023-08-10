# -*- coding: utf-8 -*-
"""
All in one verification workchain
"""
# pylint: disable=cyclic-import
from aiida import orm
from aiida.engine import ToContext, if_
from aiida.engine.processes.exit_code import ExitCode
from aiida.engine.processes.functions import calcfunction
from aiida.plugins import DataFactory, WorkflowFactory

from aiida_sssp_workflow.workflows import SelfCleanWorkChain
from aiida_sssp_workflow.workflows.convergence import _BaseConvergenceWorkChain
from aiida_sssp_workflow.workflows.convergence.caching import (
    _CachingConvergenceWorkChain,
)
from aiida_sssp_workflow.workflows.measure import _BaseMeasureWorkChain

UpfData = DataFactory("pseudo.upf")


@calcfunction
def parse_pseudo_info(pseudo: UpfData):
    """parse the pseudo info as a Dict"""
    from pseudo_parser.upf_parser import parse

    try:
        info = parse(pseudo.get_content())
    except ValueError:
        return ExitCode(100, "cannot parse the info of pseudopotential.")

    return orm.Dict(dict=info)


_REMOTE_FOLDER_DEPENDENT_CONVERENCE_PROPERTIES_LIST = [
    "convergence.bands",
    "convergence.phonon_frequencies",
]

_REMOTE_FOLDER_INDEPENDENT_CONVERGENCE_PROPERTIES_LIST = [
    "convergence.cohesive_energy",
    "convergence.pressure",
    "convergence.delta",
]

DEFAULT_CONVERGENCE_PROPERTIES_LIST = (
    _REMOTE_FOLDER_DEPENDENT_CONVERENCE_PROPERTIES_LIST
    + _REMOTE_FOLDER_INDEPENDENT_CONVERGENCE_PROPERTIES_LIST
)

DEFAULT_MEASURE_PROPERTIES_LIST = [
    "measure.precision",
    "measure.bands",
]

DEFAULT_PROPERTIES_LIST = (
    DEFAULT_MEASURE_PROPERTIES_LIST + DEFAULT_CONVERGENCE_PROPERTIES_LIST
)


class VerificationWorkChain(SelfCleanWorkChain):
    """The verification workflow to run all test for the given pseudopotential"""

    # This two class attributes will control whether a WF flow is
    # run and results write to outputs ports.
    _VALID_CONGENCENCE_WF = [
        "convergence.cohesive_energy",
        "convergence.phonon_frequencies",
        "convergence.pressure",
        "convergence.delta",
        "convergence.bands",
    ]
    _VALID_MEASURE_WF = [
        "measure.precision",
        "measure.bands",
    ]

    @classmethod
    def define(cls, spec):
        super().define(spec)
        # yapf: disable
        spec.expose_inputs(_BaseMeasureWorkChain, namespace='measure',
                    exclude=['code', 'pseudo', 'options', 'parallelization', 'clean_workdir'])
        spec.expose_inputs(_BaseConvergenceWorkChain, namespace='convergence',
                    exclude=['code', 'pseudo', 'options', 'parallelization', 'clean_workdir'])
        spec.input('pw_code', valid_type=orm.AbstractCode,
                    help='The `pw.x` code use for the `PwCalculation`.')
        spec.input('ph_code', valid_type=orm.AbstractCode, required=False,
                    help='The `ph.x` code use for the `PhCalculation`.')
        spec.input('pseudo', valid_type=UpfData, required=True,
                    help='Pseudopotential to be verified')
        spec.input('wavefunction_cutoff', valid_type=orm.Float, required=False, default=lambda: orm.Float(100.0),
                    help='The wavefunction cutoff for the Measure properties.')
        spec.input('charge_density_cutoff', valid_type=orm.Float, required=False, default=lambda: orm.Float(800.0),
                    help='The charge density cutoff for the Measure properties.')
        spec.input('label', valid_type=orm.Str, required=False,
                    help='label store for display as extra attributes.')
        spec.input('properties_list', valid_type=orm.List,
                    default=lambda: orm.List(list=DEFAULT_PROPERTIES_LIST),
                    help='The preperties will be calculated, passed as a list.')
        spec.input('options', valid_type=orm.Dict, required=False,
                    help='Optional `options`')
        spec.input('parallelization', valid_type=orm.Dict, required=False,
                    help='Parallelization options')

        spec.outline(
            cls.setup_code_resource_options,
            cls.parse_pseudo,
            cls.init_setup,
            if_(cls.is_verify_convergence)(
                if_(cls.is_caching)(
                    cls.run_caching,
                    cls.inspect_caching,
                ),
                cls.run_remote_folder_dependent_convergence,
                cls.run_remote_folder_independent_convergence,
                cls.inspect_convergence,
            ),
            if_(cls.is_verify_measure)(
                cls.run_measure,
                cls.inspect_measure,
            ),
        )
        spec.output('pseudo_info', valid_type=orm.Dict, required=True,
            help='pseudopotential info')
        for wfname in cls._VALID_MEASURE_WF:
            spec.output_namespace(wfname, dynamic=True,
                help=f'results of {wfname} calculation.')
        for wfname in cls._VALID_CONGENCENCE_WF:
            spec.output_namespace(wfname, dynamic=True,
                help=f'results of {wfname} calculation.')

        spec.exit_code(401, 'ERROR_CACHING_ON_BUT_FAILED',
            message='The caching is triggered but failed.')
        spec.exit_code(811, 'WARNING_NOT_ALL_SUB_WORKFLOW_OK',
            message='The sub-workflows {processes} is not finished ok.')
        # yapf: enable

    def setup_code_resource_options(self):
        """
        setup resource options and parallelization for `PwCalculation` from inputs
        """
        if "options" in self.inputs:
            self.ctx.options = self.inputs.options.get_dict()
        else:
            from aiida_sssp_workflow.utils import get_default_options

            self.ctx.options = get_default_options(
                with_mpi=True,
            )

        if "parallelization" in self.inputs:
            self.ctx.parallelization = self.inputs.parallelization.get_dict()
        else:
            self.ctx.parallelization = {}

    @staticmethod
    def _label_from_pseudo_info(pseudo_info) -> str:
        """derive a label string from pseudo_info dict"""
        element = pseudo_info["element"]
        pp_type = pseudo_info["pp_type"]
        z_valence = pseudo_info["z_valence"]

        return f"{element}.{pp_type}.z_{z_valence}"

    def parse_pseudo(self):
        """parse pseudo"""
        pseudo_info = parse_pseudo_info(self.inputs.pseudo)
        self.ctx.pseudo_info = pseudo_info.get_dict()
        self.node.base.extras.set_many(
            self.ctx.pseudo_info
        )  # set the extra attributes for the node

        self.out("pseudo_info", pseudo_info)

    def init_setup(self):
        """prepare inputs for all verification process"""

        if "label" in self.inputs:
            label = self.inputs.label.value
        else:
            label = self._label_from_pseudo_info(self.ctx.pseudo_info)

        self.node.base.extras.set("label", label)

        # Properties list
        valid_list = self._VALID_MEASURE_WF + self._VALID_CONGENCENCE_WF
        self.ctx.properties_list = [
            p for p in self.inputs.properties_list.get_list() if p in valid_list
        ]

        # Measure workflow: bands measure and precision measure workflows inputs setting
        measure_inputs = self.exposed_inputs(_BaseMeasureWorkChain, namespace="measure")
        measure_inputs["pseudo"] = self.inputs.pseudo
        measure_inputs["code"] = self.inputs.pw_code
        measure_inputs["options"] = self.inputs.options
        measure_inputs["parallelization"] = self.inputs.parallelization

        measure_inputs["clean_workdir"] = self.inputs.clean_workdir

        self.ctx.measure_inputs = {
            "precision": measure_inputs.copy(),
            "bands": measure_inputs.copy(),
        }

        # Convergence inputs setting, the properties of convergence test are:
        # 1. cohesive energy
        # 2. phonon frequencies
        # 3. pressue
        # 4. delta
        # 5. bands distance
        self.ctx.convergence_inputs = dict()

        convergence_inputs = self.exposed_inputs(
            _BaseConvergenceWorkChain, namespace="convergence"
        )
        convergence_inputs["code"] = self.inputs.pw_code
        convergence_inputs["pseudo"] = self.inputs.pseudo
        convergence_inputs["options"] = self.inputs.options
        convergence_inputs["parallelization"] = self.inputs.parallelization

        convergence_inputs["clean_workdir"] = self.inputs.clean_workdir

        for prop in ["cohesive_energy", "delta", "pressure"]:
            self.ctx.convergence_inputs[prop] = convergence_inputs.copy()

        # Here, the shallow copy can be used since the type of convergence_inputs
        # is AttributesDict.
        # The deepcopy can't be used, since it will create new data node.
        if "convergence.phonon_frequencies" in self.ctx.properties_list:
            inputs_phonon_frequencies = convergence_inputs.copy()
            inputs_phonon_frequencies.pop("code", None)
            inputs_phonon_frequencies["pw_code"] = self.inputs.pw_code
            inputs_phonon_frequencies["ph_code"] = self.inputs.ph_code
            inputs_phonon_frequencies["clean_workdir"] = orm.Bool(
                False
            )  # For phonon frequencies convergence workflow, the clean dir is taken care by the the finalize step of the verification workflow.

            self.ctx.convergence_inputs[
                "phonon_frequencies"
            ] = inputs_phonon_frequencies

        if "convergence.bands" in self.ctx.properties_list:
            inputs_bands = convergence_inputs.copy()
            inputs_bands["clean_workdir"] = orm.Bool(
                False
            )  # For bands convergence workflow, the clean dir is taken care by the the finalize step of the verification workflow.

            self.ctx.convergence_inputs["bands"] = inputs_bands

        # Caching inputs setting
        # The running strategy of caching is:
        # 1. run phonon_frequencies/bands convergence workflow
        # 2. run cleandir for workchains (which will be the finalize step of phonon_frequencies/bands convergence workflow)
        # 3. run cohesive_energy/pressure/delta convergence workflow which will use the cached data and clean on the fly
        # 4. get the recommended cutoffs
        # 5. run measure workflow using the recommended cutoffs
        self.ctx.caching_inputs = convergence_inputs.copy()
        self.ctx.caching_inputs["clean_workdir"] = orm.Bool(
            False
        )  # shouldn't clean until last, default of _caching but do it here explicitly

        # to collect workchains in a dict
        self.ctx.workchains = dict()

        # For store the finished_ok workflow
        self.ctx.finished_ok_wf = dict()

    def inspect_measure(self):
        """Inspect delta measure results"""
        return self._report_and_results(wname_list=self._VALID_MEASURE_WF)

    def is_verify_convergence(self):
        """Whether to run convergence test workflows"""
        if "_caching" in self.ctx.properties_list:
            # for only run caching workflow
            return True

        for p in self.ctx.properties_list:
            if "convergence" in p:
                return True

        return False

    def is_caching(self):
        """run caching when more than one convergence test"""
        # If the aiida config set pw caching off, then not caching any
        from aiida.manage.caching import get_use_cache

        identifier = "aiida.calculations:quantumespresso.pw"
        return get_use_cache(identifier=identifier)

    def run_caching(self):
        """run pressure verification for caching"""
        running = self.submit(_CachingConvergenceWorkChain, **self.ctx.caching_inputs)
        self.report(
            f"The caching is triggered, submit and run caching "
            f"workchain pk={running.pk} for following convergence test."
            ""
        )

        return ToContext(verify_caching=running)

    def inspect_caching(self):
        """Simply check whether caching run finished okay."""
        workchain = self.ctx.verify_caching

        if not workchain.is_finished_ok:
            return self.exit_codes.ERROR_CACHING_ON_BUT_FAILED

    def _run_convergence(self, plist):
        for property in self.ctx.properties_list:
            property_name = property.split(".")[1]
            if property in plist:
                ConvergenceWorkflow = WorkflowFactory(f"sssp_workflow.{property}")

                running = self.submit(
                    ConvergenceWorkflow, **self.ctx["convergence_inputs"][property_name]
                )
                self.report(
                    f"Submit {property_name} convergence workchain pk={running.pk}"
                )

                self.to_context(_=running)
                self.ctx.workchains[f"{property}"] = running

    def run_remote_folder_dependent_convergence(self):
        """
        running convergence workflow that requires remote_folder not cleaned, e.g. phonon_frequencies, bands
        """
        self._run_convergence(_REMOTE_FOLDER_DEPENDENT_CONVERENCE_PROPERTIES_LIST)

    def run_remote_folder_independent_convergence(self):
        """
        running convergence workflow that requires remote_folder not cleaned, e.g. phonon_frequencies, bands
        """
        self._run_convergence(_REMOTE_FOLDER_INDEPENDENT_CONVERGENCE_PROPERTIES_LIST)

    def inspect_convergence(self):
        """
        inspect the convergence result

        the list set the avaliable convergence workchain that will be inspected
        """
        return self._report_and_results(wname_list=self._VALID_CONGENCENCE_WF)

    def is_verify_measure(self):
        """
        Whether to run measure (delta measure, bands distance} workflow.
        """
        for p in self.ctx.properties_list:
            if "measure" in p:
                return True

        return False

    def run_measure(self):
        """Run delta measure sub-workflow"""
        for property in DEFAULT_MEASURE_PROPERTIES_LIST:
            property_name = property.split(".")[1]
            if property in self.ctx.properties_list:
                MeasureWorkflow = WorkflowFactory(f"sssp_workflow.{property}")

                running = self.submit(
                    MeasureWorkflow, **self.ctx["measure_inputs"][property_name]
                )
                self.report(f"Submit {property_name} measure workchain pk={running.pk}")

                self.to_context(_=running)
                self.ctx.workchains[f"{property}"] = running

    def _report_and_results(self, wname_list):
        """result to respective output namespace"""

        not_finished_ok_wf = {}
        for wname, workchain in self.ctx.workchains.items():
            if wname in wname_list:
                # dump all output as it is to verification workflow output
                self.ctx.finished_ok_wf[wname] = workchain.pk
                for label in workchain.outputs:
                    # output node and namespace -> verification workflow outputs
                    self.out(f"{wname}.{label}", workchain.outputs[label])

                if not workchain.is_finished_ok:
                    self.logger.warning(
                        f"The sub-workflow {wname} pk={workchain.pk} not finished ok."
                    )
                    not_finished_ok_wf[wname] = workchain.pk

        if not_finished_ok_wf:
            return self.exit_codes.WARNING_NOT_ALL_SUB_WORKFLOW_OK.format(
                processes=not_finished_ok_wf
            )

    def on_terminated(self):
        super().on_terminated()

        if not self.inputs.clean_workdir.value:
            self.report(f"{type(self)}: remote folders will not be cleaned")
