# -*- coding: utf-8 -*-
"""
All in one verification workchain
"""
# pylint: disable=cyclic-import
from aiida import orm
from aiida.engine import WorkChain, if_
from aiida.engine.processes.exit_code import ExitCode
from aiida.engine.processes.functions import calcfunction
from aiida.plugins import DataFactory, WorkflowFactory
from plumpy import ToContext

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


DEFAULT_CONVERGENCE_PROPERTIES_LIST = [
    "convergence.cohesive_energy",
    "convergence.phonon_frequencies",
    "convergence.pressure",
    "convergence.delta",
    "convergence.bands",
]

DEFAULT_ACCURACY_PROPERTIES_LIST = [
    "accuracy.delta",
    "accuracy.bands",
]

DEFAULT_PROPERTIES_LIST = (
    ["_caching"]
    + DEFAULT_ACCURACY_PROPERTIES_LIST
    + DEFAULT_CONVERGENCE_PROPERTIES_LIST
)


class VerificationWorkChain(WorkChain):
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
    _VALID_ACCURACY_WF = [
        "accuracy.delta",
        "accuracy.bands",
    ]

    @classmethod
    def define(cls, spec):
        super().define(spec)
        # yapf: disable
        spec.expose_inputs(_BaseMeasureWorkChain, namespace='accuracy',
                    exclude=['code', 'pseudo', 'options', 'parallelization'])
        spec.expose_inputs(_BaseConvergenceWorkChain, namespace='convergence',
                    exclude=['pw_code', 'pseudo', 'options', 'parallelization'])
        spec.input('pw_code', valid_type=orm.Code,
                    help='The `pw.x` code use for the `PwCalculation`.')
        spec.input('ph_code', valid_type=orm.Code,
                    help='The `ph.x` code use for the `PhCalculation`.')
        spec.input('pseudo', valid_type=UpfData, required=True,
                    help='Pseudopotential to be verified')
        spec.input('label', valid_type=orm.Str, required=False,
                    help='label store for display as extra attribut.')
        spec.input('properties_list', valid_type=orm.List,
                    default=lambda: orm.List(list=DEFAULT_PROPERTIES_LIST),
                    help='The preperties will be calculated, passed as a list.')
        spec.input('options', valid_type=orm.Dict, required=False,
                    help='Optional `options`')
        spec.input('parallelization', valid_type=orm.Dict, required=False,
                    help='Parallelization options')
        spec.input('clean_workdir_level', valid_type=orm.Int, default=lambda: orm.Int(1),
                    help='0 for not clean; 1 for clean finished ok workchain; 9 for clean all.')

        spec.outline(
            cls.setup_code_resource_options,
            cls.parse_pseudo,
            cls.init_setup,
            if_(cls.is_verify_accuracy)(
                cls.run_accuracy,
                cls.inspect_accuracy,
            ),
            if_(cls.is_verify_convergence)(
                if_(cls.is_caching)(
                    cls.run_caching,
                    cls.inspect_caching,
                ),
                cls.run_convergence,
                cls.inspect_convergence,
            ),
        )
        spec.output('pseudo_info', valid_type=orm.Dict, required=True,
            help='pseudopotential info')
        for wfname in cls._VALID_ACCURACY_WF:
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

        return f"{element}/z={z_valence}/{pp_type}"

    def parse_pseudo(self):
        """parse pseudo"""
        pseudo_info = parse_pseudo_info(self.inputs.pseudo)
        self.ctx.pseudo_info = pseudo_info.get_dict()
        self.node.set_extra_many(
            self.ctx.pseudo_info
        )  # set the extra attributes for the node

        self.out("pseudo_info", pseudo_info)

    def init_setup(self):
        """prepare inputs for all verification process"""

        if "label" in self.inputs:
            label = self.inputs.label.value
        else:
            label = self._label_from_pseudo_info(self.ctx.pseudo_info)

        self.node.set_extra("label", label)

        # Properties list
        valid_list = self._VALID_ACCURACY_WF + self._VALID_CONGENCENCE_WF
        self.ctx.properties_list = [
            p for p in self.inputs.properties_list.get_list() if p in valid_list
        ]

        # Accuracy workflow: bands measure and delta measure workflows inputs setting
        accurary_inputs = self.exposed_inputs(
            _BaseMeasureWorkChain, namespace="accuracy"
        )
        accurary_inputs["pseudo"] = self.inputs.pseudo
        accurary_inputs["code"] = self.inputs.pw_code
        accurary_inputs["options"] = self.inputs.options
        accurary_inputs["parallelization"] = self.inputs.parallelization

        self.ctx.accuracy_inputs = {
            "delta": accurary_inputs.copy(),
            "bands": accurary_inputs.copy(),
        }

        # Convergence inputs setting, the properties of convergence test are:
        # 1. cohesive energy
        # 2. phonon frequencies
        # 3. pressue
        # 4. delta
        # 5. bands distance
        convergence_inputs = self.exposed_inputs(
            _BaseConvergenceWorkChain, namespace="convergence"
        )
        convergence_inputs["pw_code"] = self.inputs.pw_code
        convergence_inputs["pseudo"] = self.inputs.pseudo
        convergence_inputs["options"] = self.inputs.options
        convergence_inputs["parallelization"] = self.inputs.parallelization

        self.ctx.convergence_inputs = {
            "cohesive_energy": convergence_inputs.copy(),
            "phonon_frequencies": {
                **convergence_inputs.copy(),
                "ph_code": self.inputs.ph_code,
            },
            "pressure": convergence_inputs.copy(),
            "delta": convergence_inputs.copy(),
            "bands": convergence_inputs.copy(),
        }

        self.ctx.caching_inputs = convergence_inputs.copy()

        # to collect workchains in a dict
        self.ctx.workchains = {}

        # For store the finished_ok workflow
        self.ctx.finished_ok_wf = {}

    def is_verify_accuracy(self):
        """
        Whether to run accuracy (delta measure, bands distance} workflow.
        """
        for p in self.ctx.properties_list:
            if "accuracy" in p:
                return True

        return False

    def run_accuracy(self):
        """Run delta measure sub-workflow"""
        for property in DEFAULT_ACCURACY_PROPERTIES_LIST:
            property_name = property.split(".")[1]
            if property in self.ctx.properties_list:
                AccuracyWorkflow = WorkflowFactory(f"sssp_workflow.{property}")

                running = self.submit(
                    AccuracyWorkflow, **self.ctx["accuracy_inputs"][property_name]
                )
                self.report(
                    f"Submit {property_name} accuracy workchain pk={running.pk}"
                )

                self.to_context(_=running)
                self.ctx.workchains[f"{property}"] = running

    def inspect_accuracy(self):
        """Inspect delta measure results"""
        return self._report_and_results(wname_list=self._VALID_ACCURACY_WF)

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
        if not get_use_cache(identifier=identifier):
            return False
        else:
            return True

    def run_caching(self):
        """run pressure verification for caching"""
        ##
        # Pressure as caching
        ##
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

    def run_convergence(self):
        """
        running all verification workflows
        """
        for property in DEFAULT_CONVERGENCE_PROPERTIES_LIST:
            property_name = property.split(".")[1]
            if property in self.ctx.properties_list:
                ConvergenceWorkflow = WorkflowFactory(f"sssp_workflow.{property}")

                running = self.submit(
                    ConvergenceWorkflow, **self.ctx["convergence_inputs"][property_name]
                )
                self.report(
                    f"Submit {property_name} convergence workchain pk={running.pk}"
                )

                self.to_context(_=running)
                self.ctx.workchains[f"{property}"] = running

    def inspect_convergence(self):
        """
        inspect the convergence result

        the list set the avaliable convergence workchain that will be inspected
        """
        return self._report_and_results(wname_list=self._VALID_CONGENCENCE_WF)

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
        """Clean the working directories of all child calculations if `clean_workdir=True` in the inputs."""
        super().on_terminated()

        clean_workdir_level = self.inputs.clean_workdir_level.value
        if clean_workdir_level == 9:
            # extermination all all!!
            cleaned_calcs = self._clean_workdir(self.node)

            if cleaned_calcs:
                self.report(
                    f"cleaned remote folders of calculations: {' '.join(map(str, cleaned_calcs))}"
                )

        elif clean_workdir_level == 0:
            self.report("remote folders will not be cleaned")
            return

        elif clean_workdir_level == 1:
            # only clean finished ok work chain
            from aiida.orm import load_node

            for wname, pk in self.ctx.finished_ok_wf.items():
                node = load_node(pk)
                cleaned_calcs = self._clean_workdir(node)

                for k, calcs in cleaned_calcs.items():
                    self.report(
                        f"cleaned remote folders of calculations {k} "
                        f"[belong to finished_ok work chain {wname}]: {' '.join(map(str, calcs))}"
                    )

            # clean the caching workdir only when phonon_frequencies sub-workflow is finished_ok
            phonon_convergence_workchain = self.ctx.workchains.get(
                "convergence.phonon_frequencies", None
            )
            if (
                phonon_convergence_workchain
                and phonon_convergence_workchain.is_finished_ok
            ):
                try:
                    caching_workchain = self.ctx.verify_caching
                    cleaned_calcs = self._clean_workdir(caching_workchain)

                    for k, calcs in cleaned_calcs.items():
                        self.report(
                            f"cleaned remote folders of calculations {k} "
                            f"[belong to finished_ok work chain _caching]: {' '.join(map(str, calcs))}"
                        )
                except AttributeError:
                    # caching not run
                    self.logger.warning("Caching is not running will not clean it.")

            else:
                self.logger.warning(
                    "Convergence verification of phonon frequecies not run, don't clean caching."
                )

    @staticmethod
    def _clean_workdir(wfnode, include_caching=True):
        """clean the remote folder of all calculation in the workchain node
        return the node pk of cleaned calculation.
        """

        def clean(node: orm.CalcJobNode):
            """clean node workdir"""
            cleaned_calcs_lst = []
            node.outputs.remote_folder._clean()  # pylint: disable=protected-access
            cleaned_calcs_lst.append(called_descendant.pk)

            return cleaned_calcs_lst

        cleaned_calcs = {}
        for called_descendant in wfnode.called_descendants:
            if isinstance(called_descendant, orm.CalcJobNode):
                try:
                    calcs = clean(called_descendant)
                    cleaned_calcs[
                        "menon"
                    ] = calcs  # menon for noumenon for not cached but real one

                    # clean caching node
                    if include_caching:
                        caching_nodes = called_descendant.get_all_same_nodes()
                        if len(caching_nodes) > 1:  # since it always contain the menon
                            for node in caching_nodes:
                                cached_calcs = clean(node)
                                cleaned_calcs["cached"] = cached_calcs

                except (IOError, OSError, KeyError) as exc:
                    raise RuntimeError(
                        "Failed to clean working dirctory of calcjob"
                    ) from exc

        return cleaned_calcs
