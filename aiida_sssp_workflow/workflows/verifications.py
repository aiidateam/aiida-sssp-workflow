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

from aiida_sssp_workflow.workflows.legacy_convergence.caching import (
    _CachingConvergenceWorkChain,
)

DeltaFactorWorkChain = WorkflowFactory("sssp_workflow.delta_measure")
ConvergenceCohesiveEnergy = WorkflowFactory(
    "sssp_workflow.legacy_convergence.cohesive_energy"
)
ConvergencePhononFrequencies = WorkflowFactory(
    "sssp_workflow.legacy_convergence.phonon_frequencies"
)
ConvergencePressureWorkChain = WorkflowFactory(
    "sssp_workflow.legacy_convergence.pressure"
)
ConvergenceBandsWorkChain = WorkflowFactory("sssp_workflow.legacy_convergence.bands")

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


DEFAULT_PROPERTIES_LIST = [
    "delta_measure",
    "convergence:cohesive_energy",
    "convergence:phonon_frequencies",
    "convergence:pressure",
]


class VerificationWorkChain(WorkChain):
    """The verification workflow to run all test for the given pseudopotential"""

    _MAX_WALLCLOCK_SECONDS = 1800 * 3

    @classmethod
    def define(cls, spec):
        super().define(spec)
        # yapf: disable
        spec.input('pw_code', valid_type=orm.Code,
                    help='The `pw.x` code use for the `PwCalculation`.')
        spec.input('ph_code', valid_type=orm.Code,
                    help='The `ph.x` code use for the `PhCalculation`.')
        spec.input('pseudo', valid_type=UpfData, required=True,
                    help='Pseudopotential to be verified')
        spec.input('protocol', valid_type=orm.Str, required=True,
                    help='The calculation protocol to use for the workchain.')
        spec.input('criteria', valid_type=orm.Str, required=True,
                    help='Criteria for convergence measurement to give recommend cutoff pair.')
        spec.input('cutoff_control', valid_type=orm.Str, required=True,
                    help='The criteria protocol to use for the workchain.')
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
            if_(cls.is_verify_delta_measure)(
                cls.run_delta_measure,
                cls.inspect_delta_measure,
            ),
            if_(cls.is_verify_convergence)(
                if_(cls.is_caching)(
                    cls.run_caching,
                ),
                cls.run_convergence,
                cls.inspect_convergence,
            ),
        )
        spec.output('pseudo_info', valid_type=orm.Dict, required=True,
            help='pseudopotential info')
        spec.output_namespace('delta_measure', dynamic=True,
                            help='results of delta factor calculation.')
        spec.output_namespace('convergence_cohesive_energy', dynamic=True,
                            help='results of convergence cohesive energy calculation.')
        spec.output_namespace('convergence_phonon_frequencies', dynamic=True,
                            help='results of convergence phonon_frequencies calculation.')
        spec.output_namespace('convergence_pressure', dynamic=True,
                            help='results of convergence pressure calculation.')
        spec.output_namespace('convergence_bands', dynamic=True,
                              help='results of convergence bands calculation.')

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
                max_wallclock_seconds=self._MAX_WALLCLOCK_SECONDS, with_mpi=True
            )

        if "parallelization" in self.inputs:
            self.ctx.parallelization = self.inputs.parallelization.get_dict()
        else:
            self.ctx.parallelization = {}

        self.report(f"resource options set to {self.ctx.options}")
        self.report(f"parallelization options set to {self.ctx.parallelization}")

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

        base_inputs = {
            "pw_code": self.inputs.pw_code,
            "pseudo": self.inputs.pseudo,
            "protocol": self.inputs.protocol,
            "cutoff_control": self.inputs.cutoff_control,
            "options": orm.Dict(dict=self.ctx.options),
            "parallelization": orm.Dict(dict=self.ctx.parallelization),
            "clean_workdir": orm.Bool(
                False
            ),  # not clean for sub-workflow clean at final
        }

        base_conv_inputs = base_inputs.copy()
        base_conv_inputs["criteria"] = self.inputs.criteria

        # Properties list
        self.ctx.properties_list = self.inputs.properties_list.get_list()

        # Delta measure inputs setting
        inputs = base_inputs.copy()
        self.ctx.delta_measure_inputs = inputs

        # Convergence inputs setting
        inputs = base_conv_inputs.copy()
        self.ctx.cohesive_energy_inputs = inputs

        inputs = base_conv_inputs.copy()
        inputs["ph_code"] = self.inputs.ph_code
        self.ctx.phonon_frequencies_inputs = inputs

        inputs = base_conv_inputs.copy()
        self.ctx.pressure_inputs = inputs

        inputs = base_conv_inputs.copy()
        self.ctx.bands_distance_inputs = inputs

        inputs = base_conv_inputs.copy()
        self.ctx.caching_inputs = inputs

        # to collect workchains in a dict
        self.ctx.workchains = {}

    def is_verify_delta_measure(self):
        """
        Whether to run delta measure workflow.
        If properties_list contain `delta_measure` return True.
        """
        return "delta_measure" in self.ctx.properties_list

    def run_delta_measure(self):
        """Run delta measure sub-workflow"""
        ##
        # delta factor
        ##
        running = self.submit(DeltaFactorWorkChain, **self.ctx.delta_measure_inputs)
        self.report(f"submit workchain delta factor pk={running}")

        self.to_context(verify_delta_measure=running)
        self.ctx.workchains["delta_measure"] = running

    def inspect_delta_measure(self):
        """Inspect delta measure results"""
        self._report_and_results(wname_list=["delta_measure"])

    def is_verify_convergence(self):
        """Whether to run convergence test workflows"""
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

        # If less than two convergence wf, no caching run.
        count = 0
        for p in self.ctx.properties_list:
            if "convergence" in p:
                count += 1

        if count < 2:
            return False
        else:
            return True

    def run_caching(self):
        """run pressure verification for caching"""
        ##
        # Pressure as caching
        ##
        running = self.submit(_CachingConvergenceWorkChain, **self.ctx.caching_inputs)
        self.report(f"submit workchain pressure as caching convergence pk={running.pk}")

        self.to_context(verify_caching=running)

    def run_convergence(self):
        """
        running all verification workflows
        """
        ##
        # Cohesive energy
        ##
        if "convergence:cohesive_energy" in self.ctx.properties_list:
            running = self.submit(
                ConvergenceCohesiveEnergy, **self.ctx.cohesive_energy_inputs
            )
            self.report(f"submit workchain cohesive energy convergence pk={running.pk}")

            self.to_context(verify_cohesive_energy=running)
            self.ctx.workchains["convergence_cohesive_energy"] = running

        ##
        # phonon frequencies convergence test
        ##
        if "convergence:phonon_frequencies" in self.ctx.properties_list:
            running = self.submit(
                ConvergencePhononFrequencies, **self.ctx.phonon_frequencies_inputs
            )
            self.report(
                f"submit workchain phonon frequencies convergence pk={running.pk}"
            )

            self.to_context(verify_phonon_frequencies=running)
            self.ctx.workchains["convergence_phonon_frequencies"] = running

        ##
        # Pressure
        ##
        if "convergence:pressure" in self.ctx.properties_list:
            running = self.submit(
                ConvergencePressureWorkChain, **self.ctx.pressure_inputs
            )
            self.report(f"submit workchain pressure convergence pk={running.pk}")

            self.to_context(verify_pressure=running)
            self.ctx.workchains["convergence_pressure"] = running

        # ##
        # # bands
        # ##
        # running = self.submit(ConvergenceBandsWorkChain,
        #                       **self.ctx.bands_distance_inputs)
        # self.report(
        #     f'submit workchain bands distance convergence pk={running.pk}')

        # self.to_context(verify_bands=running)
        # self.ctx.workchains['convergence_bands_distance'] = running

    def inspect_convergence(self):
        """inspect the convergence result"""
        self._report_and_results(
            wname_list=[
                "convergence_cohesive_energy",
                "convergence_phonon_frequencies",
                "convergence_pressure",
            ]
        )

    def _report_and_results(self, wname_list):
        """result to respective output namespace"""

        not_finished_ok_wf = {}
        self.ctx.finished_ok_wf = {}
        for wname, workchain in self.ctx.workchains.items():
            if wname in wname_list:
                # dump all output as it is to verification workflow output
                self.ctx.finished_ok_wf[wname] = workchain.pk
                for label in workchain.outputs:
                    self.out(f"{wname}.{label}", workchain.outputs[label])

                if not workchain.is_finished_ok:
                    self.report(
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

                if cleaned_calcs:
                    self.report(
                        f"cleaned remote folders of calculations "
                        f"[belong to finished_ok work chain {wname}]: {' '.join(map(str, cleaned_calcs))}"
                    )

    @staticmethod
    def _clean_workdir(wfnode):
        """clean the remote folder of all calculation in the workchain node
        return the node pk of cleaned calculation.
        """
        cleaned_calcs = []
        for called_descendant in wfnode.called_descendants:
            if isinstance(called_descendant, orm.CalcJobNode):
                try:
                    called_descendant.outputs.remote_folder._clean()  # pylint: disable=protected-access
                    cleaned_calcs.append(called_descendant.pk)
                except (IOError, OSError, KeyError):
                    pass

        return cleaned_calcs
