# -*- coding: utf-8 -*-
"""
All in one verification workchain
"""
from typing import Tuple
from pathlib import Path

from aiida import orm
from aiida.engine import if_, ProcessBuilder
from aiida.engine.processes.exit_code import ExitCode
from aiida.engine.processes.functions import calcfunction
from aiida.plugins import WorkflowFactory
from aiida_pseudo.data.pseudo import UpfData

from aiida_sssp_workflow.utils.protocol import generate_cutoff_list, get_protocol
from aiida_sssp_workflow.utils import get_default_mpi_options, parse, serialize_data
from aiida_sssp_workflow.utils.pseudo import PseudoInfo, extract_pseudo_info, get_default_dual
from aiida_sssp_workflow.workflows import SelfCleanWorkChain
from aiida_sssp_workflow.workflows.convergence.report import ConvergenceReport

# XXX: remove me if I am not used
@calcfunction
def parse_pseudo_info(pseudo):
    """parse the pseudo info as a Dict"""
    try:
        info = parse(pseudo.get_content())
    except ValueError:
        return ExitCode(100, "cannot parse the info of pseudopotential.")

    return orm.Dict(dict=info)


DEFAULT_CONVERGENCE_PROPERTIES_LIST = [
    "convergence.cohesive_energy",
    "convergence.pressure",
    "convergence.eos",
    "convergence.bands",
    "convergence.phonon_frequencies",
]

DEFAULT_MEASURE_PROPERTIES_LIST = [
    "transferability.eos",
    "transferability.bands",
]

DEFAULT_PROPERTIES_LIST = (
    DEFAULT_MEASURE_PROPERTIES_LIST + DEFAULT_CONVERGENCE_PROPERTIES_LIST
)

def compute_recommended_cutoffs(workchains: dict, pseudo: UpfData, criteria_name: str='standard'):
    """Input is a dict with workchain name and values are the workchain node,
    loop over the workchain and apply the criteria to get the recommended cutoffs.
    """
    criteria = get_protocol(category='criteria', name=criteria_name)
    success_workchains = {k: w for k, w in workchains.items() if w.is_finished_ok}

    assert len(success_workchains) <= len(DEFAULT_CONVERGENCE_PROPERTIES_LIST)

    if len(success_workchains) == len(DEFAULT_CONVERGENCE_PROPERTIES_LIST):
        # All convergence test are finished correct use the recommended cutoffs from convergence test
        ecutwfc = -1
        ecutrho = -1
        for k, w in success_workchains.items():
            k: str
            property_name = k.split('.')[-1]

            recommended_ecutwfc, recommended_ecutrho = converge_check(w.outputs.report, criteria[property_name])

            ecutwfc = max(ecutwfc, recommended_ecutwfc)
            ecutrho = max(ecutrho, recommended_ecutrho)
    

        return ecutwfc, ecutrho

    else:
        # If not all workchains are okay
        ecutwfc = 200

    # if len(success_workchains) == 0:
    #     # This simply the case when no data
    #     # We set it to 200 as default
    #     ecutwfc = 200

    dual = get_default_dual(pseudo)

    return ecutwfc, ecutwfc * dual

def converge_check(report: ConvergenceReport, criteria: dict) -> Tuple[int, int]:
    """From the report, go through evaluation node of reference and convergence test points,
    compute the convergence behavior of the convergence run and based on the criteria,
    give the recommended cutoff pair.
    It gives pair since it will anchor the evaluation workchain where it converged and write out its cutoff pair.
    So this is suitable for both ecutwfc and ecutrho check out.
    """
    # FIXME: before merge
    return 20, 80


class FullVerificationWorkChain(SelfCleanWorkChain):
    """Full verification work chain include to run convergence test, band structure and EOS verification"""

    # This two class attributes will control whether a WF flow is
    # run and results write to outputs ports.
    _VALID_CONGENCENCE_WF = DEFAULT_CONVERGENCE_PROPERTIES_LIST
    _VALID_MEASURE_WF = DEFAULT_MEASURE_PROPERTIES_LIST
    _CRITERIA = 'v2024.1001'

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input('pw_code', valid_type=orm.AbstractCode,
                    help='The `pw.x` code use for the `PwCalculation`.')
        spec.input('ph_code', valid_type=orm.AbstractCode, required=True,
                    help='The `ph.x` code use for the `PhCalculation`.')
        spec.input('pseudo', valid_type=UpfData, required=True,
                    help='Pseudopotential to be verified')
        spec.input('protocol', valid_type=orm.Str,
                   help='Verification protocol') # XXX: validate, can only be standard, quick, test
        spec.input('curate_type', valid_type=orm.Str, required=True,
                   help='sssp or nc, which oxygen to use') # XXX: validation
        spec.input('dry_run', valid_type=orm.Bool, default=lambda: orm.Bool(False))
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

        spec.outline(
            cls._setup,
            cls._prepare_subworkchain_builders,
            if_(cls._not_dry_run)(
                cls._run_convergence_test,
                cls._inspect_convergence_test,
                cls._set_cutoffs,
                cls._run_transferability_verification,
                cls._inspect_transferability_verification,
            ),
        )

        spec.output('pseudo_info', valid_type=orm.Dict, required=True,
                help='pseudopotential info')
        spec.output_namespace('builders', dynamic=True,
                help='Flat out subworkchain builders info, only output this port when it is in dry run.')
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

    @classmethod
    def get_builder(
        cls,
        pw_code: orm.Code,
        ph_code: orm.Code,
        pseudo: Path,
        protocol: str,
        curate_type: str,
        dry_run: bool = False,
        parallelization: dict | None = None,
        mpi_options: dict | None = None,
        clean_workdir: bool = True,
    ) -> ProcessBuilder:
        builder = super().get_builder()
        builder.protocol = orm.Str(protocol)
        builder.pw_code = pw_code
        builder.ph_code = ph_code
        builder.pseudo = UpfData.get_or_create(pseudo)
        builder.clean_workdir = orm.Bool(clean_workdir)
        builder.curate_type = orm.Str(curate_type)
        builder.dry_run = orm.Bool(dry_run)


        if parallelization:
            builder.parallelization = orm.Dict(parallelization)
        else:
            builder.parallelization = orm.Dict()

        if mpi_options:
            builder.mpi_options = orm.Dict(mpi_options)
        else:
            builder.mpi_options = orm.Dict(get_default_mpi_options())


        return builder

    def _setup(self):
        pseudo: UpfData = self.inputs.pseudo
        pseudo_info: PseudoInfo = extract_pseudo_info(pseudo.get_content())
        self.ctx.element = pseudo_info.element
        self.ctx.pp_type = pseudo_info.type

    def _prepare_subworkchain_builders(self):
        """Use input prepare builder for each property subworkchain
        It will builder as a dict ctx further called `builders` has properties name as key.
        """
        protocol = self.inputs.protocol.value
        mapping_to_convergence = {
            'standard': 'balanced',
            'quick': 'balanced',
            'test': 'test',
        }
        mapping_to_control = {
            'standard': 'standard',
            'quick': 'quick',
            'test': 'test',
        }

        cutoff_list = generate_cutoff_list(mapping_to_control[protocol], self.ctx.element, self.ctx.pp_type)

        builders = {}
        for property in self._VALID_CONGENCENCE_WF:
            _WorkChain = WorkflowFactory(f"sssp_workflow.{property}")
            builder_inputs = {
                "pseudo": self.inputs.pseudo,
                "protocol": mapping_to_convergence[protocol],
                "cutoff_list": cutoff_list,
                "clean_workdir": self.inputs.clean_workdir.value,
            }
            if "phonon_frequencies" in property:
                builder_inputs['pw_code'] = self.inputs.pw_code
                builder_inputs['ph_code'] = self.inputs.ph_code
            else:
                builder_inputs['code'] = self.inputs.pw_code

            # The problem with this setting is nothing is optimized for the atom 
            # and npool is always set to 1.
            # Meanwhile, I don't want to add support to all types of scheduler 
            # (Especially, I am using hyperqueue at the moment which has diffrent mpi_options inputs as slurm)
            # The ultimate solution would be to have a single interface to set for all kinds of schedule.
            if "cohesive_energy" in property:
                builder_inputs['bulk_parallelization'] = self.inputs.parallelization
                builder_inputs['bulk_mpi_options'] = self.inputs.mpi_options
                builder_inputs['atom_parallelization'] = self.inputs.parallelization
                builder_inputs['atom_mpi_options'] = self.inputs.mpi_options
            elif "phonon_frequencies" in property:
                npool = 1 # XXX: Need to be optimized
                builder_inputs['pw_parallelization'] = self.inputs.parallelization
                builder_inputs['pw_mpi_options'] = self.inputs.mpi_options
                builder_inputs['ph_mpi_options'] = self.inputs.mpi_options
                builder_inputs['ph_settings'] = {"CMDLINE": ["-npool", f"{npool}"]}
            else:
                builder_inputs['parallelization'] = self.inputs.parallelization
                builder_inputs['mpi_options'] = self.inputs.mpi_options

            builder: ProcessBuilder = _WorkChain.get_builder(
                **builder_inputs,    
            )

            builders[property] = builder

        mapping_to_eos = {
            'standard': 'standard',
            'quick': 'standard',
            'test': 'test',
        }

        mapping_to_bands = {
            'standard': 'balanced',
            'quick': 'balanced',
            'test': 'test',
        }

        _WorkChain = WorkflowFactory("sssp_workflow.transferability.eos")
        builder: ProcessBuilder = _WorkChain.get_builder(
            code=self.inputs.pw_code,
            pseudo=self.inputs.pseudo,
            protocol=mapping_to_eos[protocol],
            curate_type=self.inputs.curate_type.value,
            clean_workdir=self.inputs.clean_workdir.value,
        )
        builder.parallelization = self.inputs.parallelization
        builder.mpi_options = self.inputs.mpi_options

        builders['transferability.eos'] = builder

        _WorkChain = WorkflowFactory("sssp_workflow.transferability.bands")
        builder: ProcessBuilder = _WorkChain.get_builder(
            code=self.inputs.pw_code,
            pseudo=self.inputs.pseudo,
            protocol=mapping_to_bands[protocol],
            clean_workdir=self.inputs.clean_workdir.value,
        )
        builder.parallelization = self.inputs.parallelization
        builder.mpi_options = self.inputs.mpi_options

        builders['transferability.bands'] = builder

        self.ctx.builders = builders

    def _not_dry_run(self):
        dry_run = self.inputs.dry_run.value

        # Write to the output of all builder for check if it is dry run
        # which is helpful for test and sanity check.
        if dry_run:
            serialized_builders = {k: serialize_data(builder._inputs(prune=True)) for k, builder in self.ctx.builders.items()}

            self.out("builders", serialized_builders)

        return not dry_run

    def _run_convergence_test(self):
        workchains = {}
        for property in self._VALID_CONGENCENCE_WF:
            running = self.submit(self.ctx.builders.get(property))
            self.report(
                f"Submit {property} convergence workchain pk={running.pk}"
            )

            self.to_context(_=running)

            workchains[f"{property}"] = running

        self.ctx.convergence_workchains = workchains

    def _inspect_convergence_test(self):
        self._report_and_results(workchains=self.ctx.convergence_workchains)

    def _set_cutoffs(self):
        """Set cutoffs for the transferability verification, if full convergence
        test are run, then use the maximum cutoff for the transferability run.
        """
        for property in self._VALID_MEASURE_WF:
            wavefunction_cutoff, charge_density_cutoff = compute_recommended_cutoffs(self.ctx.convergence_workchains, self.inputs.pseudo, criteria_name=self._CRITERIA)
            builder = self.ctx.builders.get(property)

            builder['wavefunction_cutoff'] = orm.Int(wavefunction_cutoff)
            builder['charge_density_cutoff'] = orm.Int(charge_density_cutoff)

    def _run_transferability_verification(self):
        """Run delta measure sub-workflow"""
        workchains = {}
        for property in self._VALID_MEASURE_WF:
            running = self.submit(self.ctx.builders.get(property))
            self.report(f"Submit {property} measure workchain pk={running.pk}")

            self.to_context(_=running)
            workchains[f"{property}"] = running

        self.ctx.transferability_workchains = workchains

    def _inspect_transferability_verification(self):
        """Inspect delta measure results"""
        return self._report_and_results(workchains=self.ctx.transferability_workchains)

    def _report_and_results(self, workchains):
        """result to respective output namespace"""

        not_finished_ok_wf = {}
        for wname, workchain in workchains.items():
            # dump all output as it is to verification workflow output
            for label in workchain.outputs:
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
