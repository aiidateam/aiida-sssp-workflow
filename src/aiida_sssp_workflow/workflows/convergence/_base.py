# -*- coding: utf-8 -*-
"""
Base convergence workchain

This abstract class give the framework of how to run a convergence test on a given property.
The detail parameters for different properties are defined in the subclass that inherit this base class.
"""

from pathlib import Path
from abc import ABCMeta, abstractmethod
from pydantic import BaseModel

from aiida import orm
from aiida.engine import append_
from aiida.plugins import DataFactory
from aiida.engine import ProcessBuilder

from aiida_sssp_workflow.utils import (
    get_default_configuration,
    get_protocol,
    get_standard_structure,
)
from aiida_sssp_workflow.utils.pseudo import extract_pseudo_info
from aiida_sssp_workflow.utils.structure import UNARIE_CONFIGURATIONS
from aiida_sssp_workflow.workflows import SelfCleanWorkChain

UpfData = DataFactory("pseudo.upf")


# TODO: moved to the report module
class PointRunReportEntry(BaseModel):
    uuid: str
    wavefunction_cutoff: int
    charge_density_cutoff: int
    exit_status: int


class ConvergenceReport(BaseModel):
    reference: PointRunReportEntry
    convergence_list: list[PointRunReportEntry]

    @classmethod
    def construct(cls, reference: dict, convergence_list: list[dict]):
        """Construct the ConvergenceReport from dict data."""

        return cls(
            reference=PointRunReportEntry(**reference),
            convergence_list=[
                PointRunReportEntry(**entry) for entry in convergence_list
            ],
        )


class abstract_attribute(object):
    """lazy variable check: https://stackoverflow.com/a/32536493"""

    def __get__(self, obj, type):
        for cls in type.__mro__:
            for name, value in cls.__dict__.items():
                if value is self:
                    this_obj = obj if obj else type
                    raise NotImplementedError(
                        f"{this_obj!r} does not have the attribute {name!r} (abstract from class {cls.__name__!r})"
                    )

        raise NotImplementedError(
            "%s does not set the abstract attribute <unknown>", type.__name__
        )


def is_valid_convergence_configuration(value, _=None):
    """Check if the configuration is valid"""
    valid_configurations = UNARIE_CONFIGURATIONS
    if value not in valid_configurations:
        return f"Configuration {value} is not valid. Valid configurations are {valid_configurations}"


def is_valid_cutoff_list(cutoff_list, _=None):
    """Check the cutoff list is a list of tuples and the cutoffs are increasing"""
    if not all(isinstance(cutoff, tuple) for cutoff in cutoff_list):
        return "cutoff_list must be a list of tuples"
    if not all(
        cutoff_list[i][0] < cutoff_list[i + 1][0] for i in range(len(cutoff_list) - 1)
    ):
        return "cutoff_list must be a list of tuples with increasing ecutwfc"

    if not all(
        cutoff_list[i][1] < cutoff_list[i + 1][1] for i in range(len(cutoff_list) - 1)
    ):
        return "cutoff_list must be a list of tuples with increasing ecutrho"


class _BaseConvergenceWorkChain(SelfCleanWorkChain):
    """Base convergence workchain class for wavefunction cutoff convergence test.
    This is a abstract class and should be subclassed to implement the methods for specific convergence workflow.
    The work chain will run on a series of wavefunction cutoffs on the same dual for charge density cutoff.
    The convergence will be measured by the property defined in the convergence protocol.
    The target property is defined in the `_PROPERTY_NAME` class attribute by the class that inherit this base class.
    """

    __metaclass__ = ABCMeta

    _PROPERTY_NAME = abstract_attribute()  # used to get convergence protocol

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input(
            "pseudo",
            valid_type=UpfData,
            required=True,
            help="Pseudopotential to be verified",
        )
        spec.input(
            "protocol",
            valid_type=orm.Str,
            required=True,
            help="The calculation protocol to use for the workchain.",
        )
        spec.input(
            "cutoff_list",
            valid_type=orm.List,
            required=True,
            validator=is_valid_cutoff_list,
            help="The list (tuple with ecutwfc and ecutrho in Ry) of cutoffs to test. The last one is used as reference.",
        )
        spec.input(
            "configuration",
            valid_type=orm.Str,
            required=False,
            validator=is_valid_convergence_configuration,
            help="The configuration to use for the workchain, can be DC/BCC/FCC/SC.",
        )

        spec.outline(
            cls._setup_pseudos,
            cls._setup_structure,
            cls._setup_protocol,
            cls.run_reference,
            cls.inspect_reference,
            cls.run_convergence,
            cls.inspect_convergence,
            cls._finalize,
        )

        spec.output(
            "convergence_report",
            valid_type=orm.Dict,
            required=True,
            help="The output report of convergence verification, it is a dict contains the full information of convergence test, the mapping of cutoffs to the UUID of the evaluation workchain etc.",
        )
        spec.output(
            "extra_report",
            valid_type=orm.Dict,
            required=False,
            help="Extra report for specific convergence test or additional reference calculation, it is currently only used in pressure convergence workflow.",
        )
        spec.output(
            "configuration",
            valid_type=orm.Str,
            required=True,
            help="The configuration used for the convergence.",
        )
        spec.output(
            "structure",
            valid_type=orm.StructureData,
            required=True,
            help="The structure used for the convergence.",
        )

        spec.exit_code(
            401,
            "ERROR_REFERENCE_CALCULATION_FAILED",
            message="The reference calculation failed.",
        )
        spec.exit_code(
            402,
            "ERROR_SUB_PROCESS_FAILED",
            message="The sub process for `{label}` did not finish successfully.",
        )

    @property
    def structure(self):
        """Syntax sugar for self.ctx.structure"""
        if "structure" not in self.ctx:
            raise AttributeError(
                "structure is not set in the context, your step must after _setup_structure"
            )

        return self.ctx.structure

    @property
    def pseudos(self):
        """Syntax sugar for self.ctx.pseudos"""
        if "pseudos" not in self.ctx:
            raise AttributeError(
                "pseudos is not set in the context, your step must after _setup_pseudos"
            )

        return self.ctx.pseudos

    @abstractmethod
    def prepare_evaluate_builder(self, ecutwfc: int, ecutrho: int) -> dict:
        """Generate builder for the evaluate workflow.
        It must compatible with inputs format of every evaluate workflow.
        This is used in actual submit of reference and convergence tests.

        Since the ecutwfc and ecutrho has less points set to be a float with decimals.
        Therefore, always pass round to nearst int inputs for ecutwfc and ecutrho.
        It also bring the advantage that the inputs to final calcjob always the same
        for these two inputs and therefore caching will be properly triggered.
        """

    @classmethod
    def get_builder(
        cls,
        pseudo: Path,
        protocol: str,
        cutoff_list: list,
        configuration: str,
        clean_workdir: bool = True,
    ) -> ProcessBuilder:
        """Generate builder for the generic convergence workflow"""
        builder = super().get_builder()
        builder.protocol = orm.Str(protocol)
        builder.pseudo = UpfData.get_or_create(pseudo)

        if ret := is_valid_cutoff_list(cutoff_list):
            raise ValueError(ret)

        if ret := is_valid_convergence_configuration(configuration):
            raise ValueError(ret)

        builder.cutoff_list = orm.List(list=cutoff_list)
        builder.configuration = orm.Str(configuration)
        builder.clean_workdir = orm.Bool(clean_workdir)

        return builder

    def _setup_pseudos(self):
        """Setup pseudos"""
        pseudo_info = extract_pseudo_info(
            self.inputs.pseudo.get_content(),
        )
        self.ctx.element = pseudo_info.element
        self.ctx.pseudos = {self.ctx.element: self.inputs.pseudo}

    def _setup_structure(self):
        """Setup structure from input or use the default structure
        The convergence behavior may be different for different structure BCC/FCC/SC/DC.
        We use DC structure as default since it is the one that most hard to converge.
        See the paper (TODO: add arxiv link).
        """
        if "configuration" in self.inputs:
            configuration = self.inputs.configuration
        else:
            # will use the default configuration set in the protocol (mapping.json)
            configuration = get_default_configuration(
                self.ctx.element, property="convergence"
            )

        self.ctx.structure = get_standard_structure(
            self.ctx.element, configuration=configuration
        )

        self.out("configuration", configuration)
        self.out("structure", self.ctx.structure)

    def _setup_protocol(self):
        """unzip and parse protocol parameters to context"""
        protocol = get_protocol(category="convergence", name=self.inputs.protocol.value)
        self.ctx.protocol = {
            **protocol["base"],
            **protocol.get(
                self._PROPERTY_NAME,
                dict(),  # if _PROPERTY_NAME not set, simply use base
            ),
        }

    def run_reference(self):
        """Run the reference calculation on the highest cutoff pair in the list

        If there are more calculation need to run as reference, this method should be overrided.
        It happens for pressure convergence workflow, where the EOS calculation in run for reference.
        """
        ecutwfc, ecutrho = self.inputs.cutoff_list[-1]

        # Round the cutoff to the nearest integer to avoid the float comparison
        # This is important for the caching mechanism
        ecutwfc, ecutrho = round(ecutwfc), round(ecutrho)
        builder = self.prepare_evaluate_builder(ecutwfc=ecutwfc, ecutrho=ecutrho)

        running = self.submit(builder)
        running.base.extras.set("wavefunction_cutoff", ecutwfc)
        running.base.extras.set("charge_density_cutoff", ecutrho)

        self.report(
            f"launching reference calculation: {running.process_label}<{running.pk}>"
        )

        self.to_context(reference=running)

    def inspect_reference(self):
        """Inspect the reference calculation and check if it is finished successfully.

        It may also need to be overrided if additional calculations are run for reference.
        """
        try:
            workchain = self.ctx.reference
        except AttributeError as e:
            raise RuntimeError("Reference evaluation is not triggered") from e

        if not workchain.is_finished_ok:
            self.report(
                f"{workchain.process_label} pk={workchain.pk} for reference run is failed."
            )
            return self.exit_codes.ERROR_REFERENCE_CALCULATION_FAILED

    def run_convergence(self):
        """
        run on all other evaluation sample points
        """
        for ecutwfc, ecutrho in self.inputs.cutoff_list[
            :-1
        ]:  # The last one is reference
            ecutwfc, ecutrho = round(ecutwfc), round(ecutrho)
            builder = self.prepare_evaluate_builder(ecutwfc=ecutwfc, ecutrho=ecutrho)

            running = self.submit(builder)
            self.report(
                f"launching fix ecutrho={ecutrho} [ecutwfc={ecutwfc}] {running.process_label}<{running.pk}>"
            )

            # add ecutwfc and ecutrho as extras of running node
            running.base.extras.set("wavefunction_cutoff", ecutwfc)
            running.base.extras.set("charge_density_cutoff", ecutrho)

            self.to_context(children_convergence=append_(running))

    def inspect_convergence(self):
        # include reference node in the last
        self.ctx.children_convergence.append(self.ctx.reference)

        # Construct the report from the convergence runs
        reference_report = {
            "uuid": self.ctx.reference.uuid,
            "wavefunction_cutoff": self.ctx.reference.base.extras.get(
                "wavefunction_cutoff"
            ),
            "charge_density_cutoff": self.ctx.reference.base.extras.get(
                "charge_density_cutoff"
            ),
            "exit_status": self.ctx.reference.exit_status,
        }

        convergence_reports = []
        for child in self.ctx.children_convergence:
            # The convergence runs are evaluate workchain inhrited from _BaseEvaluateWorkChain
            # So the results are sure to have wavefunction_cutoff and charge_density_cutoff as outputs.
            ecutwfc = child.base.extras.get("wavefunction_cutoff")
            ecutrho = child.base.extras.get("charge_density_cutoff")

            if child.exit_status == 0:
                self.report(
                    f"{child.process_label} pk={child.pk} finished successfully with cutoffs ecutwfc={ecutwfc} ecutrho={ecutrho}."
                )
            else:
                self.report(
                    f"{child.process_label} pk={child.pk} failed with exit status {child.exit_status}"
                )

            _report = {
                "uuid": child.uuid,
                "wavefunction_cutoff": ecutwfc,
                "charge_density_cutoff": ecutrho,
                "exit_status": child.exit_status,
            }

            convergence_reports.append(_report)

        try:
            validated_report = ConvergenceReport.construct(
                reference_report, convergence_reports
            )
            self.report("Convergence report is validated.")
        except Exception as e:
            self.report(f"Convergence report is not validated: {e}")
            raise e
        else:
            self.out(
                "convergence_report",
                orm.Dict(dict=validated_report.model_dump()).store(),
            )

    def _finalize(self):
        """Construct a summary report from the convergence report.
        It will contains the analysis of the convergence test, such as the ratio of succuessful runs.
        """
        # TODO: implement the finalize method
