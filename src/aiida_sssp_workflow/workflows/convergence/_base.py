# -*- coding: utf-8 -*-
"""
Base convergence workchain

This abstract class give the framework of how to run a convergence test on a given property.
The detail parameters for different properties are defined in the subclass that inherit this base class.
"""

from typing import Union
from pathlib import Path
from abc import ABCMeta, abstractmethod

from aiida import orm
from aiida.engine import append_
from aiida.engine import ProcessBuilder
from aiida_pseudo.data.pseudo import UpfData

from aiida_sssp_workflow.utils import (
    get_default_configuration,
    get_protocol,
    get_standard_structure,
)
from aiida_sssp_workflow.utils.pseudo import extract_pseudo_info
from aiida_sssp_workflow.utils.structure import UNARIE_CONFIGURATIONS
from aiida_sssp_workflow.utils.element import UNSUPPORTED_ELEMENTS
from aiida_sssp_workflow.workflows import SelfCleanWorkChain
from aiida_sssp_workflow.workflows.convergence.report import ConvergenceReport


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
    if not all(isinstance(cutoff, (tuple, list)) for cutoff in cutoff_list):
        return "cutoff_list must be a list of tuples or list."
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
            "report",
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
        spec.exit_code(
            404,
            "ERROR_UNSUPPORTED_ELEMENT",
            message="The PP of element {} is not yet supported to be verified.",
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

    @property
    def protocol(self):
        """Syntax sugar for self.ctx.protocol"""
        if "protocol" not in self.ctx:
            raise AttributeError(
                "protocol is not set in the context, your step must after _setup_protocol"
            )

        return self.ctx.protocol

    @property
    def element(self):
        """Syntax sugar for self.ctx.element"""
        if "element" not in self.ctx:
            raise AttributeError(
                "element is not set in the context, your step must after _setup_pseudos"
            )

        return self.ctx.element

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
        pseudo: Union[Path, UpfData],
        protocol: str,
        cutoff_list: list,
        configuration: str,
        clean_workdir: bool = True,
    ) -> ProcessBuilder:
        """Generate builder for the generic convergence workflow"""
        builder = super().get_builder()
        builder.protocol = orm.Str(protocol)

        # Set the default label and description
        # The default label is set to be the base file name of PP
        # The description include which configuration and which protocol is using.
        builder.metadata.label = (
            pseudo.filename if isinstance(pseudo, UpfData) else pseudo.name
        )
        builder.metadata.description = (
            f"Run on protocol '{protocol}' and configuration '{configuration}'"
        )

        if isinstance(pseudo, Path):
            builder.pseudo = UpfData.get_or_create(pseudo)
        else:
            builder.pseudo = pseudo

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
        # We only has unaries from Z=1 (hydrogen) to Z=96 (curium), so raise an exception
        # if larger Z elements e.g from Bk comes to valified
        if self.element in UNSUPPORTED_ELEMENTS:
            return self.exit_codes.ERROR_UNSUPPORTED_ELEMENT.format(self.element)

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

        # Add link to the called workchain by '{ecutwfc}_{ecutrho}'
        builder.metadata.call_link_label = f"cutoffs_{ecutwfc}_{ecutrho}"

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
            self.logger.warning(
                f"{workchain.process_label} pk={workchain.pk} for reference run is failed."
            )
            # I use exit_code > 1000 as warning so the convergence test
            # continued for other cutoff, and the results can still get to be analyzed.
            # A typical example is in EOS calculation, the birch murnaghan fit is failed (exit_code=1701),
            # but the EOS convergence can still be analyzed with energy-volume pairs.
            if workchain.exit_code.status < 1000:
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

            # Add link to the called workchain by '{ecutwfc}_{ecutrho}'
            builder.metadata.call_link_label = f"cutoffs_{ecutwfc}_{ecutrho}"

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
                "report",
                orm.Dict(dict=validated_report.model_dump()).store(),
            )

    def _finalize(self):
        """Construct a summary report from the convergence report.
        It will contains the analysis of the convergence test, such as the ratio of succuessful runs.
        """
        # TODO: implement the finalize method
