# -*- coding: utf-8 -*-
"""
Bands distance of many input pseudos
"""

from pathlib import Path

from aiida import orm
from aiida.plugins import DataFactory
from aiida.engine import ToContext, ProcessBuilder

from aiida_sssp_workflow.utils import (
    get_protocol,
    get_standard_structure,
)
from aiida_sssp_workflow.utils import get_default_mpi_options, extract_pseudo_info
from aiida_sssp_workflow.utils.structure import (
    UNARIE_CONFIGURATIONS,
    get_default_configuration,
)
from aiida_sssp_workflow.utils.element import UNSUPPORTED_ELEMENTS
from aiida_sssp_workflow.workflows.evaluate._bands import (
    BandsWorkChain as EvaluateBandsWorkChain,
)
from aiida_sssp_workflow.workflows.measure import _BaseMeasureWorkChain
from aiida_sssp_workflow.workflows.measure.report import BandStructureReport

UpfData = DataFactory("pseudo.upf")


def validate_input_pseudos(d_pseudos, _):
    """Validate that all input pseudos map to same element"""
    element = set(pseudo.element for pseudo in d_pseudos.values())

    if len(element) > 1:
        return f"The pseudos corespond to different elements {element}."


def is_valid_convergence_configuration(value, _=None):
    """Check if the configuration is valid"""
    # XXX: I am duplicate from _base.py, combine us
    valid_configurations = UNARIE_CONFIGURATIONS
    if value not in valid_configurations:
        return f"Configuration {value} is not valid. Valid configurations are {valid_configurations}"


class BandStructureWorkChain(_BaseMeasureWorkChain):
    """WorkChain to run bands measure,
    run without sym for distance compare and band structure along the path
    """

    _EVALUATE_WORKCHAIN = EvaluateBandsWorkChain

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        # yapf: disable
        super().define(spec)
        spec.input(
            "configuration",
            valid_type=orm.Str,
            required=False,
            validator=is_valid_convergence_configuration,
            help="The configuration to use for the workchain, can be DC/BCC/FCC/SC.",
        )

        spec.outline(
            cls._setup_pseudos,
            cls._setup_protocol,
            cls._setup_structure,
            cls.run_bands,
            cls.inspect_bands,
            cls._finalize,
        )

        spec.expose_outputs(EvaluateBandsWorkChain)
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
        spec.output( # XXX: I am same as transferibility workchain. combine me.
            "report",
            valid_type=orm.Dict,
            required=True,
            help="The output report of convergence verification, it is a dict contains the full information of convergence test, the mapping of cutoffs to the UUID of the evaluation workchain etc.",
        )

    def _setup_pseudos(self):
        """Setup pseudos"""
        # XXX: this is same as trans WF, consider combine
        pseudo_info = extract_pseudo_info(
            self.inputs.pseudo.get_content(),
        )
        self.ctx.element = pseudo_info.element
        self.ctx.pseudos = {self.ctx.element: self.inputs.pseudo}

    @property
    def element(self):
        """Syntax sugar for self.ctx.element"""
        # TODO: same as from convergence/_base, consider combine
        if "element" not in self.ctx:
            raise AttributeError(
                "element is not set in the context, your step must after _setup_pseudos"
            )

        return self.ctx.element

    @property
    def pseudos(self):
        """Syntax sugar for self.ctx.pseudos"""
        # TODO: same as from convergence/_base, consider combine
        if "pseudos" not in self.ctx:
            raise AttributeError(
                "pseudos is not set in the context, your step must after _setup_pseudos"
            )

        return self.ctx.pseudos

    def _setup_protocol(self):
        """unzip and parse protocol parameters to context"""
        # XXX: this is same as trans WF, consider combine
        protocol = get_protocol(category="bands", name=self.inputs.protocol.value)
        self.ctx.protocol = protocol

    @property
    def protocol(self):
        """Syntax sugar for self.ctx.protocol"""
        # XXX: this is same as trans WF, consider combine
        if "protocol" not in self.ctx:
            raise AttributeError(
                "protocol is not set in the context, your step must after _setup_protocol"
            )

        return self.ctx.protocol

    def _setup_structure(self):
        """Set up the configuration to be run for band structure"""
        # We only has unaries from Z=1 (hydrogen) to Z=96 (curium), so raise an exception
        # if larger Z elements e.g from Bk comes to valified
        if self.element in UNSUPPORTED_ELEMENTS:
            return self.exit_codes.ERROR_UNSUPPORTED_ELEMENT.format(self.element)

        if "configuration" in self.inputs:
            configuration = self.inputs.configuration
        else:
            # will use the default configuration set in the protocol (mapping.json)
            configuration = get_default_configuration(
                self.ctx.element, property="bands"
            )

        self.ctx.structure = get_standard_structure(
            self.ctx.element, configuration=configuration
        )

        self.out("configuration", configuration)
        self.out("structure", self.ctx.structure)

    @property
    def structure(self):
        """Syntax sugar to get self structure"""
        return self.ctx.structure

    @classmethod
    def get_builder(
        cls,
        pseudo: Path | UpfData,
        protocol: str,
        wavefunction_cutoff: float,
        charge_density_cutoff: float,
        code: orm.AbstractCode,
        configuration: list | None = None,
        parallelization: dict | None = None,
        mpi_options: dict | None = None,
        clean_workdir: bool = True,  # default to clean workdir
    ) -> ProcessBuilder:
        """Return a builder to run this EOS convergence workchain"""
        builder = super().get_builder()
        builder.protocol = orm.Str(protocol)
        if isinstance(pseudo, Path):
            builder.pseudo = UpfData.get_or_create(pseudo)
        else:
            builder.pseudo = pseudo
        builder.wavefunction_cutoff = orm.Float(wavefunction_cutoff)
        builder.charge_density_cutoff = orm.Float(charge_density_cutoff)
        builder.code = code

        if configuration is not None:
            builder.configuration = orm.Str(configuration)

        # Set the default label and description
        # The default label is set to be the base file name of PP
        # The description include which configuration and which protocol is using.
        builder.metadata.call_link_label = "band_structure_verification"
        builder.metadata.label = (
            pseudo.filename if isinstance(pseudo, UpfData) else pseudo.name
        )
        builder.metadata.description = (
            f"""Run on protocol '{protocol}' | configuration '{configuration if configuration is not None else "default"}' | """
            f" base (ecutwfc, ecutrho) = ({wavefunction_cutoff}, {charge_density_cutoff})"
        )
        builder.clean_workdir = orm.Bool(clean_workdir)
        builder.code = code

        if parallelization:
            builder.parallelization = orm.Dict(parallelization)
        else:
            builder.parallelization = orm.Dict()

        if mpi_options:
            builder.mpi_options = orm.Dict(mpi_options)
        else:
            builder.mpi_options = orm.Dict(get_default_mpi_options())

        return builder

    def prepare_evaluate_builder(self):
        """Get inputs for the bands evaluation with given pseudo"""
        # Read from protocol if parameters not set from inputs
        ecutwfc = self.inputs.wavefunction_cutoff.value
        ecutrho = self.inputs.charge_density_cutoff.value

        protocol = self.protocol

        builder = self._EVALUATE_WORKCHAIN.get_builder()

        builder.clean_workdir = (
            self.inputs.clean_workdir
        )  # sync with the main workchain

        builder.structure = self.structure
        natoms = len(self.structure.sites)

        scf_pw_parameters = {
            "CONTROL": {
                "calculation": "scf",
            },
            "SYSTEM": {
                "degauss": protocol["degauss"],
                "occupations": protocol["occupations"],
                "smearing": protocol["smearing"],
                "ecutwfc": round(ecutwfc),
                "ecutrho": round(ecutrho),
            },
            "ELECTRONS": {
                "conv_thr": protocol["conv_thr_per_atom"] * natoms,
                "mixing_beta": protocol["mixing_beta"],
            },
        }
        builder.scf.pw["code"] = self.inputs.code
        builder.scf.pw["pseudos"] = self.pseudos
        builder.scf.pw["parameters"] = orm.Dict(scf_pw_parameters)
        builder.scf.pw["parallelization"] = self.inputs.parallelization
        builder.scf.pw["metadata"]["options"] = self.inputs.mpi_options.get_dict()
        builder.scf.kpoints_distance = orm.Float(protocol["kpoints_distance"])

        bands_pw_parameters = {
            "CONTROL": {
                "calculation": "bands",
            },
            "SYSTEM": {
                "degauss": protocol["degauss"],
                "occupations": protocol["occupations"],
                "smearing": protocol["smearing"],
                "ecutwfc": round(ecutwfc),
                "ecutrho": round(ecutrho),
            },
            "ELECTRONS": {
                "conv_thr": protocol["conv_thr_per_atom"] * natoms,
                "mixing_beta": protocol["mixing_beta"],
            },
        }

        builder.bands.pw["code"] = self.inputs.code
        builder.bands.pw["pseudos"] = self.pseudos
        builder.bands.pw["parameters"] = orm.Dict(bands_pw_parameters)
        builder.bands.pw["parallelization"] = self.inputs.parallelization
        builder.bands.pw["metadata"]["options"] = self.inputs.mpi_options.get_dict()

        # Generic
        builder.kpoints_distance_bands = orm.Float(protocol["kpoints_distance"])
        builder.kpoints_distance_band_structure = orm.Float(
            protocol["kpoints_distance_bs"]
        )
        builder.init_nbands_factor = orm.Int(protocol["init_nbands_factor"])
        builder.fermi_shift = orm.Float(protocol["fermi_shift"])
        builder.run_band_structure = orm.Bool(True)

        return builder

    def run_bands(self):
        """run bands evaluation"""
        builder = self.prepare_evaluate_builder()

        running = self.submit(builder)

        self.report(f"launching BandsWorkChain<{running.pk}>")

        return ToContext(bands=running)

    def inspect_bands(self):
        """inspect bands run results"""
        bands_workchain = self.ctx.bands
        self.out_many(self.exposed_outputs(bands_workchain, EvaluateBandsWorkChain))

        outgoing: orm.LinkManager = bands_workchain.base.links.get_outgoing()
        band_structure_node = outgoing.get_node_by_label("band_structure")

        # The band node into record is the one with largest bands_factor
        # ['band_with_factor_x'] -> {'band_with_factor_x': x} and sort
        bands_label = sorted(
            {
                k: int(k.split("_")[-1])
                for k in outgoing.all_link_labels()
                if "bands_with_factor" in k
            },
            key=lambda x: x[1],
        )[-1]
        bands_node = outgoing.get_node_by_label(bands_label)

        band_dict = {
            "bands": {
                "uuid": bands_node.uuid,
                "exit_status": bands_node.exit_status,
            },
            "band_structure": {
                "uuid": band_structure_node.uuid,
                "exit_status": band_structure_node.exit_status,
            },
        }

        try:
            validated_report = BandStructureReport.construct(band_dict)
            self.report("BandStructureReport report is validated")
        except Exception as e:
            self.report(f"BandStructureReport in sot validated: {e}")
            raise e
        else:
            self.out("report", orm.Dict(dict=validated_report.model_dump()).store())

    def _finalize(self):
        """Final"""
        # TODO:
