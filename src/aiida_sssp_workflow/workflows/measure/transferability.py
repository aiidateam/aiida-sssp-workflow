# -*- coding: utf-8 -*-
"""Workchain to calculate delta factor of specific psp"""

from pathlib import Path

from aiida import orm
from aiida.engine import ProcessBuilder
from aiida_pseudo.data.pseudo import UpfData

from aiida_sssp_workflow.utils import (
    ACTINIDE_ELEMENTS,
    LANTHANIDE_ELEMENTS,
    NO_GS_CONF_ELEMENTS,
    OXIDE_CONFIGURATIONS,
    UNARIE_CONFIGURATIONS,
    get_protocol,
    get_standard_structure,
)
from aiida_sssp_workflow.utils import get_default_mpi_options
from aiida_sssp_workflow.utils.pseudo import (
    extract_pseudo_info,
    compute_total_nelectrons,
    get_pseudo_O,
    CurateType,
)
from aiida_sssp_workflow.workflows.evaluate._metric import MetricWorkChain
from aiida_sssp_workflow.workflows.measure import _BaseMeasureWorkChain
from aiida_sssp_workflow.workflows.measure.report import TransferabilityReport


class EOSTransferabilityWorkChain(_BaseMeasureWorkChain):
    """Workchain run EOS on 10 structures and compute nu/delta metric factor"""

    # pylint: disable=too-many-instance-attributes

    _OXIDE_CONFIGURATIONS = OXIDE_CONFIGURATIONS

    # _UNARIE_GS_CONFIGURATIONS = UNARIE_CONFIGURATIONS + ["GS"]
    # For now, we decide not include the GS configuration since the reference data from sci 2016 paper use
    # the different parameters compared with the aiida common workflow. So the result will be different.
    _UNARIE_GS_CONFIGURATIONS = UNARIE_CONFIGURATIONS

    _NBANDS_FACTOR_FOR_LAN = 2.0

    _EVALUATE_WORKCHAIN = MetricWorkChain

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        super().define(spec)

        spec.outline(
            cls._setup_pseudos,
            cls._setup_protocol,
            cls._setup_configurations,
            cls.run_transferability,
            cls.inspect_transferability,
            cls._finalize,
        )
        # namespace for storing all detail of run on each configuration
        for configuration in (
            cls._OXIDE_CONFIGURATIONS + cls._UNARIE_GS_CONFIGURATIONS + ["RE"]
        ):
            spec.expose_outputs(
                MetricWorkChain,
                namespace=configuration,
                namespace_options={
                    "help": f"Delta calculation result of {configuration} EOS.",
                    "required": False,
                },
            )

        spec.output(
            "report",
            valid_type=orm.Dict,
            required=True,
            help="The output report of convergence verification, it is a dict contains the full information of convergence test, the mapping of cutoffs to the UUID of the evaluation workchain etc.",
        )
        spec.exit_code(
            401,
            "ERROR_METRIC_WORKCHAIN_NOT_FINISHED_OK",
            message="The metric workchain of configuration {confs} not finished ok.",
        )

    def _setup_configurations(self):
        """Get the configuration and the corresponding structure to run."""
        # Structures for delta factor calculation as provided in
        # http:// molmod.ugent.be/deltacodesdft/
        # Exception for lanthanides use nitride structures from
        # https://doi.org/10.1016/j.commatsci.2014.07.030 and from
        # common-workflow set from acwf paper xsf files all store in `statics/structures`.
        # keys here are: BCC, FCC, SC, Diamond, XO, XO2, XO3, X2O, X2O3, X2O5, RE (Lanthanide that will use RE-N), GS
        # XXX: at the moment all three if..else cases give the same configuration_list since we didn't consider the GS configuration.
        if self.ctx.element in NO_GS_CONF_ELEMENTS + ACTINIDE_ELEMENTS:
            # Don't have ground state structure for At, Fr, Ra
            # We didn't consider the ground state structure from sci.2016 in transferability verification.
            self.ctx.configuration_list = (
                self._OXIDE_CONFIGURATIONS + UNARIE_CONFIGURATIONS
            )
        elif self.ctx.element in LANTHANIDE_ELEMENTS:
            self.ctx.configuration_list = (
                self._OXIDE_CONFIGURATIONS
                + UNARIE_CONFIGURATIONS  # TODO: add back?? + ["RE"]
            )
        else:
            self.ctx.configuration_list = (
                self._OXIDE_CONFIGURATIONS + UNARIE_CONFIGURATIONS
            )

        # narrow the configuration from input by popping up confs not passing into
        if "configurations" in self.inputs:
            clist = self.inputs.configurations.get_list()
        else:
            clist = self.ctx.configuration_list

        # set structures except RARE earth element and actinides elements with will be set independently
        # in sepecific step. Other wise, the gs structure is request but not provided, which
        # will raise error.
        self.ctx.configuration_structure_mapping = dict()
        for c in clist:
            self.ctx.configuration_structure_mapping[c] = get_standard_structure(
                self.ctx.element,
                configuration=c,
            )

        for key in list(self.ctx.configuration_structure_mapping.keys()):
            if key not in clist:
                self.ctx.configuration_structure_mapping.pop(key)

        # Used for _finalize check
        self.ctx.final_verified_configurations = clist

    def _setup_pseudos(self):
        """Setup pseudos"""
        pseudo_info = extract_pseudo_info(
            self.inputs.pseudo.get_content(),
        )
        self.ctx.element = pseudo_info.element
        self.ctx.pseudos = {self.ctx.element: self.inputs.pseudo}

        # this is the pseudo dict for the element
        self.ctx.pseudos_unary = {self.ctx.element: self.inputs.pseudo}

        pseudo_O = self.inputs.oxygen_pseudo

        self.ctx.pseudos_oxide = {
            self.ctx.element: self.inputs.pseudo,
            "O": pseudo_O,
        }

        # For oxygen, still run for oxides but use only the pseudo.
        if self.ctx.element == "O":
            self.ctx.pseudos_oxide = {
                "O": self.inputs.pseudo,
            }

    def get_pseudos(self, configuration) -> dict:
        """Syntax sugar to get the pseudos from configuration"""
        if configuration in self._OXIDE_CONFIGURATIONS:
            # pseudos for oxides
            pseudos = self.ctx.pseudos_oxide

        elif configuration in self._UNARIE_GS_CONFIGURATIONS:
            # pseudos for BCC, FCC, SC, Diamond and TYPYCAL configurations
            pseudos = self.ctx.pseudos_unary

        else:
            raise ValueError(f"can not find pseudos for {configuration}")

        return pseudos

    def _setup_protocol(self):
        """unzip and parse protocol parameters to context"""
        protocol = get_protocol(
            category="transferability", name=self.inputs.protocol.value
        )
        self.ctx.protocol = protocol

    @property
    def protocol(self):
        """Syntax sugar for self.ctx.protocol"""
        if "protocol" not in self.ctx:
            raise AttributeError(
                "protocol is not set in the context, your step must after _setup_protocol"
            )

        return self.ctx.protocol

    @classmethod
    def get_builder(
        cls,
        pseudo: Path | UpfData,
        protocol: str,
        wavefunction_cutoff: float,
        charge_density_cutoff: float,
        code: orm.AbstractCode,
        configurations: list | None = None,
        curate_type: str | None = None,  # sssp -> pslib O; nc -> dojo O
        oxygen_pseudo: Path | UpfData | None = None,
        oxygen_ecutwfc: float | None = None,
        oxygen_ecutrho: float | None = None,
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

        if configurations is not None:
            builder.configurations = orm.List(configurations)

        # When both oxygen_pseudo and curate_type are set, use oxygen_pseudo which is more explicit
        if oxygen_pseudo is not None:
            if oxygen_ecutwfc is None or oxygen_ecutrho is None:
                raise ValueError(
                    "oxygen_ecutwfc and oxygen_ecutrho need to pass along with oxygen_pseudo."
                )

            _oxygen_pseudo = oxygen_pseudo
            _oxygen_ecutwfc = orm.Float(oxygen_ecutwfc)
            _oxygen_ecutrho = orm.Float(oxygen_ecutrho)
        elif curate_type is not None:
            match curate_type.lower():
                case "sssp":
                    ct = CurateType.SSSP
                case "nc":
                    ct = CurateType.NC
                case _:
                    ct = curate_type

            _oxygen_pseudo, _oxygen_ecutwfc, _oxygen_ecutrho = get_pseudo_O(ct)
        else:
            raise ValueError("Set at least curate_type or oxygen_pseudo.")

        if isinstance(_oxygen_pseudo, Path):
            builder.oxygen_pseudo = UpfData.get_or_create(_oxygen_pseudo)
        else:
            builder.oxygen_pseudo = _oxygen_pseudo

        builder.oxygen_ecutwfc = orm.Float(_oxygen_ecutwfc)
        builder.oxygen_ecutrho = orm.Float(_oxygen_ecutrho)

        # Set the default label and description
        # The default label is set to be the base file name of PP
        # The description include which configuration and which protocol is using.
        builder.metadata.call_link_label = "transferability_eos"
        builder.metadata.label = (
            pseudo.filename if isinstance(pseudo, UpfData) else pseudo.name
        )
        builder.metadata.description = (
            f"""Run on protocol '{protocol}' | configurations '{configurations if configurations is not None else "all"}' | """
            f"with oxygen_pseudo '{builder.oxygen_pseudo.filename}' | base (ecutwfc, ecutrho) = ({wavefunction_cutoff}, {charge_density_cutoff})"
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

    def prepare_evaluate_builder(
        self, configuration: str, structure: orm.StructureData
    ):
        """Set pw parameters and pseudos based on configuration and structure"""
        # set up the ecutwfc and ecutrho
        ecutwfc = self.inputs.wavefunction_cutoff.value
        ecutrho = self.inputs.charge_density_cutoff.value

        # Read from protocol if parameters not set from inputs
        protocol = self.protocol
        natoms = len(structure.sites)

        # Cutoff is depend on the structure
        ecutwfc, ecutrho = self._get_pw_cutoff(
            structure,
            ecutwfc,
            ecutrho,
        )

        pw_parameters = {
            "CONTROL": {
                "calculation": "scf",
                "disk_io": "nowf",
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

        pseudos = self.get_pseudos(configuration)

        # Increase nbands for Rare-earth oxides which help the electronic step convergence.
        # See https://github.com/aiidateam/aiida-sssp-workflow/issues/161
        # This is not easy to be set in the rare-earth step since it will
        # finally act on here
        if (
            self.ctx.element in LANTHANIDE_ELEMENTS
            and configuration in self._OXIDE_CONFIGURATIONS
        ):
            nbnd_factor = self._NBANDS_FACTOR_FOR_LAN
            nbnd = (
                nbnd_factor
                * int(
                    compute_total_nelectrons(
                        configuration,
                        pseudos,
                    )
                )
                // 2
            )
            pw_parameters["SYSTEM"]["nbnd"] = int(nbnd)

        builder = self._EVALUATE_WORKCHAIN.get_builder()

        builder.clean_workdir = (
            self.inputs.clean_workdir
        )  # sync with the main workchain

        builder.element = orm.Str(self.ctx.element)
        builder.configuration = orm.Str(configuration)

        builder.eos.metadata.call_link_label = "transferability_EOS"
        builder.eos.structure = structure
        builder.eos.kpoints_distance = orm.Float(protocol["kpoints_distance"])
        builder.eos.scale_count = orm.Int(protocol["scale_count"])
        builder.eos.scale_increment = orm.Float(protocol["scale_increment"])

        # pw
        builder.eos.pw["code"] = self.inputs.code
        builder.eos.pw["pseudos"] = pseudos
        builder.eos.pw["parameters"] = orm.Dict(dict=pw_parameters)
        builder.eos.pw["parallelization"] = self.inputs.parallelization
        builder.eos.pw["metadata"]["options"] = self.inputs.mpi_options.get_dict()

        return builder

        # if configuration == "GS" and self.ctx.element in MAGNETIC_ELEMENTS:
        #     # specific setting for magnetic elements gs since mag on
        #
        #     # reconstruct configuration, O1, O2 for sites
        #     (
        #         structure,
        #         pw_magnetic_parameters,
        #     ) = get_magnetic_inputs(structure)
        #
        #     # override pseudos setting
        #     # required for O, Mn, Cr where the kind names varies for sites
        #     self.ctx.pseudos_magnetic = reset_pseudos_for_magnetic(
        #         self.inputs.pseudo, structure
        #     )
        #
        #     pseudos = self.ctx.pseudos_magnetic
        #     pw_parameters = update_dict(self.ctx.pw_parameters, pw_magnetic_parameters)
        #
        # if configuration == "RE":
        #     # pseudos for nitrides
        #     pseudo_N = get_pseudo_N()
        #     pseudo_RE = self.inputs.pseudo
        #     self.ctx.pseudos_nitride = {"N": pseudo_N, self.ctx.element: pseudo_RE}
        #     pseudos = self.ctx.pseudos_nitride
        #
        #     # perticular parameters for RE-N
        #     # Since the reference data is from https://doi.org/10.1016/j.commatsci.2014.07.030
        #     # Here I need to use the same input parameters
        #     nbnd_factor = self._NBANDS_FACTOR_FOR_LAN
        #     nbnd = nbnd_factor * (pseudo_N.z_valence + pseudo_RE.z_valence)
        #
        #     pw_parameters = self.ctx.pw_parameters
        #     # Set the namespace directly will override the original value set in `self.ctx.pw_parameters`
        #     pw_parameters = update_dict(
        #         self.ctx.pw_parameters,
        #         get_extra_parameters_for_lanthanides(self.ctx.element, nbnd),
        #     )
        #     pw_parameters["SYSTEM"]["occupations"] = "tetrahedra"
        #     pw_parameters["SYSTEM"].pop("smearing")
        #
        #     # sparse kpoints, we use tetrahedra occupation
        #     kpoints_distance = self.ctx.kpoints_distance + 0.1

    def run_transferability(self):
        """run eos workchain"""

        for (
            configuration,
            structure,
        ) in self.ctx.configuration_structure_mapping.items():
            builder = self.prepare_evaluate_builder(configuration, structure)

            future = self.submit(builder)

            self.report(
                f"launching DeltaWarkChain<{future.pk}> for {configuration} structure."
            )

            self.to_context(**{f"{configuration}_metric": future})

    def inspect_transferability(self):
        """Inspect the results of MetricWorkChain"""
        failed_configuration_lst = list()
        transferability_reports = {}
        for configuration in self.ctx.configuration_structure_mapping.keys():
            child = self.ctx[f"{configuration}_metric"]

            if not child.is_finished_ok:
                self.logger.warning(
                    f"MetricWorkChain of {configuration} failed with exit status {child.exit_status}"
                )
                failed_configuration_lst.append(configuration)

            self.out_many(
                self.exposed_outputs(
                    child,
                    MetricWorkChain,
                    namespace=configuration,
                )
            )

            _report = {
                "uuid": child.uuid,
                "exit_status": child.exit_status,
            }

            transferability_reports[configuration] = _report

        if failed_configuration_lst:
            return self.exit_codes.ERROR_METRIC_WORKCHAIN_NOT_FINISHED_OK.format(
                confs=f"{failed_configuration_lst}",
            )

        try:
            validated_report = TransferabilityReport.construct(transferability_reports)
            self.report("TransferabilityReport report is validated.")
        except Exception as e:
            self.report(f"TransferabilityReport report is not validated: {e}")
            raise e
        else:
            self.out(
                "report",
                orm.Dict(dict=validated_report.model_dump()).store(),
            )

    def _finalize(self):
        """calculate the delta factor"""
        # TODO: see what need to be added here
