# -*- coding: utf-8 -*-
"""Workchain to calculate delta factor of specific psp"""

from aiida import orm
from aiida.engine import if_
from aiida.plugins import DataFactory

from aiida_sssp_workflow.utils import (
    ACTINIDE_ELEMENTS,
    HIGH_DUAL_ELEMENTS,
    LANTHANIDE_ELEMENTS,
    MAGNETIC_ELEMENTS,
    NO_GS_CONF_ELEMENTS,
    OXIDE_CONFIGURATIONS,
    UNARIE_CONFIGURATIONS,
    get_magnetic_inputs,
    get_protocol,
    get_standard_structure,
    reset_pseudos_for_magnetic,
    update_dict,
)
from aiida_sssp_workflow.workflows.common import (
    get_extra_parameters_for_lanthanides,
    get_pseudo_element_and_type,
    get_pseudo_N,
    get_pseudo_O,
)
from aiida_sssp_workflow.workflows.evaluate._metric import MetricWorkChain
from aiida_sssp_workflow.workflows.measure import _BaseMeasureWorkChain
from pseudo_parser.upf_parser import parse_element, parse_pseudo_type

UpfData = DataFactory("pseudo.upf")


class PrecisionMeasureWorkChain(_BaseMeasureWorkChain):
    """Workchain to calculate delta factor of specific pseudopotential"""

    # pylint: disable=too-many-instance-attributes

    _OXIDE_CONFIGURATIONS = OXIDE_CONFIGURATIONS
    _UNARIE_GS_CONFIGURATIONS = UNARIE_CONFIGURATIONS + ["GS"]

    _NBANDS_FACTOR_FOR_LAN = 1.5

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        # yapf: disable
        super().define(spec)

        spec.outline(
            cls.setup,
            cls.setup_pw_parameters_from_protocol,
            cls.setup_configurations,
            cls.run_metric,
            cls.inspect_metric,
            cls.finalize,
        )
        # namespace for storing all detail of run on each configuration
        for configuration in cls._OXIDE_CONFIGURATIONS + cls._UNARIE_GS_CONFIGURATIONS + ["RE"]:
            spec.expose_outputs(MetricWorkChain, namespace=configuration,
                        namespace_options={
                            'help': f'Delta calculation result of {configuration} EOS.',
                            'required': False,
                        })

        spec.output('output_parameters',
                    help='The summary output parameters of all delta measures to describe the precision of EOS compare '
                        ' with the AE equation of state.')
        spec.exit_code(401, 'ERROR_METRIC_WORKCHAIN_NOT_FINISHED_OK', message='The metric workchain of configuration {confs} not finished ok.')
        # yapf: enable

    def _setup_pseudo_and_configuration(self):
        """Depend on the element set the proper pseudo and configuration list"""

        # this is the pseudo dict for the element
        self.ctx.pseudos_unary = {self.ctx.element: self.inputs.pseudo}

        # for the oxide, need to pseudo of oxygen,
        # the pseudo is the one select after the oxygen verification and
        # store in the `statics/upf/O.**.upf`
        pseudo_O = get_pseudo_O()
        self.ctx.pseudos_oxide = {
            self.ctx.element: self.inputs.pseudo,
            "O": pseudo_O,
        }

        # For oxygen, still run for oxides but use only the pseudo.
        if self.ctx.element == "O":
            self.ctx.pseudos_oxide = {
                self.ctx.element: self.inputs.pseudo,
            }

        # Structures for delta factor calculation as provided in
        # http:// molmod.ugent.be/deltacodesdft/
        # Exception for lanthanides use nitride structures from
        # https://doi.org/10.1016/j.commatsci.2014.07.030 and from
        # common-workflow set from acwf paper xsf files all store in `statics/structures`.
        # keys here are: BCC, FCC, SC, Diamond, XO, XO2, XO3, X2O, X2O3, X2O5, RE (Lanthanide that will use RE-N), GS
        if self.ctx.element in NO_GS_CONF_ELEMENTS + ACTINIDE_ELEMENTS:
            # Don't have ground state structure for At, Fr, Ra
            self.ctx.configuration_list = (
                self._OXIDE_CONFIGURATIONS + UNARIE_CONFIGURATIONS
            )
        elif self.ctx.element in LANTHANIDE_ELEMENTS:
            self.ctx.configuration_list = (
                self._OXIDE_CONFIGURATIONS + UNARIE_CONFIGURATIONS + ["RE"]
            )
        else:
            self.ctx.configuration_list = (
                self._OXIDE_CONFIGURATIONS + self._UNARIE_GS_CONFIGURATIONS
            )

        # set structures except RARE earth element and actinides elements with will be set independently
        # in sepecific step. Other wise, the gs structure is request but not provided, which
        # will raise error.
        self.ctx.structures = dict()
        for configuration in self.ctx.configuration_list:
            self.ctx.structures[configuration] = get_standard_structure(
                self.ctx.element,
                prop="delta",
                configuration=configuration,
            )

    def setup(self):
        """
        This step contains all preparation before actaul setup, e.g. set
        the context of element, base_structure, base pw_parameters and pseudos.
        """
        # parse pseudo and output its header information
        content = self.inputs.pseudo.get_content()
        self.ctx.element = parse_element(content)
        self.ctx.pseudo_type = parse_pseudo_type(content)

        self.ctx.pw_parameters = {}

        self._setup_pseudo_and_configuration()

        # set up the ecutwfc and ecutrho
        self.ctx.ecutwfc = self.inputs.wavefunction_cutoff.value
        self.ctx.ecutrho = self.inputs.charge_density_cutoff.value

    def setup_pw_parameters_from_protocol(self):
        """Input validation"""
        # pylint: disable=invalid-name, attribute-defined-outside-init

        # Read from protocol if parameters not set from inputs
        protocol = get_protocol(category="precision", name=self.inputs.protocol.value)
        self._DEGAUSS = protocol["degauss"]
        self._OCCUPATIONS = protocol["occupations"]
        self._SMEARING = protocol["smearing"]
        self._CONV_THR_PER_ATOM = protocol["conv_thr_per_atom"]
        self._MIXING_BETA = protocol["mixing_beta"]
        self.ctx.kpoints_distance = self._KDISTANCE = protocol["kpoints_distance"]
        self.ctx.scale_count = self._SCALE_COUNT = protocol["scale_count"]
        self.ctx.scale_increment = self._SCALE_INCREMENT = protocol["scale_increment"]

        parameters = {
            "CONTROL": {
                "calculation": "scf",
                "disk_io": "nowf",  # safe to hard-code, this will never be the parent calculation of other calculations
            },
            "SYSTEM": {
                "degauss": self._DEGAUSS,
                "occupations": self._OCCUPATIONS,
                "smearing": self._SMEARING,
            },
            "ELECTRONS": {
                "conv_thr": self._CONV_THR_PER_ATOM,
                "mixing_beta": self._MIXING_BETA,
            },
        }

        self.ctx.pw_parameters = update_dict(self.ctx.pw_parameters, parameters)

        self.logger.info(f"The pw parameters for EOS step is: {self.ctx.pw_parameters}")

    def setup_configurations(self):
        # narrow the configuration list by protocol
        # this is used for test protocol which only has limited configurations to be verified
        if "configurations" in self.inputs:
            clist = self.inputs.configurations.get_list()
        else:
            clist = self.ctx.configuration_list

        for key in list(self.ctx.structures.keys()):
            if key not in clist:
                self.ctx.structures.pop(key)

    @staticmethod
    def _compute_nelectrons_of_oxide(configuration, z_O, z_X):
        """Will return the number of electrons of oxide configurations with pseudos
        z_O is the number of electrons of oxygen pseudo
        z_X is the number of electrons of X pseudo
        """
        if configuration == "XO":
            return z_X + z_O

        elif configuration == "XO2":
            return z_X + z_O * 2

        elif configuration == "XO3":
            return z_X + z_O * 3

        elif configuration == "X2O":
            return z_X * 2 + z_O

        elif configuration == "X2O3":
            return z_X * 4 + z_O * 6

        elif configuration == "X2O5":
            return z_X * 4 + z_O * 10

        else:
            raise ValueError(
                f"Cannot compute the number electrons of configuration {configuration} with z_O={z_O} and z_X={z_X}."
            )

    def _get_inputs(self, structure, configuration):
        """Set pw parameters and pseudos based on configuration and structure"""
        if configuration in self._OXIDE_CONFIGURATIONS:
            # pseudos for oxides
            pseudos = self.ctx.pseudos_oxide
            pw_parameters = self.ctx.pw_parameters
            kpoints_distance = self.ctx.kpoints_distance

            # need also increase nbands for Rare-earth oxides.
            # See https://github.com/aiidateam/aiida-sssp-workflow/issues/161
            # This is not easy to be set in the rare-earth step since it will
            # finally act on here
            if self.ctx.element in LANTHANIDE_ELEMENTS:
                nbnd_factor = self._NBANDS_FACTOR_FOR_LAN
                pseudo_O = get_pseudo_O()
                pseudo_RE = self.inputs.pseudo
                nbnd = (
                    nbnd_factor
                    * int(
                        self._compute_nelectrons_of_oxide(
                            configuration, pseudo_O.z_valence, pseudo_RE.z_valence
                        )
                    )
                    // 2
                )
                pw_parameters["SYSTEM"]["nbnd"] = int(nbnd)

        if configuration in self._UNARIE_GS_CONFIGURATIONS:  # include regular 'GS'
            # pseudos for BCC, FCC, SC, Diamond and TYPYCAL configurations
            pseudos = self.ctx.pseudos_unary
            pw_parameters = self.ctx.pw_parameters
            kpoints_distance = self.ctx.kpoints_distance

        if configuration == "GS" and self.ctx.element in MAGNETIC_ELEMENTS:
            # specific setting for magnetic elements gs since mag on

            # reconstruct configuration, O1, O2 for sites
            (
                structure,
                pw_magnetic_parameters,
            ) = get_magnetic_inputs(structure)

            # override pseudos setting
            # required for O, Mn, Cr where the kind names varies for sites
            self.ctx.pseudos_magnetic = reset_pseudos_for_magnetic(
                self.inputs.pseudo, structure
            )

            pseudos = self.ctx.pseudos_magnetic
            pw_parameters = update_dict(self.ctx.pw_parameters, pw_magnetic_parameters)

        if configuration == "RE":
            # pseudos for nitrides
            pseudo_N = get_pseudo_N()
            pseudo_RE = self.inputs.pseudo
            self.ctx.pseudos_nitride = {"N": pseudo_N, self.ctx.element: pseudo_RE}
            pseudos = self.ctx.pseudos_nitride

            # perticular parameters for RE-N
            # Since the reference data is from https://doi.org/10.1016/j.commatsci.2014.07.030
            # Here I need to use the same input parameters
            nbnd_factor = self._NBANDS_FACTOR_FOR_LAN
            nbnd = nbnd_factor * (pseudo_N.z_valence + pseudo_RE.z_valence)

            pw_parameters = self.ctx.pw_parameters
            # Set the namespace directly will override the original value set in `self.ctx.pw_parameters`
            pw_parameters = update_dict(
                self.ctx.pw_parameters,
                get_extra_parameters_for_lanthanides(self.ctx.element, nbnd),
            )
            pw_parameters["SYSTEM"]["occupations"] = "tetrahedra"
            pw_parameters["SYSTEM"].pop("smearing")

            # sparse kpoints, we use tetrahedra occupation
            kpoints_distance = self.ctx.kpoints_distance + 0.1

        ecutwfc, ecutrho = self._get_pw_cutoff(
            structure, self.ctx.ecutwfc, self.ctx.ecutrho
        )
        parameters = {
            "SYSTEM": {
                "ecutwfc": round(ecutwfc, 1),
                "ecutrho": round(ecutrho, 1),
            },
        }
        parameters = update_dict(parameters, pw_parameters)

        # conv_thr is extensive, like the total energy so we need to scale it with the number of atoms
        natoms = len(structure.sites)
        parameters["ELECTRONS"]["conv_thr"] = (
            parameters["ELECTRONS"]["conv_thr"] * natoms
        )

        inputs = {
            "eos": {
                "metadata": {"call_link_label": "delta_EOS"},
                "structure": structure,
                "kpoints_distance": orm.Float(kpoints_distance),
                "scale_count": orm.Int(self.ctx.scale_count),
                "scale_increment": orm.Float(self.ctx.scale_increment),
                "pw": {
                    "code": self.inputs.code,
                    "pseudos": pseudos,
                    "parameters": orm.Dict(dict=parameters),
                    "metadata": {
                        "options": self.inputs.options.get_dict(),
                    },
                    "parallelization": self.inputs.parallelization,
                },
            },
            "element": orm.Str(self.ctx.element),
            "configuration": orm.Str(configuration),
            "clean_workdir": self.inputs.clean_workdir,
        }

        return inputs

    def run_metric(self):
        """run eos workchain"""

        for configuration, structure in self.ctx.structures.items():
            inputs = self._get_inputs(structure, configuration)

            future = self.submit(MetricWorkChain, **inputs)
            self.report(
                f"launching DeltaWarkChain<{future.pk}> for {configuration} structure."
            )

            self.to_context(**{f"{configuration}_metric": future})

    def inspect_metric(self):
        """Inspect the results of MetricWorkChain"""
        failed_configuration_lst = list()
        for configuration in self.ctx.structures.keys():
            workchain = self.ctx[f"{configuration}_metric"]

            if not workchain.is_finished_ok:
                self.logger.warning(
                    f"MetricWorkChain of {configuration} failed with exit status {workchain.exit_status}"
                )
                failed_configuration_lst.append(configuration)

            self.out_many(
                self.exposed_outputs(
                    workchain,
                    MetricWorkChain,
                    namespace=configuration,
                )
            )

        if failed_configuration_lst:
            return self.exit_codes.ERROR_METRIC_WORKCHAIN_NOT_FINISHED_OK.format(
                confs=f"{failed_configuration_lst}",
            )

    def finalize(self):
        """calculate the delta factor"""
        output_parameters = {}

        for configuration in self.ctx.configuration_list:
            try:
                output = self.outputs[configuration].get("output_parameters")
            except KeyError:
                self.logger.warning(
                    f"Can not get the key {configuration} from outputs, not verify or failed."
                )
                continue

            try:
                output_parameters[configuration] = {
                    "delta": output["delta"],
                    "delta/natoms": output["delta/natoms"],
                    "nu": output["rel_errors_vec_length"],
                }
            except KeyError:
                self.logger.warning(
                    f"Can not get the metric, check EOS result or directly recalculate metric from EOS."
                )
                continue

        self.out("output_parameters", orm.Dict(dict=output_parameters).store())
