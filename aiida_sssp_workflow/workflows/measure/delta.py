# -*- coding: utf-8 -*-
"""Workchain to calculate delta factor of specific psp"""

from aiida import orm
from aiida.engine import if_
from aiida.plugins import DataFactory

from aiida_sssp_workflow.utils import (
    OXIDES_CONFIGURATIONS,
    RARE_EARTH_ELEMENTS,
    UNARIE_CONFIGURATIONS,
    get_protocol,
    get_standard_structure,
    update_dict,
)
from aiida_sssp_workflow.workflows.common import (
    get_extra_parameters_for_lanthanides,
    get_pseudo_N,
    get_pseudo_O,
)
from aiida_sssp_workflow.workflows.evaluate._delta import DeltaWorkChain
from aiida_sssp_workflow.workflows.measure import _BaseMeasureWorkChain
from pseudo_parser.upf_parser import parse_element, parse_pseudo_type

UpfData = DataFactory("pseudo.upf")


class DeltaMeasureWorkChain(_BaseMeasureWorkChain):
    """Workchain to calculate delta factor of specific pseudopotential"""

    # pylint: disable=too-many-instance-attributes

    _OXIDE_CONFIGURATIONS = OXIDES_CONFIGURATIONS
    _UNARIE_CONFIGURATIONS = UNARIE_CONFIGURATIONS

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        # yapf: disable
        super().define(spec)

        spec.outline(
            cls.init_setup,
            if_(cls.is_rare_earth_element)(
                cls.extra_setup_for_rare_earth_element, ),
            cls.setup_pw_parameters_from_protocol,
            cls.run_delta,
            cls.inspect_delta,
            cls.finalize,
        )
        # namespace for storing all detail of run on each configuration
        for configuration in cls._OXIDE_CONFIGURATIONS + cls._UNARIE_CONFIGURATIONS + ["RE"]:
            spec.expose_outputs(DeltaWorkChain, namespace=configuration,
                        namespace_options={
                            'help': f'Delta calculation result of {configuration} EOS.',
                            'required': False,
                        })

        spec.output('output_parameters',
                    help='The summary output parameters of all delta measures to describe the accuracy of EOS compare '
                        ' with the AE equation of state.')
        spec.exit_code(201, 'ERROR_SUB_PROCESS_FAILED_EOS',
                    message=f'The {DeltaWorkChain.__name__} sub process failed.')
        # yapf: enable

    def init_setup(self):
        """
        This step contains all preparation before actaul setup, e.g. set
        the context of element, base_structure, base pw_parameters and pseudos.
        """
        # parse pseudo and output its header information
        content = self.inputs.pseudo.get_content()
        element = parse_element(content)
        pseudo_type = parse_pseudo_type(content)
        self.ctx.element = element
        self.ctx.pseudo_type = pseudo_type
        self.ctx.pseudos_elementary = {element: self.inputs.pseudo}

        pseudo_O = get_pseudo_O()

        self.ctx.pseudos_oxide = {
            element: self.inputs.pseudo,
            "O": pseudo_O,
        }

        self.ctx.pw_parameters = {}

        # Structures for delta factor calculation as provided in
        # http:// molmod.ugent.be/deltacodesdft/
        # Exception for lanthanides use nitride structures from
        # https://doi.org/10.1016/j.commatsci.2014.07.030 and from
        # common-workflow set from
        # https://github.com/aiidateam/commonwf-oxides-scripts/tree/main/0-preliminary-do-not-run/cifs-set2

        # keys here are: BCC, FCC, SC, Diamond, XO, XO2, XO3, X2O, X2O3, X2O5, RE
        # parentatheses means not supported yet.
        self.ctx.structures = {}
        if self.ctx.element == "O":
            # For oxygen, only unaries are available.
            configuration_list = self._UNARIE_CONFIGURATIONS
        else:
            configuration_list = (
                self._OXIDE_CONFIGURATIONS + self._UNARIE_CONFIGURATIONS
            )

        for configuration in configuration_list:
            self.ctx.structures[configuration] = get_standard_structure(
                element,
                prop="delta",
                configuration=configuration,
            )

    def is_rare_earth_element(self):
        """Check if the element is rare earth"""
        return self.ctx.element in RARE_EARTH_ELEMENTS

    def extra_setup_for_rare_earth_element(self):
        """Extra setup for rare earth element"""
        nbnd_factor = 1.5
        pseudo_N = get_pseudo_N()
        pseudo_RE = self.inputs.pseudo
        self.ctx.pseudos_nitride = {"N": pseudo_N, self.ctx.element: pseudo_RE}
        nbnd = nbnd_factor * (pseudo_N.z_valence + pseudo_RE.z_valence)
        self.ctx.pw_nitride_parameters = get_extra_parameters_for_lanthanides(
            self.ctx.element, nbnd
        )

        # set configuration list for rare earth
        self.ctx.structures = {}
        for configuration in self._OXIDE_CONFIGURATIONS + ["RE"]:
            self.ctx.structures[configuration] = get_standard_structure(
                self.ctx.element,
                prop="delta",
                configuration=configuration,
            )

    def setup_pw_parameters_from_protocol(self):
        """Input validation"""
        # pylint: disable=invalid-name, attribute-defined-outside-init

        # Read from protocol if parameters not set from inputs
        protocol = get_protocol(category="delta", name=self.inputs.protocol.value)
        self._DEGAUSS = protocol["degauss"]
        self._OCCUPATIONS = protocol["occupations"]
        self._SMEARING = protocol["smearing"]
        self._CONV_THR = protocol["electron_conv_thr"]
        self.ctx.kpoints_distance = self._KDISTANCE = protocol["kpoints_distance"]
        self.ctx.scale_count = self._SCALE_COUNT = protocol["scale_count"]
        self.ctx.scale_increment = self._SCALE_INCREMENT = protocol["scale_increment"]

        cutoff_control = get_protocol(
            category="control", name=self.inputs.cutoff_control.value
        )
        self._ECUTWFC = cutoff_control["max_wfc"]

        parameters = {
            "SYSTEM": {
                "degauss": self._DEGAUSS,
                "occupations": self._OCCUPATIONS,
                "smearing": self._SMEARING,
            },
            "ELECTRONS": {
                "conv_thr": self._CONV_THR,
            },
        }

        self.ctx.ecutwfc = self._ECUTWFC
        self.ctx.ecutrho = self._ECUTWFC * 8

        self.ctx.pw_parameters = update_dict(self.ctx.pw_parameters, parameters)

        self.logger.info(f"The pw parameters for EOS step is: {self.ctx.pw_parameters}")

    def _get_inputs(self, structure, configuration):
        if configuration in OXIDES_CONFIGURATIONS:
            # pseudos for oxides
            pseudos = self.ctx.pseudos_oxide
            pw_parameters = self.ctx.pw_parameters
            kpoints_distance = self.ctx.kpoints_distance
        if configuration in UNARIE_CONFIGURATIONS:
            # pseudos for BCC, FCC, SC, Diamond
            pseudos = self.ctx.pseudos_elementary
            pw_parameters = self.ctx.pw_parameters
            kpoints_distance = self.ctx.kpoints_distance
        if configuration == "RE":
            # pseudos for nitrides
            pseudos = self.ctx.pseudos_nitride

            parameters = {
                "SYSTEM": {
                    "occupations": "tetrahedra",
                },
                "ELECTRONS": {
                    "conv_thr": self._CONV_THR,
                },
            }
            pw_parameters = update_dict(parameters, self.ctx.pw_nitride_parameters)
            kpoints_distance = self.ctx.kpoints_distance + 1

        inputs = {
            "code": self.inputs.code,
            "pseudos": pseudos,
            "structure": structure,
            "element": orm.Str(self.ctx.element),  # _base wf hold attribute `element`
            "configuration": orm.Str(configuration),
            "pw_base_parameters": orm.Dict(dict=pw_parameters),
            "ecutwfc": orm.Int(self.ctx.ecutwfc),
            "ecutrho": orm.Int(self.ctx.ecutrho),
            "kpoints_distance": orm.Float(kpoints_distance),
            "scale_count": orm.Int(self.ctx.scale_count),
            "scale_increment": orm.Float(self.ctx.scale_increment),
            "options": self.inputs.options,
            "parallelization": self.inputs.parallelization,
        }

        return inputs

    def run_delta(self):
        """run eos workchain"""

        for configuration, structure in self.ctx.structures.items():
            inputs = self._get_inputs(structure, configuration)

            future = self.submit(DeltaWorkChain, **inputs)
            self.report(
                f"launching DeltaWarkChain<{future.pk}> for {configuration} structure."
            )

            self.to_context(**{f"{configuration}_delta": future})

    def inspect_delta(self):
        """Inspect the results of DeltaWorkChain"""
        failed = []
        for configuration in self.ctx.structures.keys():
            workchain = self.ctx[f"{configuration}_delta"]

            if not workchain.is_finished_ok:
                self.logger.warning(
                    f"DeltaWorkChain of {configuration} failed with exit status {workchain.exit_status}"
                )
                failed.append(configuration)

            self.out_many(
                self.exposed_outputs(
                    workchain,
                    DeltaWorkChain,
                    namespace=configuration,
                )
            )

        if failed:
            pass
            # TODO ERROR

    def finalize(self):
        """calculate the delta factor"""
        output_parameters = {}

        for configuration in (
            self._OXIDE_CONFIGURATIONS + self._UNARIE_CONFIGURATIONS + ["RE"]
        ):
            try:
                output = self.outputs[configuration].get("output_parameters")
            except KeyError:
                self.logger.warning(
                    f"Can not get the key {configuration} from outputs, not verify or failed."
                )
                continue

            output_parameters[configuration] = {
                "delta": output["delta"],
                "nu": output["rel_errors_vec_length"],
            }

        self.out("output_parameters", orm.Dict(dict=output_parameters).store())
