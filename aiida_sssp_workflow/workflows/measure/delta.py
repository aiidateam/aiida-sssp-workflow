# -*- coding: utf-8 -*-
"""Workchain to calculate delta factor of specific psp"""
import importlib

from aiida import orm
from aiida.engine import WorkChain
from aiida.plugins import DataFactory

from aiida_sssp_workflow.utils import (
    OXIDES_CONFIGURATIONS,
    RARE_EARTH_ELEMENTS,
    UNARIE_CONFIGURATIONS,
    get_protocol,
    get_standard_structure,
    update_dict,
)
from aiida_sssp_workflow.workflows.evaluate._delta import DeltaWorkChain
from pseudo_parser.upf_parser import parse_element, parse_pseudo_type

UpfData = DataFactory("pseudo.upf")


class DeltaMeasureWorkChain(WorkChain):
    """Workchain to calculate delta factor of specific pseudopotential"""

    # pylint: disable=too-many-instance-attributes

    _OXIDE_CONFIGURATIONS = OXIDES_CONFIGURATIONS
    _UNARIE_CONFIGURATIONS = UNARIE_CONFIGURATIONS

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        # yapf: disable
        super().define(spec)
        spec.input('pw_code', valid_type=orm.Code,
                    help='The `pw.x` code use for the `PwCalculation`.')
        spec.input('pseudo', valid_type=UpfData, required=True,
                    help='Pseudopotential to be verified')
        spec.input('protocol', valid_type=orm.Str, required=True,
                    help='The protocol which define input calculation parameters.')
        spec.input('cutoff_control', valid_type=orm.Str, default=lambda: orm.Str('standard'),
                    help='The control protocol where define max_wfc.')
        spec.input('options', valid_type=orm.Dict, required=False,
                    help='Optional `options` to use for the `PwCalculations`.')
        spec.input('parallelization', valid_type=orm.Dict, required=False,
                    help='Parallelization options for the `PwCalculations`.')
        spec.input('clean_workdir', valid_type=orm.Bool, default=lambda: orm.Bool(False),
                    help='If `True`, work directories of all called calculation will be cleaned at the end of execution.')
        spec.outline(
            cls.init_setup,
            cls.setup_pw_parameters_from_protocol,
            cls.setup_pw_resource_options,
            cls.run_delta,
            cls.inspect_delta,
            cls.finalize,
        )
        # namespace for storing all detail of run on each configuration
        for configuration in cls._OXIDE_CONFIGURATIONS + cls._UNARIE_CONFIGURATIONS:
            spec.expose_outputs(DeltaWorkChain, namespace=configuration,
                        namespace_options={'help': f'Delta calculation result of {configuration} EOS.'})

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

        # Import oxygen pseudopotential file and set the pseudos
        import_path = importlib.resources.path(
            "aiida_sssp_workflow.statics.upf", "O.pbe-n-kjpaw_psl.0.1.upf"
        )
        with import_path as pp_path, open(pp_path, "rb") as stream:
            pseudo_O = UpfData(stream)

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

        # keys here are: BCC, FCC, SC, Diamond, XO, XO2, XO3, X2O, X2O3, X2O5
        # parentatheses means not supported yet.
        self.ctx.structures = {}
        if self.ctx.element == "O":
            # For oxygen, only unaries are available.
            configuration_list = self._UNARIE_CONFIGURATIONS
        elif self.ctx.element is RARE_EARTH_ELEMENTS:
            # For lanthanides, oxides are verifid
            # TODO: add lanthanides nitrides
            configuration_list = self._OXIDE_CONFIGURATIONS
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

    def setup_pw_resource_options(self):
        """
        setup resource options and parallelization for `PwCalculation` from inputs
        """
        if "options" in self.inputs:
            self.ctx.options = self.inputs.options.get_dict()
        else:
            from aiida_sssp_workflow.utils import get_default_options

            self.ctx.options = get_default_options(with_mpi=True)

        if "parallelization" in self.inputs:
            self.ctx.parallelization = self.inputs.parallelization.get_dict()
        else:
            self.ctx.parallelization = {}

    def _get_inputs(self, structure, configuration):
        if "O" in configuration:
            # pseudos for oxides
            pseudos = self.ctx.pseudos_oxide
        else:
            pseudos = self.ctx.pseudos_elementary
        inputs = {
            "code": self.inputs.pw_code,
            "pseudos": pseudos,
            "structure": structure,
            "element": orm.Str(self.ctx.element),  # _base wf hold attribute `element`
            "configuration": orm.Str(configuration),
            "pw_base_parameters": orm.Dict(dict=self.ctx.pw_parameters),
            "ecutwfc": orm.Float(self.ctx.ecutwfc),
            "ecutrho": orm.Float(self.ctx.ecutrho),
            "kpoints_distance": orm.Float(self.ctx.kpoints_distance),
            "scale_count": orm.Int(self.ctx.scale_count),
            "scale_increment": orm.Float(self.ctx.scale_increment),
            "options": orm.Dict(dict=self.ctx.options),
            "parallelization": orm.Dict(dict=self.ctx.parallelization),
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

        for configuration in self._OXIDE_CONFIGURATIONS + self._UNARIE_CONFIGURATIONS:
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

    def on_terminated(self):
        """Clean the working directories of all child calculations if `clean_workdir=True` in the inputs."""
        super().on_terminated()

        if self.inputs.clean_workdir.value is False:
            self.report("remote folders will not be cleaned")
            return

        cleaned_calcs = []

        for called_descendant in self.node.called_descendants:
            if isinstance(called_descendant, orm.CalcJobNode):
                try:
                    called_descendant.outputs.remote_folder._clean()  # pylint: disable=protected-access
                    cleaned_calcs.append(called_descendant.pk)
                except (IOError, OSError, KeyError):
                    pass

        if cleaned_calcs:
            self.report(
                f"cleaned remote folders of calculations: {' '.join(map(str, cleaned_calcs))}"
            )
