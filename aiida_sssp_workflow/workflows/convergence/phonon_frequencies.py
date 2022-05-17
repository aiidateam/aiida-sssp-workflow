# -*- coding: utf-8 -*-
"""
Convergence test on phonon frequencies of a given pseudopotential
"""

from aiida import orm
from aiida.engine import calcfunction
from aiida.plugins import DataFactory

from aiida_sssp_workflow.utils import update_dict
from aiida_sssp_workflow.workflows.convergence._base import BaseLegacyWorkChain
from aiida_sssp_workflow.workflows.evaluate._phonon_frequencies import (
    PhononFrequenciesWorkChain,
)

UpfData = DataFactory("pseudo.upf")


@calcfunction
def helper_phonon_frequencies_difference(
    input_parameters: orm.Dict, ref_parameters: orm.Dict
) -> orm.Dict:
    """
    doc
    """
    import numpy as np

    input_frequencies = input_parameters["dynamical_matrix_0"]["frequencies"]
    ref_frequencies = ref_parameters["dynamical_matrix_0"]["frequencies"]
    diffs = np.array(input_frequencies) - np.array(ref_frequencies)
    weights = np.array(ref_frequencies)

    omega_max = np.amax(input_frequencies)

    absolute_diff = np.mean(diffs)
    absolute_max_diff = np.amax(diffs)

    relative_diff = np.sqrt(np.mean((diffs / weights) ** 2)) * 100
    relative_max_diff = np.amax(diffs / weights) * 100

    return orm.Dict(
        dict={
            "omega_max": omega_max,
            "relative_diff": relative_diff,
            "relative_max_diff": relative_max_diff,
            "absolute_diff": absolute_diff,
            "absolute_max_diff": absolute_max_diff,
            "absolute_unit": "cm-1",
            "relative_unit": "%",
        }
    )


class ConvergencePhononFrequenciesWorkChain(BaseLegacyWorkChain):
    """WorkChain to converge test on cohisive energy of input structure"""

    # pylint: disable=too-many-instance-attributes

    _PROPERTY_NAME = "phonon_frequencies"
    _EVALUATE_WORKCHAIN = PhononFrequenciesWorkChain
    _MEASURE_OUT_PROPERTY = "relative_diff"

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input(
            "ph_code",
            valid_type=orm.Code,
            help="The `ph.x` code use for the `PhCalculation`.",
        )

    def init_setup(self):
        super().init_setup()
        self.ctx.extra_ph_parameters = {}
        self.ctx.extra_pw_parameters = {}

    def extra_setup_for_rare_earth_element(self):
        super().extra_setup_for_rare_earth_element()

        extra_ph_parameters = {
            "INPUTPH": {
                "diagonalization": "cg",
            }
        }
        self.ctx.extra_ph_parameters = update_dict(
            self.ctx.extra_ph_parameters, extra_ph_parameters
        )

    def setup_code_parameters_from_protocol(self):
        """Input validation"""
        # pylint: disable=invalid-name, attribute-defined-outside-init

        # Read from protocol if parameters not set from inputs
        super().setup_code_parameters_from_protocol()

        protocol = self.ctx.protocol
        # PW
        self._DEGAUSS = protocol["degauss"]
        self._OCCUPATIONS = protocol["occupations"]
        self._SMEARING = protocol["smearing"]
        self._CONV_THR = protocol["electron_conv_thr"]
        self._KDISTANCE = protocol["kpoints_distance"]

        # PH
        self._QPOINTS_LIST = protocol["qpoints_list"]
        self._PH_EPSILON = protocol["epsilon"]
        self._PH_TR2_PH = protocol["tr2_ph"]

        self.ctx.qpoints_list = self._QPOINTS_LIST

        self.ctx.pw_parameters = super()._get_pw_base_parameters(
            self._DEGAUSS, self._OCCUPATIONS, self._SMEARING, self._CONV_THR
        )

        # self.ctx.pw_parameters = update_dict(self.ctx.pw_parameters,
        #                                 self.ctx.extra_pw_parameters)

        self.ctx.ph_parameters = {
            "INPUTPH": {
                "tr2_ph": self._PH_TR2_PH,
                "epsil": self._PH_EPSILON,
            }
        }

        self.ctx.ph_parameters = update_dict(
            self.ctx.ph_parameters, self.ctx.extra_ph_parameters
        )
        self.ctx.kpoints_distance = self._KDISTANCE

        self.logger.info(f"The pw parameters for convergence is: {self.ctx.pw_parameters}")
        self.logger.info(f"The ph parameters for convergence is: {self.ctx.ph_parameters}")

    def _get_inputs(self, ecutwfc, ecutrho):
        """
        get inputs for the evaluation CohesiveWorkChain by provide ecutwfc and ecutrho,
        all other parameters are fixed for the following steps
        """
        inputs = {
            "pw_code": self.inputs.pw_code,
            "ph_code": self.inputs.ph_code,
            "pseudos": self.ctx.pseudos,
            "structure": self.ctx.structure,
            "pw_base_parameters": orm.Dict(dict=self.ctx.pw_parameters),
            "ph_base_parameters": orm.Dict(dict=self.ctx.ph_parameters),
            "ecutwfc": orm.Int(ecutwfc),
            "ecutrho": orm.Int(ecutrho),
            "qpoints": orm.List(list=self.ctx.qpoints_list),
            "kpoints_distance": orm.Float(self.ctx.kpoints_distance),
            "options": orm.Dict(dict=self.ctx.options),
            "parallelization": orm.Dict(dict=self.ctx.parallelization),
        }

        return inputs

    def get_result_metadata(self):
        return {
            "absolute_unit": "cm-1",
            "relative_unit": "%",
        }

    def helper_compare_result_extract_fun(self, sample_node, reference_node, **kwargs):
        """extract"""
        sample_output = sample_node.outputs.output_parameters
        reference_output = reference_node.outputs.output_parameters

        return helper_phonon_frequencies_difference(
            sample_output, reference_output
        ).get_dict()
