# -*- coding: utf-8 -*-
"""
Convergence test on phonon frequencies of a given pseudopotential
"""

from aiida import orm
from aiida.engine import calcfunction
from aiida.plugins import DataFactory

from aiida_sssp_workflow.utils import update_dict
from aiida_sssp_workflow.workflows.convergence._base import _BaseConvergenceWorkChain
from aiida_sssp_workflow.workflows.evaluate._phonon_frequencies import (
    PhononFrequenciesWorkChain,
)

UpfData = DataFactory("pseudo.upf")


@calcfunction
def helper_phonon_frequencies_difference(
    element: orm.Str,
    configuration: orm.Str,
    input_parameters: orm.Dict,
    ref_parameters: orm.Dict,
) -> orm.Dict:
    """
    The phonon frequencies are calculated at BZ boundary qpoint (1/2, 1/2, 1/2).
    The difference between the test cutoff and reference cutoff are compared.

    For some elements, we have neglected the first n frequencies in the summation above,
    because the frequencies are negative and/or with strong oscillations as
    function of the cutoff for all the considered pseudos).
    We have neglected the first 4 frequencies for H and I, 12 for N and Cl,
    6 for O and ??SiF4 (which replaces F)??.
    """
    import numpy as np

    input_frequencies = input_parameters["dynamical_matrix_1"]["frequencies"]
    ref_frequencies = ref_parameters["dynamical_matrix_1"]["frequencies"]

    # set strat_idx the idx of frequencies start to count
    element = element.value
    configuration = configuration.value
    if configuration == "GS":
        # leftover setting from SSSP v1
        # Otherwise the phonon frequencies calculated at BZ boundary qpoint (1/2, 1/2, 1/2) are not converged.
        if element == "N" or element == "Cl":
            start_idx = 12
        elif element == "H" or element == "I":
            start_idx = 4
        elif element == "O":
            start_idx = 6
        else:
            start_idx = 0
    else:
        start_idx = 0

    input_frequencies = input_frequencies[start_idx:]
    ref_frequencies = ref_frequencies[start_idx:]

    # calculate the diff
    diffs = np.array(input_frequencies) - np.array(ref_frequencies)
    weights = np.array(ref_frequencies)

    omega_max = np.amax(input_frequencies)

    absolute_diff = np.mean(diffs)
    absolute_max_diff = np.amax(diffs)

    relative_diff = np.sqrt(np.mean((diffs / weights) ** 2))
    relative_max_diff = np.amax(diffs / weights)

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


class ConvergencePhononFrequenciesWorkChain(_BaseConvergenceWorkChain):
    """WorkChain to converge test on cohisive energy of input structure"""

    # pylint: disable=too-many-instance-attributes

    _PROPERTY_NAME = "phonon_frequencies"
    _EVALUATE_WORKCHAIN = PhononFrequenciesWorkChain
    _MEASURE_OUT_PROPERTY = "relative_diff"

    @classmethod
    def define(cls, spec):
        super().define(spec)
        # yapf: disable
        del spec.inputs['code'] # For this input code port set in base work chain need to be unset.
        spec.input("pw_code", valid_type=orm.AbstractCode,
            help="The `pw.x` code use for the `PwCalculation`.")
        spec.input("ph_code",valid_type=orm.AbstractCode,
            help="The `ph.x` code  use for the `PhCalculation`.",
        )
        # yapf: enable

    def init_setup(self):
        super().init_setup()
        self.ctx.extra_ph_parameters = {}
        self.ctx.extra_pw_parameters = {
            "CONTROL": {
                "disk_io": "low",  # no wavefunction file
            },
        }

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
        self._CONV_THR_PER_ATOM = protocol["conv_thr_per_atom"]
        self._KDISTANCE = protocol["kpoints_distance"]

        # PH
        self._QPOINTS_LIST = protocol["qpoints_list"]
        self._PH_EPSILON = protocol["epsilon"]
        self._PH_TR2_PH = protocol["tr2_ph"]

        self.ctx.qpoints_list = self._QPOINTS_LIST

        self.ctx.pw_parameters = super()._get_pw_base_parameters(
            self._DEGAUSS, self._OCCUPATIONS, self._SMEARING, self._CONV_THR_PER_ATOM
        )

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

        self.logger.info(
            f"The pw parameters for convergence is: {self.ctx.pw_parameters}"
        )
        self.logger.info(
            f"The ph parameters for convergence is: {self.ctx.ph_parameters}"
        )

    def _get_inputs(self, ecutwfc, ecutrho):
        """
        get inputs for the evaluation CohesiveWorkChain by provide ecutwfc and ecutrho,
        all other parameters are fixed for the following steps
        """
        pw_parameters = {
            "SYSTEM": {
                "ecutwfc": ecutwfc,
                "ecutrho": ecutrho,
            },
        }
        pw_parameters = update_dict(pw_parameters, self.ctx.pw_parameters)

        qpoints = orm.KpointsData()
        qpoints.set_cell_from_structure(self.ctx.structure)
        qpoints.set_kpoints(self.ctx.qpoints_list)

        # convert parallelization to CMDLINE for PH
        # since ph calculation now doesn't support parallelization
        cmdline_list = []
        for key, value in self.ctx.parallelization.items():
            cmdline_list.append(f"-{str(key)}")
            cmdline_list.append(str(value))

        # Sinec PH calculation always runs more time then the correspoding pw calculation
        # set the walltime to 4 times as set in option.
        ph_options = update_dict(self.ctx.options, {})
        pw_max_walltime = self.ctx.options.get("max_wallclock_seconds", None)
        if pw_max_walltime:
            ph_options["max_wallclock_seconds"] = pw_max_walltime * 4

        inputs = {
            "scf": {
                "metadata": {"call_link_label": "SCF"},
                "pw": {
                    "structure": self.ctx.structure,
                    "code": self.inputs.pw_code,
                    "pseudos": self.ctx.pseudos,
                    "parameters": orm.Dict(dict=pw_parameters),
                    "metadata": {
                        "options": self.ctx.options,
                    },
                    "parallelization": orm.Dict(dict=self.ctx.parallelization),
                },
                "kpoints_distance": orm.Float(self.ctx.kpoints_distance),
            },
            "phonon": {
                "metadata": {"call_link_label": "PH"},
                "ph": {
                    "code": self.inputs.ph_code,
                    "qpoints": qpoints,
                    "parameters": orm.Dict(dict=self.ctx.ph_parameters),
                    "metadata": {
                        "options": ph_options,
                    },
                    "settings": orm.Dict(dict={"CMDLINE": cmdline_list}),
                },
            },
            "clean_workdir": self.inputs.clean_workdir,
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
            orm.Str(self.ctx.element),
            orm.Str(self.ctx.configuration),
            sample_output,
            reference_output,
        ).get_dict()
