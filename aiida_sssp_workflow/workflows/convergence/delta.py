# -*- coding: utf-8 -*-
"""
Convergence test on cohesive energy of a given pseudopotential
"""
from aiida import orm
from aiida.engine import calcfunction
from aiida.plugins import DataFactory

from aiida_sssp_workflow.utils import LANTHANIDE_ELEMENTS, update_dict
from aiida_sssp_workflow.workflows.convergence._base import _BaseConvergenceWorkChain
from aiida_sssp_workflow.workflows.evaluate._metric import MetricWorkChain

UpfData = DataFactory("pseudo.upf")


@calcfunction
def helper_delta_difference(
    input_parameters: orm.Dict, ref_parameters: orm.Dict
) -> orm.Dict:
    """calculate the delta difference from parameters"""
    res_delta = input_parameters.get_dict().get("delta/natoms", None)
    # There is a change the delta cannot be calculated, e.g. the cutoff is too low.
    if res_delta is None:
        return orm.Dict(dict={})
    ref_delta = ref_parameters["delta/natoms"]
    absolute_diff = abs(res_delta - ref_delta)
    relative_diff = abs((res_delta - ref_delta) / ref_delta) * 100

    res = {
        "delta/natoms": res_delta,
        "absolute_diff": absolute_diff,
        "relative_diff": relative_diff,
        "delta_unit": "meV",
        "relative_unit": "%",
    }

    return orm.Dict(dict=res)


class ConvergenceDeltaWorkChain(_BaseConvergenceWorkChain):
    """WorkChain to converge test on delta factor of input structure"""

    # pylint: disable=too-many-instance-attributes

    _PROPERTY_NAME = "delta"
    _EVALUATE_WORKCHAIN = MetricWorkChain
    _MEASURE_OUT_PROPERTY = "absolute_diff"

    def init_setup(self):
        super().init_setup()
        self.ctx.extra_pw_parameters = {
            "CONTROL": {
                # no wavefunction file, this means no calculation will be from cache
                "disk_io": "nowf",
            },
        }

    def extra_setup_for_lanthanide_element(self):
        super().extra_setup_for_lanthanide_element()

    def setup_code_parameters_from_protocol(self):
        """Input validation"""
        # pylint: disable=invalid-name, attribute-defined-outside-init

        # Read from protocol if parameters not set from inputs
        super().setup_code_parameters_from_protocol()

        # parse protocol
        protocol = self.ctx.protocol
        self._DEGAUSS = protocol["degauss"]
        self._OCCUPATIONS = protocol["occupations"]
        self._SMEARING = protocol["smearing"]
        self._CONV_THR_PER_ATOM = protocol["conv_thr_per_atom"]
        self._KDISTANCE = protocol["kpoints_distance"]

        self.ctx.scale_count = self._SCALE_COUNT = protocol["scale_count"]
        self.ctx.scale_increment = self._SCALE_INCREMENT = protocol["scale_increment"]

        # Set context parameters
        self.ctx.kpoints_distance = self._KDISTANCE
        self.ctx.pw_parameters = super()._get_pw_base_parameters(
            self._DEGAUSS,
            self._OCCUPATIONS,
            self._SMEARING,
            self._CONV_THR_PER_ATOM,
        )

        # set extra pw parameters for eos only
        self._MIXING_BETA = protocol["mixing_beta"]
        self.ctx.pw_parameters["ELECTRONS"]["mixing_beta"] = self._MIXING_BETA

        self.logger.info(
            f"The pw parameters for convergence is: {self.ctx.pw_parameters}"
        )

    def _get_inputs(self, ecutwfc, ecutrho):
        """
        get inputs for the evaluation MetricWorkChain by provide ecutwfc and ecutrho,
        all other parameters are fixed for the following steps
        """
        parameters = {
            "SYSTEM": {
                "ecutwfc": ecutwfc,
                "ecutrho": ecutrho,
            },
        }
        parameters = update_dict(parameters, self.ctx.pw_parameters)
        parameters["CONTROL"].pop(
            "tstress", None
        )  # this will rule this work chain out from caching

        # sparse kpoints and tetrahedra occupation in EOS reference calculation
        if self.ctx.element in LANTHANIDE_ELEMENTS:
            self.ctx.kpoints_distance = self._KDISTANCE + 0.05
            parameters["SYSTEM"].pop("smearing", None)
            parameters["SYSTEM"].pop("degauss", None)
            parameters["SYSTEM"]["occupations"] = "tetrahedra"

        inputs = {
            "eos": {
                "metadata": {"call_link_label": "delta_EOS"},
                "structure": self.ctx.structure,
                "kpoints_distance": orm.Float(self.ctx.kpoints_distance),
                "scale_count": orm.Int(self.ctx.scale_count),
                "scale_increment": orm.Float(self.ctx.scale_increment),
                "pw": {
                    "code": self.inputs.code,
                    "pseudos": self.ctx.pseudos,
                    "parameters": orm.Dict(dict=parameters),
                    "metadata": {
                        "options": self.ctx.options,
                    },
                    "parallelization": orm.Dict(dict=self.ctx.parallelization),
                },
                "clean_workdir": self.inputs.clean_workdir,
            },
            "element": orm.Str(self.ctx.element),
            "configuration": orm.Str(self.ctx.configuration),
            # "clean_workdir": self.inputs.clean_workdir,  # should already cleaned by eos, here for safe
        }

        return inputs

    def helper_compare_result_extract_fun(self, sample_node, reference_node, **kwargs):
        """extract"""
        sample_output = sample_node.outputs.output_parameters
        reference_output = reference_node.outputs.output_parameters

        res = helper_delta_difference(sample_output, reference_output).get_dict()

        return res

    def get_result_metadata(self):
        return {
            "delta_unit": "meV",
            "relative_unit": "%",
        }
