# -*- coding: utf-8 -*-
"""
Convergence test on pressure of a given pseudopotential
"""
from aiida import orm
from aiida.engine import calcfunction
from aiida.plugins import DataFactory

from aiida_sssp_workflow.utils import LANTHANIDE_ELEMENTS, update_dict
from aiida_sssp_workflow.workflows.convergence._base import _BaseConvergenceWorkChain
from aiida_sssp_workflow.workflows.evaluate._eos import _EquationOfStateWorkChain
from aiida_sssp_workflow.workflows.evaluate._pressure import PressureWorkChain

UpfData = DataFactory("pseudo.upf")


def helper_get_volume_from_pressure_birch_murnaghan(P, V0, B0, B1):
    """
    Knowing the pressure P and the Birch-Murnaghan equation of state
    parameters, gets the volume the closest to V0 (relatively) that is
    such that P_BirchMurnaghan(V)=P

    retrun unit is (%)

    !! The unit of P and B0 must be compatible. We use eV/angs^3 here.
    Therefore convert P from GPa to eV/angs^3
    """
    import numpy as np

    # convert P from GPa to eV/angs^3
    P = P / 160.21766208

    # coefficients of the polynomial in x=(V0/V)^(1/3) (aside from the
    # constant multiplicative factor 3B0/2)
    polynomial = [
        3.0 / 4.0 * (B1 - 4.0),
        0,
        1.0 - 3.0 / 2.0 * (B1 - 4.0),
        0,
        3.0 / 4.0 * (B1 - 4.0) - 1.0,
        0,
        0,
        0,
        0,
        -2 * P / (3.0 * B0),
    ]
    V = min(
        [
            V0 / (x.real**3)
            for x in np.roots(polynomial)
            if abs(x.imag) < 1e-8 * abs(x.real)
        ],
        key=lambda V: abs(V - V0) / float(V0),
    )

    return abs(V - V0) / V0 * 100


@calcfunction
def helper_pressure_difference(
    input_parameters: orm.Dict,
    ref_parameters: orm.Dict,
    V0: orm.Float,
    B0: orm.Float,
    B1: orm.Float,
) -> orm.Dict:
    """
    The unit of output pressure and absolute diff is GPascal
    therefore the B0 unit should also be GPa, otherwise the results are wrong.
    """
    res_pressure = input_parameters["hydrostatic_stress"]
    ref_pressure = ref_parameters["hydrostatic_stress"]
    absolute_diff = abs(res_pressure - ref_pressure)
    relative_diff = helper_get_volume_from_pressure_birch_murnaghan(
        absolute_diff, V0.value, B0.value, B1.value
    )

    return orm.Dict(
        dict={
            "pressure": res_pressure,
            "relative_diff": relative_diff,
            "absolute_diff": absolute_diff,
            "absolute_unit": "GPascal",
            "relative_unit": "%",
        }
    )


class ConvergencePressureWorkChain(_BaseConvergenceWorkChain):
    """WorkChain to converge test on pressure of input structure"""

    # pylint: disable=too-many-instance-attributes

    _PROPERTY_NAME = "pressure"
    _EVALUATE_WORKCHAIN = PressureWorkChain
    _MEASURE_OUT_PROPERTY = "relative_diff"

    def init_setup(self):
        super().init_setup()
        self.ctx.pw_parameters = {}
        self.ctx.extra_pw_parameters = {
            "CONTROL": {
                "disk_io": "low",
            },
        }

    def setup_code_parameters_from_protocol(self):
        """Input validation"""
        # pylint: disable=invalid-name, attribute-defined-outside-init

        # Read from protocol if parameters not set from inputs
        super().setup_code_parameters_from_protocol()

        protocol = self.ctx.protocol
        self._DEGAUSS = protocol["degauss"]
        self._OCCUPATIONS = protocol["occupations"]
        self._SMEARING = protocol["smearing"]
        self._CONV_THR_PER_ATOM = protocol["conv_thr_per_atom"]
        self._KDISTANCE = protocol["kpoints_distance"]

        self._EOS_SCALE_COUNT = protocol["scale_count"]
        self._EOS_SCALE_INCREMENT = protocol["scale_increment"]

        self.ctx.pw_parameters = super()._get_pw_base_parameters(
            self._DEGAUSS, self._OCCUPATIONS, self._SMEARING, self._CONV_THR_PER_ATOM
        )

        # set extra pw parameters for eos only
        self.ctx.mixing_beta = protocol["mixing_beta"]

        self.ctx.kpoints_distance = self._KDISTANCE

        self.logger.info(
            f"The pw parameters for convergence is: {self.ctx.pw_parameters}"
        )

    def _get_inputs(self, ecutwfc, ecutrho):
        """
        get inputs for the evaluation PressureWorkChain by provide ecutwfc and ecutrho,
        all other parameters are fixed for the following steps
        """
        parameters = {
            "SYSTEM": {
                "ecutwfc": ecutwfc,
                "ecutrho": ecutrho,
            },
        }
        parameters = update_dict(parameters, self.ctx.pw_parameters)

        inputs = {
            "metadata": {
                "call_link_label": "prepare_pw_scf"
            },  # used for checking if caching is working
            "pw": {
                "code": self.inputs.code,
                "structure": self.ctx.structure,
                "pseudos": self.ctx.pseudos,
                "parameters": orm.Dict(dict=parameters),
                "metadata": {
                    "options": self.ctx.options,
                },
                "parallelization": orm.Dict(dict=self.ctx.parallelization),
            },
            "kpoints_distance": orm.Float(self.ctx.kpoints_distance),
            "clean_workdir": self.inputs.clean_workdir,
        }

        return inputs

    def run_reference(self):
        """
        run on reference calculation
        """
        super().run_reference()

        # For pressure convergence workflow, the birch murnagen fitting result is used to
        # calculating the pressure. There is an extra workflow (run at ecutwfc of reference point)
        # for it which need to be run before the following step.

        # This workflow is shared with precesion measure workchain for birch murnagan fitting.
        ecutwfc = self.ctx.reference_ecutwfc
        ecutrho = ecutwfc * self.ctx.dual
        parameters = {
            "SYSTEM": {
                "ecutwfc": round(ecutwfc),
                "ecutrho": round(ecutrho),
            },
        }
        parameters = update_dict(parameters, self.ctx.pw_parameters)

        # It is important to set CONTROL here to distinguish from the caching calculation
        # otherwise since we use clean_workdir for this workflow, it may clean the original
        # folder (manually, since the clean_workdir is optimized to avoid this case, but I can not
        # sure it won't happened if the remote folder is cleaned by hand) that still needed to be
        # used for caching calculation.
        parameters["CONTROL"].pop("tstress", None)
        parameters["CONTROL"]["disk_io"] = "nowf"

        # sparse kpoints and tetrahedra occupation in EOS reference calculation
        if self.ctx.element in LANTHANIDE_ELEMENTS:
            self.ctx.kpoints_distance = self._KDISTANCE + 0.05
            parameters["SYSTEM"].pop("smearing", None)
            parameters["SYSTEM"].pop("degauss", None)
            parameters["SYSTEM"]["occupations"] = "tetrahedra"

        parameters["ELECTRONS"]["mixing_beta"] = self.ctx.mixing_beta

        inputs = {
            "metadata": {"call_link_label": "pressure_ref_EOS"},
            "structure": self.ctx.structure,
            "kpoints_distance": orm.Float(self.ctx.kpoints_distance),
            "scale_count": orm.Int(self._EOS_SCALE_COUNT),
            "scale_increment": orm.Float(self._EOS_SCALE_INCREMENT),
            "pw": {
                "code": self.inputs.code,
                "pseudos": self.ctx.pseudos,
                "parameters": orm.Dict(dict=parameters),
                "metadata": {
                    "options": self.ctx.options,
                },
                "parallelization": orm.Dict(dict=self.ctx.parallelization),
            },
            "clean_workdir": self.inputs.clean_workdir,  # exposed from PwBaseWorkChain
        }

        running = self.submit(_EquationOfStateWorkChain, **inputs)
        self.report(f"launching _EquationOfStateWorkChain<{running.pk}>")

        self.to_context(extra_reference=running)

    def inspect_reference(self):
        super().inspect_reference()

        workchain = self.ctx.extra_reference
        if not workchain.is_finished_ok:
            self.logger.warning(
                f"{workchain.process_label} pk={workchain.pk} for extra reference of "
                "pressure convergence run is failed with exit_code={workchain.exit_status}."
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED.format(
                label="extra_reference"
            )

        extra_reference = self.ctx.extra_reference
        extra_reference_parameters = extra_reference.outputs.output_birch_murnaghan_fit

        V0 = extra_reference_parameters["volume0"]
        B0 = extra_reference_parameters["bulk_modulus0"]  # The unit is eV/angstrom^3
        B1 = extra_reference_parameters["bulk_deriv0"]

        self.ctx.extra_parameters = {
            "V0": orm.Float(V0),
            "B0": orm.Float(B0),
            "B1": orm.Float(B1),
        }

    def get_result_metadata(self):
        return {
            "absolute_unit": "GPascal",
            "relative_unit": "%",
        }

    def helper_compare_result_extract_fun(self, sample_node, reference_node, **kwargs):
        """implement"""
        sample_output = sample_node.outputs.output_parameters
        reference_output = reference_node.outputs.output_parameters

        extra_parameters = kwargs["extra_parameters"]
        res = helper_pressure_difference(
            sample_output, reference_output, **extra_parameters
        ).get_dict()

        return res
