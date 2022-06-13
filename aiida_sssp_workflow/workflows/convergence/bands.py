# -*- coding: utf-8 -*-
"""
Convergence test on bands of a given pseudopotential
"""

from aiida import orm
from aiida.engine import calcfunction
from aiida.plugins import DataFactory

from aiida_sssp_workflow.calculations.calculate_bands_distance import get_bands_distance
from aiida_sssp_workflow.utils import NONMETAL_ELEMENTS, update_dict
from aiida_sssp_workflow.workflows.convergence._base import _BaseConvergenceWorkChain
from aiida_sssp_workflow.workflows.evaluate._bands import BandsWorkChain

UpfData = DataFactory("pseudo.upf")


@calcfunction
def helper_bands_distence_difference(
    band_structure_a: orm.BandsData,
    band_parameters_a: orm.Dict,
    band_structure_b: orm.BandsData,
    band_parameters_b: orm.Dict,
    smearing: orm.Float,
    fermi_shift: orm.Float,
    do_smearing: orm.Bool,
):
    """doc"""
    # `get_bands_distance` require the less electrons results pass
    # to inputs of label 'a', do swap a and b if not.
    num_electrons_a = band_parameters_a["number_of_electrons"]
    num_electrons_b = band_parameters_b["number_of_electrons"]

    if num_electrons_a > num_electrons_b:
        band_parameters_a, band_parameters_b = band_parameters_b, band_parameters_a
        band_structure_a, band_structure_b = band_structure_b, band_structure_a

    res = get_bands_distance(
        band_structure_a,
        band_structure_b,
        band_parameters_a,
        band_parameters_b,
        smearing.value,
        fermi_shift.value,
        do_smearing.value,
    )
    eta = res.get("eta_c", None)
    shift = res.get("shift_c", None)
    max_diff = res.get("max_diff_c", None)

    return orm.Dict(
        dict={
            "eta_c": eta * 1000,
            "shift_c": shift * 1000,
            "max_diff_c": max_diff * 1000,
            "bands_unit": "meV",  # unit mev with value * 1000
        }
    )


class ConvergenceBandsWorkChain(_BaseConvergenceWorkChain):
    """WorkChain to converge test on cohisive energy of input structure"""

    # pylint: disable=too-many-instance-attributes

    _RY_TO_EV = 13.6056980659

    _PROPERTY_NAME = "bands"
    _EVALUATE_WORKCHAIN = BandsWorkChain
    _MEASURE_OUT_PROPERTY = "eta_c"

    def init_setup(self):
        super().init_setup()
        self.ctx.extra_pw_parameters = {}

    def setup_code_parameters_from_protocol(self):
        """Input validation"""
        # pylint: disable=invalid-name, attribute-defined-outside-init

        # Read from protocol if parameters not set from inputs
        super().setup_code_parameters_from_protocol()

        # parse protocol
        protocol = self.ctx.protocol
        self.ctx.degauss = self._DEGAUSS = protocol["degauss"]
        self._OCCUPATIONS = protocol["occupations"]
        self._SMEARING = protocol["smearing"]
        self._CONV_THR = protocol["electron_conv_thr"]
        self.ctx.kpoints_distance_scf = protocol["kpoints_distance_scf"]
        self.ctx.kpoints_distance_bands = protocol["kpoints_distance_bands"]

        # Set context parameters
        self.ctx.pw_parameters = super()._get_pw_base_parameters(
            self._DEGAUSS, self._OCCUPATIONS, self._SMEARING, self._CONV_THR
        )

        self.ctx.fermi_shift = protocol["fermi_shift"]
        self.ctx.init_nbands_factor = protocol["init_nbands_factor"]
        self.ctx.is_metal = self.ctx.element not in NONMETAL_ELEMENTS

        self.logger.info(f"The parameters for convergence is: {self.ctx.pw_parameters}")

    def _get_inputs(self, ecutwfc, ecutrho):
        """
        get inputs for the evaluation CohesiveWorkChain by provide ecutwfc and ecutrho,
        all other parameters are fixed for the following steps
        """
        parameters = {
            "SYSTEM": {
                "ecutwfc": ecutwfc,
                "ecutrho": ecutrho,
            },
        }

        parameters = update_dict(parameters, self.ctx.pw_parameters)

        parameters_bands = update_dict(parameters, {})
        parameters_bands["SYSTEM"].pop("nbnd", None)
        parameters_bands["CONTROL"].pop("tstress", None)

        inputs = {
            "structure": self.ctx.structure,
            "scf": {
                "pw": {
                    "code": self.inputs.code,
                    "pseudos": self.ctx.pseudos,
                    "parameters": orm.Dict(dict=parameters),
                    "metadata": {
                        "options": self.ctx.options,
                    },
                    "parallelization": orm.Dict(dict=self.ctx.parallelization),
                },
                "kpoints_distance": orm.Float(self.ctx.kpoints_distance_scf),
            },
            "bands": {
                "pw": {
                    "code": self.inputs.code,
                    "pseudos": self.ctx.pseudos,
                    "parameters": orm.Dict(dict=parameters_bands),
                    "metadata": {
                        "options": self.ctx.options,
                    },
                    "parallelization": orm.Dict(dict=self.ctx.parallelization),
                },
            },
            "kpoints_distance_bands": orm.Float(self.ctx.kpoints_distance_bands),
            "init_nbands_factor": orm.Float(self.ctx.init_nbands_factor),
            "fermi_shift": orm.Float(self.ctx.fermi_shift),
            "run_bands_structure": orm.Bool(
                False
            ),  # for convergence with no band structure evaluate
            "clean_workchain": self.inputs.clean_workchain,
        }

        return inputs

    def helper_compare_result_extract_fun(self, sample_node, reference_node, **kwargs):
        """implement"""
        sample_band_parameters = sample_node.outputs["bands"].band_parameters
        reference_band_parameters = reference_node.outputs["bands"].band_parameters

        sample_band_structure = sample_node.outputs["bands"].band_structure
        reference_band_structure = reference_node.outputs["bands"].band_structure

        # Always process smearing to find fermi level even for non-metal elements.
        res = helper_bands_distence_difference(
            sample_band_structure,
            sample_band_parameters,
            reference_band_structure,
            reference_band_parameters,
            smearing=orm.Float(self.ctx.degauss * self._RY_TO_EV),
            fermi_shift=orm.Float(self.ctx.fermi_shift),
            do_smearing=orm.Bool(True),
        ).get_dict()

        return res

    def get_result_metadata(self):
        return {
            "fermi_shift": self.ctx.fermi_shift,
            "bands_unit": "meV",
        }
