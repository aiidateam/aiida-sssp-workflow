# -*- coding: utf-8 -*-
"""
Convergence test on bands of a given pseudopotential
"""

from aiida import orm
from aiida.engine import calcfunction
from aiida.plugins import DataFactory

from aiida_sssp_workflow.calculations.calculate_bands_distance import get_bands_distance
from aiida_sssp_workflow.utils import NONMETAL_ELEMENTS
from aiida_sssp_workflow.workflows.convergence._base import BaseConvergenceWorkChain
from aiida_sssp_workflow.workflows.evaluate._bands import BandsWorkChain

UpfData = DataFactory("pseudo.upf")


@calcfunction
def helper_bands_distence_difference(
    bands_structure_a: orm.BandsData,
    bands_parameters_a: orm.Dict,
    bands_structure_b: orm.BandsData,
    bands_parameters_b: orm.Dict,
    smearing: orm.Float,
    fermi_shift: orm.Float,
    do_smearing: orm.Bool,
):
    """doc"""
    res = get_bands_distance(
        bands_structure_a,
        bands_structure_b,
        bands_parameters_a,
        bands_parameters_b,
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


class ConvergenceBandsWorkChain(BaseConvergenceWorkChain):
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
        self.ctx.parameters = super()._get_pw_base_parameters(
            self._DEGAUSS, self._OCCUPATIONS, self._SMEARING, self._CONV_THR
        )

        self.ctx.fermi_shift = protocol["fermi_shift"]
        self.ctx.init_nbands_factor = protocol["init_nbands_factor"]
        self.ctx.is_metal = self.ctx.element not in NONMETAL_ELEMENTS

        self.logger.info(f"The atom parameters for convergence is: {self.ctx.parameters}")

    def _get_inputs(self, ecutwfc, ecutrho):
        """
        get inputs for the evaluation CohesiveWorkChain by provide ecutwfc and ecutrho,
        all other parameters are fixed for the following steps
        """
        inputs = {
            "code": self.inputs.pw_code,
            "pseudos": self.ctx.pseudos,
            "structure": self.ctx.structure,
            "pw_base_parameters": orm.Dict(dict=self.ctx.parameters),
            "ecutwfc": orm.Int(ecutwfc),
            "ecutrho": orm.Int(ecutrho),
            "kpoints_distance_scf": orm.Float(self.ctx.kpoints_distance_scf),
            "kpoints_distance_bands": orm.Float(self.ctx.kpoints_distance_bands),
            "init_nbands_factor": orm.Float(self.ctx.init_nbands_factor),
            "fermi_shift": orm.Float(self.ctx.fermi_shift),
            "should_run_bands_structure": orm.Bool(
                False
            ),  # for convergence no band structure evaluate
            "options": orm.Dict(dict=self.ctx.options),
            "parallelization": orm.Dict(dict=self.ctx.parallelization),
        }

        return inputs

    def helper_compare_result_extract_fun(self, sample_node, reference_node, **kwargs):
        """implement"""
        sample_bands_output = sample_node.outputs["bands"].band_parameters
        reference_bands_output = reference_node.outputs["bands"].band_parameters

        sample_bands_structure = sample_node.outputs["bands"].band_structure
        reference_bands_structure = reference_node.outputs["bands"].band_structure

        # Always process smearing to find fermi level even for non-metal elements.
        res = helper_bands_distence_difference(
            sample_bands_structure,
            sample_bands_output,
            reference_bands_structure,
            reference_bands_output,
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
