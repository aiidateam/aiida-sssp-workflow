# -*- coding: utf-8 -*-
"""
Convergence test on bands of a given pseudopotential
"""

from aiida import orm
from aiida.engine import calcfunction
from aiida.plugins import DataFactory

from aiida_sssp_workflow.calculations.calculate_bands_distance import get_bands_distance
from aiida_sssp_workflow.utils import NONMETAL_ELEMENTS
from aiida_sssp_workflow.workflows.convergence._base import BaseLegacyWorkChain
from aiida_sssp_workflow.workflows.evaluate._bands import BandsWorkChain

UpfData = DataFactory('pseudo.upf')

@calcfunction
def helper_bands_distence_difference(
    bands_structure_a: orm.BandsData,
    bands_parameters_a: orm.Dict,
    bands_structure_b: orm.BandsData,
    bands_parameters_b: orm.Dict,
    smearing: orm.Float,
    is_metal: orm.Bool,
):
    """doc"""
    res = get_bands_distance(
        bands_structure_a,
        bands_structure_b,
        bands_parameters_a,
        bands_parameters_b,
        smearing.value,
        is_metal.value,
    )
    eta_10 = res.get("eta_10", None)
    shift_10 = res.get("shift_10", None)
    max_diff_10 = res.get("max_diff_10", None)

    return orm.Dict(
        dict={
            "eta_10": eta_10 * 1000,
            "shift_10": shift_10 * 1000,
            "max_diff_10": max_diff_10 * 1000,
            "bands_unit": "meV",    # unit mev with value * 1000
        }
    )


class ConvergenceBandsWorkChain(BaseLegacyWorkChain):
    """WorkChain to converge test on cohisive energy of input structure"""
    # pylint: disable=too-many-instance-attributes

    _RY_TO_EV = 13.6056980659

    _PROPERTY_NAME = 'bands'
    _EVALUATE_WORKCHAIN = BandsWorkChain
    _MEASURE_OUT_PROPERTY = 'eta_10'

    def init_setup(self):
        super().init_setup()
        self.ctx.extra_pw_parameters = {}

    def extra_setup_for_magnetic_element(self):
        """Extra setup for magnetic element"""
        super().extra_setup_for_magnetic_element()

    def setup_code_parameters_from_protocol(self):
        """Input validation"""
        # pylint: disable=invalid-name, attribute-defined-outside-init

        # Read from protocol if parameters not set from inputs
        super().setup_code_parameters_from_protocol()

        # parse protocol
        protocol = self.ctx.protocol
        self.ctx.degauss = self._DEGAUSS = protocol['degauss']
        self._OCCUPATIONS = protocol['occupations']
        self._SMEARING = protocol['smearing']
        self._CONV_THR = protocol['electron_conv_thr']
        self.ctx.kpoints_distance = self._KDISTANCE = protocol['kpoints_distance']

        # Set context parameters
        self.ctx.parameters = super()._get_pw_base_parameters(self._DEGAUSS,
                                                            self._OCCUPATIONS,
                                                            self._SMEARING,
                                                            self._CONV_THR)

        self.ctx.bands_shift = protocol['bands_shift']
        self.ctx.init_nbands_factor = protocol['init_nbands_factor']
        self.ctx.is_metal = self.ctx.element not in NONMETAL_ELEMENTS

        self.report(
            f'The atom parameters for convergence is: {self.ctx.parameters}'
        )

    def _get_inputs(self, ecutwfc, ecutrho):
        """
        get inputs for the evaluation CohesiveWorkChain by provide ecutwfc and ecutrho,
        all other parameters are fixed for the following steps
        """
        inputs = {
            'code': self.inputs.pw_code,
            'pseudos': self.ctx.pseudos,
            'structure': self.ctx.structure,
            'pw_base_parameters': orm.Dict(dict=self.ctx.parameters),
            'ecutwfc': orm.Float(ecutwfc),
            'ecutrho': orm.Float(ecutrho),
            'kpoints_distance': orm.Float(self.ctx.kpoints_distance),
            'init_nbands_factor': orm.Float(self.ctx.init_nbands_factor),
            'bands_shift': orm.Float(self.ctx.bands_shift),
            'should_run_bands_structure': orm.Bool(False), # for convergence no band structure evaluate
            'options': orm.Dict(dict=self.ctx.options),
            'parallelization': orm.Dict(dict=self.ctx.parallelization),
            'clean_workdir': orm.Bool(False),   # will leave the workdir clean to outer most wf
        }

        return inputs

    def helper_compare_result_extract_fun(self, sample_node, reference_node,
                                          **kwargs):
        """implement"""
        sample_bands_output = sample_node.outputs['bands'].band_parameters
        reference_bands_output = reference_node.outputs['bands'].band_parameters

        sample_bands_structure = sample_node.outputs['bands'].band_structure
        reference_bands_structure = reference_node.outputs['bands'].band_structure

        res = helper_bands_distence_difference(sample_bands_structure,
                                               sample_bands_output,
                                               reference_bands_structure,
                                               reference_bands_output,
                                               smearing=orm.Float(self.ctx.degauss * self._RY_TO_EV),
                                               is_metal=orm.Bool(self.ctx.is_metal),
                                               ).get_dict()

        return res

    def get_result_metadata(self):
        return {
            'bands_unit': 'meV',
        }
