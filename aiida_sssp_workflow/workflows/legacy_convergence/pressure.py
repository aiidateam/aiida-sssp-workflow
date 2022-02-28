# -*- coding: utf-8 -*-
"""
Convergence test on pressure of a given pseudopotential
"""
import importlib_resources

from aiida.engine import calcfunction
from aiida import orm
from aiida.engine import append_
from aiida.plugins import DataFactory

from aiida_sssp_workflow.utils import update_dict, \
    get_standard_cif_filename_from_element, convergence_analysis
from aiida_sssp_workflow.workflows.evaluate._pressure import PressureWorkChain
from aiida_sssp_workflow.workflows._eos import _EquationOfStateWorkChain
from aiida_sssp_workflow.workflows.legacy_convergence._base import BaseLegacyWorkChain

UpfData = DataFactory('pseudo.upf')


def helper_get_volume_from_pressure_birch_murnaghan(P, V0, B0, B1):
    """
    Knowing the pressure P and the Birch-Murnaghan equation of state
    parameters, gets the volume the closest to V0 (relatively) that is
    such that P_BirchMurnaghan(V)=P

    retrun unit is (%)
    """
    import numpy as np

    # coefficients of the polynomial in x=(V0/V)^(1/3) (aside from the
    # constant multiplicative factor 3B0/2)
    polynomial = [
        3. / 4. * (B1 - 4.), 0, 1. - 3. / 2. * (B1 - 4.), 0,
        3. / 4. * (B1 - 4.) - 1., 0, 0, 0, 0, -2 * P / (3. * B0)
    ]
    V = min([
        V0 / (x.real**3)
        for x in np.roots(polynomial) if abs(x.imag) < 1e-8 * abs(x.real)
    ],
            key=lambda V: abs(V - V0) / float(V0))

    return abs(V - V0) / V0 * 100


@calcfunction
def helper_pressure_difference(input_parameters: orm.Dict,
                               ref_parameters: orm.Dict, V0: orm.Float,
                               B0: orm.Float, B1: orm.Float) -> orm.Dict:
    """
    doc
    """
    res_pressure = input_parameters['hydrostatic_stress']
    ref_pressure = ref_parameters['hydrostatic_stress']
    absolute_diff = abs(res_pressure - ref_pressure)
    relative_diff = helper_get_volume_from_pressure_birch_murnaghan(
        absolute_diff, V0.value, B0.value, B1.value)

    return orm.Dict(
        dict={
            'pressure': res_pressure,
            'relative_diff': relative_diff,
            'absolute_diff': absolute_diff,
            'absolute_unit': 'GPascal',
            'relative_unit': '%'
        })


class ConvergencePressureWorkChain(BaseLegacyWorkChain):
    """WorkChain to converge test on pressure of input structure"""
    # pylint: disable=too-many-instance-attributes

    # parameters control inner EOS reference workflow
    _EOS_SCALE_COUNT = 7
    _EOS_SCALE_INCREMENT = 0.02
    
    _EVALUATE_WORKCHAIN = PressureWorkChain
    _MEASURE_OUT_PROPERTY = 'relative_diff'

    @classmethod
    def define(cls, spec):
        super().define(spec)
        # yapf: disable
        spec.input('code', valid_type=orm.Code,
                    help='The `pw.x` code use for the `PwCalculation`.')
        # yapy: enable

    def init_setup(self):
        super().init_setup()
        self.ctx.pw_parameters = {}

    def extra_setup_for_magnetic_element(self):
        """Extra setup for magnetic element"""
        super().extra_setup_for_magnetic_element()

    def extra_setup_for_rare_earth_element(self):
        super().extra_setup_for_rare_earth_element()


    def setup_code_parameters_from_protocol(self):
        """Input validation"""
        # pylint: disable=invalid-name, attribute-defined-outside-init

        # Read from protocol if parameters not set from inputs
        super().setup_code_parameters_from_protocol()
        
        protocol = self.ctx.protocol_calculation['convergence']['pressure']
        self._DEGAUSS = protocol['degauss']
        self._OCCUPATIONS = protocol['occupations']
        self._SMEARING = protocol['smearing']
        self._CONV_THR = protocol['electron_conv_thr']
        self._KDISTANCE = protocol['kpoints_distance']

        self._MAX_EVALUATE = protocol['max_evaluate']
        self._REFERENCE_ECUTWFC = protocol['reference_ecutwfc']
        self._NUM_OF_RHO_TEST = protocol['num_of_rho_test']

        self.ctx.max_evaluate = self._MAX_EVALUATE
        self.ctx.reference_ecutwfc = self._REFERENCE_ECUTWFC

        self.ctx.pw_parameters = {
            'SYSTEM': {
                'degauss': self._DEGAUSS,
                'occupations': self._OCCUPATIONS,
                'smearing': self._SMEARING,
            },
            'ELECTRONS': {
                'conv_thr': self._CONV_THR,
            },
            'CONTROL': {
                'calculation': 'scf',
                'wf_collect': True,
                'tstress': True,
            },
        }

        self.ctx.pw_parameters = update_dict(self.ctx.pw_parameters,
                                        self.ctx.extra_pw_parameters)

        self.ctx.kpoints_distance = self._KDISTANCE

        self.report(
            f'The pw parameters for convergence is: {self.ctx.pw_parameters}'
        )
        
    def setup_criteria_parameters_from_protocol(self):
        """Input validation"""
        # pylint: disable=invalid-name, attribute-defined-outside-init

        # Read from protocol if parameters not set from inputs
        super().setup_criteria_parameters_from_protocol()

        self.ctx.criteria = self.ctx.protocol_criteria['convergence']['pressure']

    def _get_inputs(self, ecutwfc, ecutrho):
        """
        get inputs for the evaluation CohesiveWorkChain by provide ecutwfc and ecutrho,
        all other parameters are fixed for the following steps
        """
        # yapf: disable
        inputs = {
            'code': self.inputs.code,
            'pseudos': self.ctx.pseudos,
            'structure': self.ctx.structure,
            'pw_base_parameters': orm.Dict(dict=self.ctx.pw_parameters),
            'ecutwfc': orm.Float(ecutwfc),
            'ecutrho': orm.Float(ecutrho),
            'kpoints_distance': orm.Float(self.ctx.kpoints_distance),
            'options': orm.Dict(dict=self.ctx.options),
            'parallelization': orm.Dict(dict=self.ctx.parallelization),
            'clean_workdir': orm.Bool(False),   # will leave the workdir clean to outer most wf
        }
        # yapf: enable

        return inputs

    def run_reference(self):
        """
        run on reference calculation
        """
        super().run_reference()

        # For pressure convergence workflow, the birch murnagen fitting result is used to
        # calculating the pressure. There is an extra workflow (run at ecutwfc of reference point)
        # for it which need to be run before the following step.

        # This workflow is shared with delta factor workchain for birch murnagan fitting.
        ecutwfc = self.ctx.reference_ecutwfc
        ecutrho = ecutwfc * self.ctx.init_dual
        parameters = {
            'SYSTEM': {
                'ecutwfc': ecutwfc,
                'ecutrho': ecutrho,
            },
        }
        self.ctx.pw_parameters = update_dict(self.ctx.pw_parameters,
                                             parameters)

        self.report(f'{self.ctx.pw_parameters}')
        # yapf: disable
        inputs = {
            'structure': self.ctx.structure,
            'kpoints_distance': orm.Float(self._KDISTANCE),
            'scale_count': orm.Int(self._EOS_SCALE_COUNT),
            'scale_increment': orm.Float(self._EOS_SCALE_INCREMENT),
            'metadata': {
                'call_link_label': 'EOS'
            },
            'scf': {
                'pw': {
                    'code': self.inputs.code,
                    'pseudos': self.ctx.pseudos,
                    'parameters': orm.Dict(dict=self.ctx.pw_parameters),
                    'metadata': {
                        'options': self.ctx.options
                    },
                    'parallelization': orm.Dict(dict=self.ctx.parallelization),
                },
            }
        }
        # yapf: enable

        running = self.submit(_EquationOfStateWorkChain, **inputs)
        self.report(f'launching _EquationOfStateWorkChain<{running.pk}>')

        self.to_context(extra_reference=running)
        
    def inspect_reference(self):
        super().inspect_reference()
            
        workchain = self.ctx.extra_reference
        if not workchain.is_finished_ok:
            self.report(
                f'{workchain.process_label} pk={workchain.pk} for reference run is failed with exit_code={workchain.exit_status}.'
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED.format(
                label='extra_reference')
            
        extra_reference = self.ctx.extra_reference
        extra_reference_parameters = extra_reference.outputs.output_birch_murnaghan_fit

        V0 = extra_reference_parameters['volume0']
        B0 = extra_reference_parameters['bulk_modulus0']
        B1 = extra_reference_parameters['bulk_deriv0']

        self.ctx.extra_parameters={
            'V0': orm.Float(V0),
            'B0': orm.Float(B0),
            'B1': orm.Float(B1)
        }
            
    def get_result_metadata(self):
        return {
            'absolute_unit': 'GPascal',
            'relative_unit': '%',
        }

    def helper_compare_result_extract_fun(self, sample_node, reference_node,
                                          **kwargs):
        """implement"""
        sample_output = sample_node.outputs.output_parameters
        reference_output = reference_node.outputs.output_parameters

        extra_parameters = kwargs['extra_parameters']
        res = helper_pressure_difference(sample_output, reference_output,
                                         **extra_parameters).get_dict()

        return res
