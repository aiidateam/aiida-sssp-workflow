# -*- coding: utf-8 -*-
"""
Convergence test on bands of a given pseudopotential
"""
from aiida.common import AttributeDict
from aiida.engine import workfunction
from aiida import orm
from aiida.plugins import WorkflowFactory

from aiida_sssp_workflow.utils import update_dict, NONMETAL_ELEMENTS
from aiida_sssp_workflow.calculations.calculate_bands_distance import calculate_bands_distance
from .base import BaseConvergenceWorkChain

BandsWorkChain = WorkflowFactory('sssp_workflow.bands')

PARA_ECUTWFC_LIST = lambda: orm.List(list=[
    20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 110,
    120, 130, 140, 160, 180, 200
])

PARA_ECUTRHO_LIST = lambda: orm.List(list=[
    160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680, 720,
    760, 800, 880, 960, 1040, 1120, 1280, 1440, 1600
])


@workfunction
def helper_cohesive_energy_difference(input_band_structure: orm.BandsData,
                                      ref_band_structure: orm.BandsData,
                                      input_band_parameters: orm.Dict,
                                      ref_band_parameters: orm.Dict,
                                      smearing: orm.Float, is_metal: orm.Bool):
    res = calculate_bands_distance(input_band_structure,
                                   ref_band_structure,
                                   input_band_parameters,
                                   ref_band_parameters,
                                   smearing=smearing,
                                   is_metal=is_metal)

    return orm.Dict(
        dict={
            'eta_v': res.get('eta_v', None).value,
            'eta_10': res.get('eta_10', None).value,
            'max_diff_v': res.get('max_diff_v', None).value,
            'max_diff_10': res.get('max_diff_10', None).value,
            'bands_unit': 'eV'
        }).store()


class ConvergenceBandsWorkChain(BaseConvergenceWorkChain):
    """WorkChain to converge test on cohisive energy of input structure"""
    # pylint: disable=too-many-instance-attributes

    _RY_TO_EV = 13.6056980659

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input('code',
                   valid_type=orm.Code,
                   help='The `pw.x` code use for the `PwCalculation`.')

    def setup_protocol(self):
        # pylint: disable=invalid-name, attribute-defined-outside-init
        protocol_name = self.inputs.protocol.value
        protocol = self._get_protocol()[protocol_name]
        protocol = protocol['convergence']['bands_distance']
        self._DEGUASS = protocol['degauss']
        self._OCCUPATIONS = protocol['occupations']
        self._SMEARING = protocol['smearing']
        self._CONV_THR_EVA = protocol['electron_conv_thr']
        self._SCF_KDISTANCE = protocol['scf_kpoints_distance']
        self._BAND_KDISTANCE = protocol['band_kpoints_distance']
        self._ININ_NBND_FACTOR = protocol['init_nbnd_factor']

        self._TOLERANCE = protocol['tolerance']
        self._CONV_THR_CONV = protocol['convergence_conv_thr']
        self._CONV_WINDOW = protocol['convergence_window']

    def init_step(self):
        element = self.inputs.pseudo.element
        self.ctx.is_metal = element not in NONMETAL_ELEMENTS

    def get_create_process(self):
        return BandsWorkChain

    def get_evaluate_process(self):
        return helper_cohesive_energy_difference

    def get_parsed_results(self):
        return {
            'eta_v': ('The difference eta of valence bands', 'eV'),
            'eta_10':
            ('The difference eta of valence bands and conduct bands up to 10.0eV',
             'eV'),
            'max_diff_v': ('The max difference of valence bands', 'eV'),
            'max_diff_10':
            ('The max difference of valence bands and conduct bands up to 10.0eV',
             'eV'),
        }

    def get_converge_y(self):
        return 'eta_10', 'eV'

    def get_create_process_inputs(self):
        _PW_PARAS = {   # pylint: disable=invalid-name
            'SYSTEM': {
                'degauss': self._DEGUASS,
                'occupations': self._OCCUPATIONS,
                'smearing': self._SMEARING,
            },
            'ELECTRONS': {
                'conv_thr': self._CONV_THR_EVA,
            },
        }

        inputs = AttributeDict({
            'code': self.inputs.code,
            'pseudos': self.ctx.pseudos,
            'structure': self.ctx.structure,
            'parameters': {
                'pw':
                orm.Dict(
                    dict=update_dict(_PW_PARAS, self.ctx.base_pw_parameters)),
                'scf_kpoints_distance':
                orm.Float(self._SCF_KDISTANCE),
                'bands_kpoints_distance':
                orm.Float(self._BAND_KDISTANCE),
                'nbands_factor':
                orm.Float(self._ININ_NBND_FACTOR)
            },
        })

        return inputs

    def get_evaluate_process_inputs(self):
        ref_workchain = self.ctx.ref_workchain

        res = {
            'ref_band_parameters': ref_workchain.outputs.band_parameters,
            'ref_band_structure': ref_workchain.outputs.band_structure,
            'smearing': orm.Float(self._DEGUASS * self._RY_TO_EV),
            'is_metal': orm.Bool(self.ctx.is_metal),
        }

        return res

    def get_output_input_mapping(self):
        res = orm.Dict(
            dict={
                'band_parameters': 'input_band_parameters',
                'band_structure': 'input_band_structure',
            })
        return res
