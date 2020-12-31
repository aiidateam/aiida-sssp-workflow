# -*- coding: utf-8 -*-
"""
Convergence test on cohesive energy of a given pseudopotential
"""
from aiida.engine import calcfunction
from aiida import orm
from aiida.plugins import WorkflowFactory

from aiida_sssp_workflow.utils import update_dict
from .base import BaseConvergenceWorkChain

PhononFrequenciesWorkChain = WorkflowFactory(
    'sssp_workflow.phonon_frequencies')


@calcfunction
def helper_phonon_frequencies_difference(input_parameters: orm.Dict,
                                         ref_parameters: orm.Dict) -> orm.Dict:
    """
    doc
    """
    import numpy as np

    input_frequencies = input_parameters['dynamical_matrix_0']['frequencies']
    ref_frequencies = ref_parameters['dynamical_matrix_0']['frequencies']
    diffs = np.array(input_frequencies) - np.array(ref_frequencies)
    weights = np.array(ref_frequencies)

    absolute_diff = np.mean(diffs)
    absolute_max_diff = np.amax(diffs)

    relative_diff = np.sqrt(np.mean((diffs / weights)**2)) * 100
    relative_max_diff = np.amax(diffs / weights) * 100

    return orm.Dict(
        dict={
            'relative_diff': relative_diff,
            'relative_max_diff': relative_max_diff,
            'absolute_diff': absolute_diff,
            'absolute_max_diff': absolute_max_diff,
            'absolute_unit': 'cm-1',
            'relative_unit': '%'
        })


PARA_ECUTWFC_LIST = lambda: orm.List(list=[
    20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 110,
    120, 130, 140, 160, 180, 200
])

PARA_ECUTRHO_LIST = lambda: orm.List(list=[
    160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680, 720,
    760, 800, 880, 960, 1040, 1120, 1280, 1440, 1600
])


class ConvergencePhononFrequenciesWorkChain(BaseConvergenceWorkChain):
    """WorkChain to converge test on cohisive energy of input structure"""

    # hard code parameters of evaluate workflow
    _DEGUASS = 0.00735
    _OCCUPATIONS = 'smearing'
    _SMEARING = 'marzari-vanderbilt'
    _CONV_THR = 1e-10
    _QPOINTS_LIST = [[0.5, 0.5, 0.5]]
    _KDISTANCE = 0.15

    _PH_PARAMETERS = {
        'INPUTPH': {
            'tr2_ph': 1e-16,
            'epsil': False,
        }
    }

    # hard code parameters of convergence workflow
    _TOLERANCE = 1e-1
    _CONV_THR_CONV = 1e-1
    _CONV_WINDOW = 3

    _ABSOLUTE_UNIT = 'cm-1'

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input('pw_code',
                   valid_type=orm.Code,
                   help='The `pw.x` code use for the `PwCalculation`.')
        spec.input('ph_code',
                   valid_type=orm.Code,
                   help='The `ph.x` code use for the `PhCalculation`.')

    def get_create_process(self):
        return PhononFrequenciesWorkChain

    def get_evaluate_process(self):
        return helper_phonon_frequencies_difference

    def get_parsed_results(self):
        return {
            'relative_max_diff': ('The relative max phonon difference', '%'),
            'absolute_max_diff':
            ('The absolute max phonon difference', 'cm-1'),
            'relative_diff':
            ('The relative phonon frequencies difference', '%'),
            'absolute_diff':
            ('The absolute phonon frequencies difference', 'cm-1'),
        }

    def get_converge_y(self):
        return 'relative_diff', '%'

    def get_create_process_inputs(self):
        _PW_PARAS = {   # pylint: disable=invalid-name
            'SYSTEM': {
                'degauss': self._DEGUASS,
                'occupations': self._OCCUPATIONS,
                'smearing': self._SMEARING,
            },
            'ELECTRONS': {
                'conv_thr': self._CONV_THR,
            },
        }
        _PH_PARAS = self._PH_PARAMETERS  # pylint: disable=invalid-name

        inputs = {
            'pw_code': self.inputs.pw_code,
            'ph_code': self.inputs.ph_code,
            'pseudos': self.ctx.pseudos,
            'structure': self.ctx.structure,
            'parameters': {
                'pw':
                orm.Dict(
                    dict=update_dict(_PW_PARAS, self.ctx.base_pw_parameters)),
                'ph':
                orm.Dict(dict=_PH_PARAS),
                'kpoints_distance':
                orm.Float(self._KDISTANCE),
                'qpoints':
                orm.List(list=self._QPOINTS_LIST),
            },
        }

        return inputs

    def get_evaluate_process_inputs(self):
        ref_workchain = self.ctx.ref_workchain

        res = {
            'ref_parameters': ref_workchain.outputs.output_parameters,
        }

        return res

    def get_output_input_mapping(self):
        res = orm.Dict(dict={'output_parameters': 'input_parameters'})
        return res
