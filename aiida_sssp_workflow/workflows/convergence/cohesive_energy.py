# -*- coding: utf-8 -*-
"""
Convergence test on cohesive energy of a given pseudopotential
"""
from aiida.engine import calcfunction
from aiida import orm

from aiida_sssp_workflow.utils import update_dict
from aiida_sssp_workflow.workflows.cohesive_energy import CohesiveEnergyWorkChain
from .base import BaseConvergenceWorkChain


@calcfunction
def helper_cohesive_energy_difference(input_parameters: orm.Dict,
                                      ref_parameters: orm.Dict) -> orm.Dict:
    """calculate the cohesive energy difference from parameters"""
    res_energy = input_parameters['cohesive_energy_per_atom']
    ref_energy = ref_parameters['cohesive_energy_per_atom']
    absolute_diff = abs(res_energy - ref_energy)
    relative_diff = abs((res_energy - ref_energy) / ref_energy) * 100

    res = {
        'absolute_diff': absolute_diff,
        'relative_diff': relative_diff,
        'absolute_unit': 'eV/atom',
        'relative_unit': '%'
    }

    return orm.Dict(dict=res)


PARA_ECUTWFC_LIST = lambda: orm.List(list=[
    20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 110,
    120, 130, 140, 160, 180, 200
])

PARA_ECUTRHO_LIST = lambda: orm.List(list=[
    160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680, 720,
    760, 800, 880, 960, 1040, 1120, 1280, 1440, 1600
])


class ConvergenceCohesiveEnergyWorkChain(BaseConvergenceWorkChain):
    """WorkChain to converge test on cohisive energy of input structure"""
    # pylint: disable=too-many-instance-attributes

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
        protocol = protocol['convergence']['cohesive_energy']
        self.ctx._DEGAUSS = protocol['degauss']
        self.ctx._OCCUPATIONS = protocol['occupations']
        self.ctx._BULK_SMEARING = protocol['bulk_smearing']
        self.ctx._ATOM_SMEARING = protocol['atom_smearing']
        self.ctx._CONV_THR_EVA = protocol['electron_conv_thr']
        self.ctx._KDISTANCE = protocol['kpoints_distance']
        self.ctx._VACUUM_LENGTH = protocol['vaccum_length']

        self.ctx._TOLERANCE = protocol['tolerance']
        self.ctx._CONV_THR_CONV = protocol['convergence_conv_thr']
        self.ctx._CONV_WINDOW = protocol['convergence_window']

    def get_create_process(self):
        return CohesiveEnergyWorkChain

    def get_evaluate_process(self):
        return helper_cohesive_energy_difference

    def get_parsed_results(self):
        return {
            'absolute_diff': ('The absolute cohesive difference', 'eV/atom'),
            'relative_diff': ('The relative cohesive difference', '%'),
        }

    def get_converge_y(self):
        return 'absolute_diff', 'eV/atom'

    def get_create_process_inputs(self):
        _PW_BULK_PARAS = {   # pylint: disable=invalid-name
            'SYSTEM': {
                'degauss': self.ctx._DEGAUSS,
                'occupations': self.ctx._OCCUPATIONS,
                'smearing': self.ctx._BULK_SMEARING,
            },
            'ELECTRONS': {
                'conv_thr': self.ctx._CONV_THR_EVA,
            },
        }
        _PW_ATOM_PARAS = {   # pylint: disable=invalid-name
            'SYSTEM': {
                'degauss': self.ctx._DEGAUSS,
                'occupations': self.ctx._OCCUPATIONS,
                'smearing': self.ctx._ATOM_SMEARING,
            },
            'ELECTRONS': {
                'conv_thr': self.ctx._CONV_THR_EVA,
            },
        }
        inputs = {
            'code': self.inputs.code,
            'pseudos': self.ctx.pseudos,
            'structure': self.ctx.structure,
            'parameters': {
                'pw_bulk':
                orm.Dict(dict=update_dict(_PW_BULK_PARAS,
                                          self.ctx.base_pw_parameters)),
                'pw_atom':
                orm.Dict(dict=_PW_ATOM_PARAS),
                'kpoints_distance':
                orm.Float(self.ctx._KDISTANCE),
                'vacuum_length':
                orm.Float(self.ctx._VACUUM_LENGTH),
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
