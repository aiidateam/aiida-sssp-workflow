# -*- coding: utf-8 -*-
"""
Convergence test on cohesive energy of a given pseudopotential
"""

from aiida import orm
from aiida.engine import calcfunction
from aiida.plugins import DataFactory

from aiida_sssp_workflow.workflows.convergence._base import BaseLegacyWorkChain
from aiida_sssp_workflow.workflows.evaluate._cohesive_energy import (
    CohesiveEnergyWorkChain,
)

UpfData = DataFactory('pseudo.upf')


@calcfunction
def helper_cohesive_energy_difference(input_parameters: orm.Dict,
                                      ref_parameters: orm.Dict) -> orm.Dict:
    """calculate the cohesive energy difference from parameters"""
    res_energy = input_parameters['cohesive_energy_per_atom']
    ref_energy = ref_parameters['cohesive_energy_per_atom']
    absolute_diff = abs(res_energy - ref_energy) * 1000.0
    relative_diff = abs((res_energy - ref_energy) / ref_energy) * 100

    res = {
        'cohesive_energy_per_atom': res_energy,
        'absolute_diff': absolute_diff,
        'relative_diff': relative_diff,
        'absolute_unit': 'meV/atom',
        'relative_unit': '%'
    }

    return orm.Dict(dict=res)


class ConvergenceCohesiveEnergyWorkChain(BaseLegacyWorkChain):
    """WorkChain to converge test on cohisive energy of input structure"""
    # pylint: disable=too-many-instance-attributes

    _PROPERTY_NAME = 'cohesive_energy'
    _EVALUATE_WORKCHAIN = CohesiveEnergyWorkChain
    _MEASURE_OUT_PROPERTY = 'absolute_diff'

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
        self._DEGAUSS = protocol['degauss']
        self._OCCUPATIONS = protocol['occupations']
        self._BULK_SMEARING = protocol['smearing']
        self._ATOM_SMEARING = protocol['atom_smearing']
        self._CONV_THR = protocol['electron_conv_thr']
        self.ctx.kpoints_distance = self._KDISTANCE = protocol['kpoints_distance']
        self.ctx.vacuum_length = self._VACUUM_LENGTH = protocol['vacuum_length']

        # Set context parameters
        self.ctx.bulk_parameters = super()._get_pw_base_parameters(self._DEGAUSS,
                                                                   self._OCCUPATIONS,
                                                                   self._BULK_SMEARING,
                                                                   self._CONV_THR)

        # self.ctx.bulk_parameters = update_dict(self.ctx.bulk_parameters,
        #                                 self.ctx.extra_pw_parameters)

        self.ctx.atom_parameters = {
            'SYSTEM': {
                'degauss': self._DEGAUSS,
                'occupations': self._OCCUPATIONS,
                'smearing': self._ATOM_SMEARING,
            },
            'ELECTRONS': {
                'conv_thr': self._CONV_THR,
            },
        }

        self.report(
            f'The bulk parameters for convergence is: {self.ctx.bulk_parameters}'
        )
        self.report(
            f'The atom parameters for convergence is: {self.ctx.atom_parameters}'
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
            'bulk_parameters': orm.Dict(dict=self.ctx.bulk_parameters),
            'atom_parameters': orm.Dict(dict=self.ctx.atom_parameters),
            'ecutwfc': orm.Float(ecutwfc),
            'ecutrho': orm.Float(ecutrho),
            'kpoints_distance': orm.Float(self.ctx.kpoints_distance),
            'vacuum_length': orm.Float(self.ctx.vacuum_length),
            'options': orm.Dict(dict=self.ctx.options),
            'parallelization': orm.Dict(dict=self.ctx.parallelization),
            'clean_workdir': orm.Bool(False),   # will leave the workdir clean to outer most wf
        }

        return inputs

    def helper_compare_result_extract_fun(self, sample_node, reference_node,
                                          **kwargs):
        """extract"""
        sample_output = sample_node.outputs.output_parameters
        reference_output = reference_node.outputs.output_parameters

        res = helper_cohesive_energy_difference(sample_output,
                                                reference_output).get_dict()

        return res

    def get_result_metadata(self):
        return {
            'absolute_unit': 'eV/atom',
            'relative_unit': '%',
        }
