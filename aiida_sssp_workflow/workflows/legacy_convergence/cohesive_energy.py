# -*- coding: utf-8 -*-
"""
Convergence test on cohesive energy of a given pseudopotential
"""
import importlib_resources

from aiida.engine import append_, calcfunction, ToContext
from aiida import orm
from aiida.plugins import DataFactory

from aiida_sssp_workflow.utils import update_dict, \
    get_standard_cif_filename_from_element, convergence_analysis
from aiida_sssp_workflow.workflows.legacy_convergence._base import BaseLegacyWorkChain
from aiida_sssp_workflow.workflows.evaluate._cohesive_energy import CohesiveEnergyWorkChain

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
    
    _EVALUATE_WORKCHAIN = CohesiveEnergyWorkChain
    _MEASURE_OUT_PROPERTY = 'absolute_diff'

    @classmethod
    def define(cls, spec):
        super().define(spec)
        # yapf: disable
        spec.input('code', valid_type=orm.Code,
                    help='The `pw.x` code use for the `PwCalculation`.')
        # yapy: enable

    def init_setup(self):
        super().init_setup()

    def extra_setup_for_magnetic_element(self):
        """Extra setup for magnetic element"""
        super().extra_setup_for_magnetic_element()

    def extra_setup_for_rare_earth_element(self):
        super().extra_setup_for_rare_earth_element()

    def extra_setup_for_fluorine_element(self):
        """Extra setup for fluorine element"""
        cif_file = get_standard_cif_filename_from_element('SiF4')
        self.ctx.structure = orm.CifData.get_or_create(
            cif_file, use_first=True)[0].get_structure(primitive_cell=True)

        # setting pseudos
        import_path = importlib_resources.path(
            'aiida_sssp_workflow.REF.UPFs', 'Si.pbe-n-rrkjus_psl.1.0.0.UPF')
        with import_path as pp_path, open(pp_path, 'rb') as stream:
            upf_silicon = UpfData(stream)
            self.ctx.pseudos['Si'] = upf_silicon

    def setup_code_parameters_from_protocol(self):
        """Input validation"""
        # pylint: disable=invalid-name, attribute-defined-outside-init

        # Read from protocol if parameters not set from inputs
        super().setup_code_parameters_from_protocol()

        protocol = self.ctx.protocol_calculation['convergence']['cohesive_energy']
        self._DEGAUSS = protocol['degauss']
        self._OCCUPATIONS = protocol['occupations']
        self._BULK_SMEARING = protocol['bulk_smearing']
        self._ATOM_SMEARING = protocol['atom_smearing']
        self._CONV_THR = protocol['electron_conv_thr']
        self._KDISTANCE = protocol['kpoints_distance']
        self._VACUUM_LENGTH = protocol['vacuum_length']

        self._MAX_EVALUATE = protocol['max_evaluate']
        self._REFERENCE_ECUTWFC = protocol['reference_ecutwfc']
        self._NUM_OF_RHO_TEST = protocol['num_of_rho_test']

        self.ctx.vacuum_length = self._VACUUM_LENGTH

        self.ctx.max_evaluate = self._MAX_EVALUATE
        self.ctx.reference_ecutwfc = self._REFERENCE_ECUTWFC

        self.ctx.bulk_parameters = {
            'SYSTEM': {
                'degauss': self._DEGAUSS,
                'occupations': self._OCCUPATIONS,
                'smearing': self._BULK_SMEARING,
            },
            'ELECTRONS': {
                'conv_thr': self._CONV_THR,
            },
        }

        self.ctx.bulk_parameters = update_dict(self.ctx.bulk_parameters,
                                        self.ctx.extra_pw_parameters)

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

        self.ctx.kpoints_distance = self._KDISTANCE

        self.report(
            f'The bulk parameters for convergence is: {self.ctx.bulk_parameters}'
        )
        self.report(
            f'The atom parameters for convergence is: {self.ctx.atom_parameters}'
        )

    def setup_criteria_parameters_from_protocol(self):
        """Input validation"""
        # pylint: disable=invalid-name, attribute-defined-outside-init

        # Read from protocol if parameters not set from inputs
        super().setup_criteria_parameters_from_protocol()

        self.ctx.criteria = self.ctx.protocol_criteria['convergence']['cohesive_energy']

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
        # yapf: enable

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
