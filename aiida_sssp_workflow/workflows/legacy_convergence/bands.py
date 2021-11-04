# -*- coding: utf-8 -*-
"""
Convergence test on bands of a given pseudopotential
"""
import importlib_resources

from aiida import orm
from aiida.engine import append_, ToContext
from aiida.plugins import DataFactory

from aiida_sssp_workflow.workflows.evaluate._bands import BandsWorkChain
from aiida_sssp_workflow.calculations import calculate_bands_distance
from aiida_sssp_workflow.utils import update_dict, \
    NONMETAL_ELEMENTS, \
    get_standard_cif_filename_from_element
from aiida_sssp_workflow.workflows.legacy_convergence._base import BaseLegacyWorkChain

UpfData = DataFactory('pseudo.upf')

helper_bands_distence_difference = calculate_bands_distance.calculate_bands_distance


class ConvergenceBandsWorkChain(BaseLegacyWorkChain):
    """WorkChain to converge test on cohisive energy of input structure"""
    # pylint: disable=too-many-instance-attributes

    _INIT_NBANDS_FACTOR = 3.0
    _RY_TO_EV = 13.6056980659

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        super().define(spec)
        # yapf: disable
        spec.input('code', valid_type=orm.Code,
                    help='The `pw.x` code use for the `PwCalculation`.')
        # yapy: enable

    def init_setup(self):
        super().init_setup()

        self.ctx.pw_parameters = {}
        self.ctx.extra_parameters = {}

        # extra setting for bands convergence
        self.ctx.is_metal = self.ctx.element not in NONMETAL_ELEMENTS

    def extra_setup_for_rare_earth_element(self):
        """Extra setup for rare earth element"""
        import_path = importlib_resources.path('aiida_sssp_workflow.REF.UPFs',
                                               'N.pbe-n-radius_5.UPF')
        with import_path as pp_path, open(pp_path, 'rb') as stream:
            upf_nitrogen = UpfData(stream)
            self.ctx.pseudos['N'] = upf_nitrogen

        # In rare earth case, increase the initial number of bands,
        # otherwise the occupation will not fill up in the highest band
        # which always trigger the `PwBaseWorkChain` sanity check.
        nbands = self.inputs.pseudo.z_valence + upf_nitrogen.z_valence // 2
        nbands_factor = 2

        self.ctx.extra_parameters = {
            'SYSTEM': {
                'nbnd': int(nbands * nbands_factor),
            },
        }

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
        protocol_name = self.inputs.protocol.value
        protocol = self._get_protocol()[protocol_name]
        protocol = protocol['convergence']['bands_distance']
        self._DEGAUSS = protocol['degauss']
        self._OCCUPATIONS = protocol['occupations']
        self._SMEARING = protocol['smearing']
        self._CONV_THR = protocol['electron_conv_thr']
        self._KDISTANCE = protocol['kpoints_distance']

        self._MAX_EVALUATE = protocol['max_evaluate']
        self._REFERENCE_ECUTWFC = protocol['reference_ecutwfc']

        self.ctx.max_evaluate = self._MAX_EVALUATE
        self.ctx.reference_ecutwfc = self._REFERENCE_ECUTWFC

        self.ctx.degauss = self._DEGAUSS    # for band distance calculation

        self.ctx.pw_parameters = {
            'SYSTEM': {
                'degauss': self._DEGAUSS,
                'occupations': self._OCCUPATIONS,
                'smearing': self._SMEARING,
            },
            'ELECTRONS': {
                'conv_thr': self._CONV_THR,
            },
        }

        self.ctx.pw_parameters = update_dict(self.ctx.pw_parameters,
                                        self.ctx.extra_parameters)

        # set the ecutrho according to the type of pseudopotential
        # dual 4 for NC and 8 for all other type of PP.
        if self.ctx.pseudo_type in ['NC', 'SL']:
            dual = 4.0
        else:
            dual = 8.0

        if 'dual' in self.inputs:
            dual = self.inputs.dual

        self.ctx.dual = dual

        self.ctx.kpoints_distance = self._KDISTANCE

        self.report(
            f'The pw parameters for convergence is: {self.ctx.pw_parameters}'
        )

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
            'init_nbands_factor': orm.Float(self._INIT_NBANDS_FACTOR),
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
        ecutwfc = self.ctx.reference_ecutwfc
        ecutrho = ecutwfc * self.ctx.dual
        inputs = self._get_inputs(ecutwfc=ecutwfc, ecutrho=ecutrho)

        running = self.submit(BandsWorkChain, **inputs)
        self.report(f'launching reference BandsWorkChain<{running.pk}>')

        return ToContext(reference=running)

    def run_samples(self):
        """
        run on all other evaluation sample points
        """
        workchain = self.ctx.reference

        if not workchain.is_finished_ok:
            self.report(
                f'PwBaseWorkChain pk={workchain.pk} for reference run is failed.'
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED.format(
                label='reference')

        for idx in range(self.ctx.max_evaluate):
            ecutwfc = self._ECUTWFC_LIST[idx]
            ecutrho = ecutwfc * self.ctx.dual
            inputs = self._get_inputs(ecutwfc=ecutwfc, ecutrho=ecutrho)

            running = self.submit(BandsWorkChain, **inputs)
            self.report(f'launching [{idx}] BandsWorkChain<{running.pk}>')

            self.to_context(children=append_(running))

    def get_result_metadata(self):
        return {
            'band_unit': 'eV',
        }

    def helper_compare_result_extract_fun(self, sample_node, reference_node,
                                          **kwargs):
        """implement"""
        sample_bands_output = sample_node.outputs.output_bands_parameters
        reference_bands_output = reference_node.outputs.output_bands_parameters

        sample_bands_structure = sample_node.outputs.output_bands_structure
        reference_bands_structure = reference_node.outputs.output_bands_structure

        extra_parameters = kwargs['extra_parameters']
        res = helper_bands_distence_difference(sample_bands_structure,
                                               sample_bands_output,
                                               reference_bands_structure,
                                               reference_bands_output,
                                               **extra_parameters).get_dict()

        return res

    def results(self):
        """
        results
        """
        extra_parameters = {
            'smearing': orm.Float(self.ctx.degauss * self._RY_TO_EV),
            'is_metal': orm.Bool(self.ctx.is_metal)
        }
        output_parameters = self.result_general_process(
            extra_parameters=extra_parameters)

        self.out('output_parameters', orm.Dict(dict=output_parameters).store())
