# -*- coding: utf-8 -*-
"""
Convergence test on cohesive energy of a given pseudopotential
"""
from aiida.common import AttributeDict
from aiida.engine import WorkChain, ToContext, append_, calcfunction
from aiida import orm
from aiida.plugins import WorkflowFactory

from aiida_sssp_workflow.utils import update_dict
from ..helper import get_pw_inputs_from_pseudo

PhononFrequenciesWorkChain = WorkflowFactory(
    'sssp_workflow.phonon_frequencies')

PARA_ECUTWFC_LIST = lambda: orm.List(list=[
    20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 110,
    120, 130, 140, 160, 180, 200
])

PARA_ECUTRHO_LIST = lambda: orm.List(list=[
    160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680, 720,
    760, 800, 880, 960, 1040, 1120, 1280, 1440, 1600
])


@calcfunction
def helper_get_relative_phonon_frequencies(
        frequencies: orm.List, ref_frequencies: orm.List) -> orm.Dict:
    """
    doc
    """
    import numpy as np

    diffs = np.array(frequencies.get_list()) - np.array(
        ref_frequencies.get_list())
    weights = np.array(ref_frequencies.get_list())

    absolute_diff = np.mean(diffs)
    absolute_max_diff = np.amax(diffs)

    relative_diff = np.sqrt(np.mean((diffs / weights)**2))
    relative_max_diff = np.amax(diffs / weights)

    return orm.Dict(
        dict={
            'relative_diff': relative_diff * 100,
            'relative_max_diff': relative_max_diff * 100,
            'absolute_diff': absolute_diff,
            'absolute_max_diff': absolute_max_diff,
        })


class ConvergencePhononFrequenciesWorkChain(WorkChain):
    """WorkChain to converge test on cohisive energy of input structure"""
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

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input('pw_code',
                   valid_type=orm.Code,
                   help='The `pw.x` code use for the `PwCalculation`.')
        spec.input('ph_code',
                   valid_type=orm.Code,
                   help='The `ph.x` code use for the `PwCalculation`.')
        spec.input('pseudo',
                   valid_type=orm.UpfData,
                   required=True,
                   help='Pseudopotential to be verified')
        spec.input('options',
                   valid_type=orm.Dict,
                   required=False,
                   help='Optional `options` to use for the `PwCalculations`.')
        spec.input_namespace('parameters', help='Para')
        spec.input('parameters.ecutrho_list',
                   valid_type=orm.List,
                   default=lambda: PARA_ECUTRHO_LIST,
                   help='dual value for ecutrho list.')
        spec.input('parameters.ecutwfc_list',
                   valid_type=orm.List,
                   default=PARA_ECUTWFC_LIST,
                   help='list of ecutwfc evaluate list.')
        spec.input('parameters.ref_cutoff_pair',
                   valid_type=orm.List,
                   required=True,
                   default=lambda: orm.List(list=[200, 1600]),
                   help='ecutwfc/ecutrho pair for reference calculation.')
        spec.outline(
            cls.setup,
            cls.validate_structure,
            cls.run_ref,
            cls.run_all,
            cls.results,
        )
        spec.output(
            'output_parameters',
            valid_type=orm.Dict,
            required=True,
            help=
            'The output parameters include cohesive energy of the structure.')
        spec.output(
            'xy_data',
            valid_type=orm.XyData,
            required=True,
            help='The output XY data for plot use; the x axis is ecutwfc.')
        spec.exit_code(
            400,
            'ERROR_SUB_PROCESS_FAILED',
            message='The sub processes {pk} did not finish successfully.')

    def setup(self):
        self.ctx.ecutwfc_list = self.inputs.parameters.ecutwfc_list.get_list()
        self.ctx.ecutrho_list = self.inputs.parameters.ecutrho_list.get_list()
        if not len(self.ctx.ecutwfc_list) == len(self.ctx.ecutrho_list):
            return self.exit_codes.ERROR_DIFFERENT_SIZE_OF_ECUTOFF_PAIRS

    def validate_structure(self):
        res = get_pw_inputs_from_pseudo(pseudo=self.inputs.pseudo)

        self.ctx.structure = res['structure']
        self.ctx.pseudos = res['pseudos']
        self.ctx.base_pw_parameters = res['base_pw_parameters']

    def get_inputs(self, ecutwfc, ecutrho):
        _PW_PARAS = {   # pylint: disable=invalid-name
            'SYSTEM': {
                'degauss': self._DEGUASS,
                'occupations': self._OCCUPATIONS,
                'smearing': self._SMEARING,
                'ecutrho': ecutrho,
                'ecutwfc': ecutwfc,
            },
            'ELECTRONS': {
                'conv_thr': self._CONV_THR,
            },
        }
        _PH_PARAS = self._PH_PARAMETERS  # pylint: disable=invalid-name

        inputs = AttributeDict({
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
        })

        return inputs

    def run_ref(self):
        """
        Running the calculation for the reference point
        hard code to 200Ry at the moment
        """
        cutoff_pair = self.inputs.parameters.ref_cutoff_pair.get_list()
        ecutwfc = cutoff_pair[0]
        ecutrho = cutoff_pair[1]
        inputs = self.get_inputs(ecutwfc, ecutrho)

        running = self.submit(PhononFrequenciesWorkChain, **inputs)

        self.report(
            f'launching reference PhononFrequenciesWorkChain<{running.pk}>.')

        return ToContext(ref_workchain=running)

    def run_all(self):
        """
        Running the calculation for other points
        """
        ref_workchain = self.ctx.ref_workchain

        if not ref_workchain.is_finished_ok:
            self.report(
                f'Reference run of PhononFrequenciesWorkChain failed with exit status {ref_workchain.exit_status}'
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED.format(
                pk=ref_workchain.pk)

        self.ctx.ref_frequencies = ref_workchain.outputs.output_parameters[
            'dynamical_matrix_0']['frequencies']

        for ecutwfc, ecutrho in zip(self.ctx.ecutwfc_list,
                                    self.ctx.ecutrho_list):
            inputs = self.get_inputs(ecutwfc, ecutrho)

            workchain = self.submit(PhononFrequenciesWorkChain, **inputs)
            self.report(
                f'submitting cohesive energy evaluation {workchain.pk} on ecutwfc={ecutwfc} ecutrho={ecutrho}.'
            )
            self.to_context(children=append_(workchain))

    def results(self):
        """
        doc
        """
        import numpy as np

        pks = [
            child.pk for child in self.ctx.children if not child.is_finished_ok
        ]
        if pks:
            # TODO failed only when points are not enough < 80%
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED.format(pk=pks)

        success_child = [
            child for child in self.ctx.children if child.is_finished_ok
        ]

        relative_diffs = []
        relative_max_diffs = []
        absolute_diffs = []
        absolute_max_diffs = []
        ecutwfc_list = []
        ecutrho_list = []

        ref_frequencies = self.ctx.ref_frequencies
        for child in success_child:
            ecutwfc = child.inputs.parameters__pw['SYSTEM']['ecutwfc']
            ecutrho = child.inputs.parameters__pw['SYSTEM']['ecutrho']
            frequencies = child.outputs.output_parameters[
                'dynamical_matrix_0']['frequencies']

            ecutwfc_list.append(ecutwfc)
            ecutrho_list.append(ecutrho)

            res = helper_get_relative_phonon_frequencies(
                orm.List(list=frequencies), orm.List(list=ref_frequencies))
            relative_diff = res['relative_diff']
            relative_max_diff = res['relative_max_diff']
            absolute_diff = res['absolute_diff']
            absolute_max_diff = res['absolute_max_diff']

            relative_diffs.append(relative_diff)
            relative_max_diffs.append(relative_max_diff)
            absolute_diffs.append(absolute_diff)
            absolute_max_diffs.append(absolute_max_diff)

        xy_data = orm.XyData()
        xy_data.set_x(np.array(ecutwfc_list), 'wavefunction cutoff', 'Rydberg')
        xy_data.set_y(np.array(relative_diffs),
                      'Relative values of phonon frequencies', '%')

        output_parameters = orm.Dict(
            dict={
                'ecutwfc_list': ecutwfc_list,
                'ecutrho_list': ecutrho_list,
                'relative_diff_list': relative_diffs,
                'relative_max_diff_list': relative_max_diffs,
                'absolute_diff_list': absolute_diffs,
                'absolute_max_diff_list': absolute_max_diffs,
                'cutoff_unit': 'Ry',
                'relative_unit': '%',
                'absolute_unit': 'cm-1',
            })
        self.out('output_parameters', output_parameters.store())
        self.out('xy_data', xy_data.store())
