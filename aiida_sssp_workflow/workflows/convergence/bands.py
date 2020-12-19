# -*- coding: utf-8 -*-
"""
Convergence test on bands of a given pseudopotential
"""
import importlib_resources
from aiida.common import AttributeDict
from aiida.engine import WorkChain, ToContext, append_
from aiida import orm
from aiida.plugins import WorkflowFactory

from aiida.common.files import md5_file
from aiida_sssp_workflow.workflows.delta_factor import helper_parse_upf, \
    get_standard_cif_filename_from_element, \
    helper_create_standard_cif_from_element, \
    RARE_EARTH_ELEMENTS, update_dict

BandsWorkChain = WorkflowFactory('sssp_workflow.bands')

PARA_ECUTWFC_LIST = lambda: orm.List(list=[
    20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 110,
    120, 130, 140, 160, 180, 200
])

PARA_ECUTRHO_LIST = lambda: orm.List(list=[
    160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680, 720,
    760, 800, 880, 960, 1040, 1120, 1280, 1440, 1600
])


class ConvergenceBandsWorkChain(WorkChain):
    """WorkChain to converge test on cohisive energy of input structure"""

    _DEGUASS = 0.00735
    _OCCUPATIONS = 'smearing'
    _SMEARING = 'marzari-vanderbilt'
    _CONV_THR = 1e-10
    _SCF_KDISTANCE = 0.15
    _BAND_KDISTANCE = 0.20

    _RY_TO_EV = 13.6056980659

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input('code',
                   valid_type=orm.Code,
                   help='The `pw.x` code use for the `PwCalculation`.')
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
        upf_info = helper_parse_upf(self.inputs.pseudo)
        element = orm.Str(upf_info['element'])

        pseudos = {element.value: self.inputs.pseudo}
        self.ctx.element = element

        if element.value == 'F':
            self.ctx.element = orm.Str('SiF4')

            fpath = importlib_resources.path('aiida_sssp_workflow.REF.UPFs',
                                             'Si.pbe-n-rrkjus_psl.1.0.0.UPF')
            with fpath as path:
                filename = str(path)
                upf_silicon = orm.UpfData.get_or_create(filename)[0]
                pseudos['Si'] = upf_silicon

        pw_parameters = {}
        if element.value in RARE_EARTH_ELEMENTS:
            fpath = importlib_resources.path('aiida_sssp_workflow.REF.UPFs',
                                             'N.pbe-n-radius_5.UPF')
            with fpath as path:
                filename = str(path)
                upf_nitrogen = orm.UpfData.get_or_create(filename)[0]
                pseudos['N'] = upf_nitrogen

            # z_valence_RE = upf_info['z_valence']  # pylint: disable=invalid-name
            # z_valence_N = helper_parse_upf(upf_nitrogen)['z_valence']  # pylint: disable=invalid-name
            # nbands = (z_valence_N + z_valence_RE) // 2
            # nbands_factor = 2
            pw_parameters = {
                'SYSTEM': {
                    # nbnd should not appear with nbnd_factor setting in sub-workflow
                    # 'nbnd': int(nbands * nbands_factor),
                },
            }

        filename = get_standard_cif_filename_from_element(
            self.ctx.element.value)

        md5 = md5_file(filename)
        cifs = orm.CifData.from_md5(md5)
        if not cifs:
            # cif not stored, create it with calcfunction and return it
            cif_data = helper_create_standard_cif_from_element(
                self.ctx.element)
        else:
            # The Cif is already store let's return it
            cif_data = orm.CifData.get_or_create(filename)[0]

        self.ctx.structure = cif_data.get_structure(primitive_cell=True)
        self.ctx.pseudos = pseudos
        self.ctx.pw_parameters = pw_parameters

    def get_inputs(self,
                   ecutwfc,
                   ecutrho,
                   nbands_factor: orm.Float = orm.Float(2.0)):
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

        inputs = AttributeDict({
            'code': self.inputs.code,
            'pseudos': self.ctx.pseudos,
            'structure': self.ctx.structure,
            'parameters': {
                'pw':
                orm.Dict(dict=update_dict(_PW_PARAS, self.ctx.pw_parameters)),
                'scf_kpoints_distance': orm.Float(self._SCF_KDISTANCE),
                'bands_kpoints_distance': orm.Float(self._BAND_KDISTANCE),
                'nbands_factor': nbands_factor,
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

        running = self.submit(BandsWorkChain, **inputs)

        self.report(f'launching reference BandsWorkChain<{running.pk}>.')

        return ToContext(ref_workchain=running)

    def run_all(self):
        """
        Running the calculation for other points
        """
        ref_workchain = self.ctx.ref_workchain

        if not ref_workchain.is_finished_ok:
            self.report(
                f'Reference run of BandsWorkChain failed with exit status {ref_workchain.exit_status}'
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED.format(
                pk=ref_workchain.pk)

        self.ctx.ref_bands = {
            'bands': ref_workchain.outputs.band_structure,
            'parameters': ref_workchain.outputs.band_parameters,
        }
        nbands_factor = ref_workchain.outputs.nbands_factor

        for ecutwfc, ecutrho in zip(self.ctx.ecutwfc_list,
                                    self.ctx.ecutrho_list):
            inputs = self.get_inputs(ecutwfc, ecutrho, nbands_factor)

            workchain = self.submit(BandsWorkChain, **inputs)
            self.report(
                f'submitting bands evaluation {workchain.pk} on ecutwfc={ecutwfc} ecutrho={ecutrho}.'
            )
            self.to_context(children=append_(workchain))

    def results(self):
        """
        doc
        """
        import numpy as np
        from aiida_sssp_workflow.utils import NONMETAL_ELEMENTS
        from aiida_sssp_workflow.calculations.calculate_bands_distance import calculate_bands_distance

        pks = [
            child.pk for child in self.ctx.children if not child.is_finished_ok
        ]
        if pks:
            # TODO failed only when points are not enough < 80%
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED.format(pk=pks)

        success_child = [
            child for child in self.ctx.children if child.is_finished_ok
        ]

        eta_v_list = []
        eta_10_list = []
        max_diff_v_list = []
        max_diff_10_list = []
        ecutwfc_list = []
        ecutrho_list = []

        ref_bands = self.ctx.ref_bands
        for child in success_child:
            ecutwfc = child.inputs.parameters__pw['SYSTEM']['ecutwfc']
            ecutrho = child.inputs.parameters__pw['SYSTEM']['ecutrho']
            bands = {
                'bands': child.outputs.band_structure,
                'parameters': child.outputs.band_parameters,
            }

            ecutwfc_list.append(ecutwfc)
            ecutrho_list.append(ecutrho)

            smearing = orm.Float(0.00735 * self._RY_TO_EV)
            if self.inputs.pseudo.element in NONMETAL_ELEMENTS:
                is_metal = orm.Bool(False)
            else:
                is_metal = orm.Bool(True)
            res = calculate_bands_distance(bands['bands'],
                                           ref_bands['bands'],
                                           bands['parameters'],
                                           ref_bands['parameters'],
                                           smearing=smearing,
                                           is_metal=is_metal)

            self.report(f'The bands distance results are '
                        f'eta_v={res.get("eta_v", None).value};'
                        f'eta_10={res.get("eta_10", None).value}; '
                        f'max_diff_v={res.get("max_diff_v", None).value}; '
                        f'max_diff_10={res.get("max_diff_10", None).value}.')
            eta_v_list.append(res.get('eta_v', None).value)
            eta_10_list.append(res.get('eta_10', None).value)
            max_diff_v_list.append(res.get('max_diff_v', None).value)
            max_diff_10_list.append(res.get('max_diff_10', None).value)

        xy_data = orm.XyData()
        xy_data.set_x(np.array(ecutwfc_list), 'wavefunction cutoff', 'Rydberg')
        xy_data.set_y(np.array(eta_v_list), 'eta_v', '(dl)')

        output_parameters = orm.Dict(
            dict={
                'ecutwfc_list': ecutwfc_list,
                'ecutrho_list': ecutrho_list,
                'eta_v_list': eta_v_list,
                'eta_10_list': eta_10_list,
                'max_diff_v_list': max_diff_v_list,
                'max_diff_10_list': max_diff_10_list,
                'cutoff_unit': 'Ry',
                'relative_unit': '(dl)',
            })
        self.out('output_parameters', output_parameters.store())
        self.out('xy_data', xy_data.store())
