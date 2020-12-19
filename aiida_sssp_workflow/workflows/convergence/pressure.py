# -*- coding: utf-8 -*-
"""
Convergence test on cohesive energy of a given pseudopotential
"""
import importlib_resources
from aiida.common import AttributeDict
from aiida.engine import WorkChain, ToContext, append_, calcfunction
from aiida import orm
from aiida.plugins import WorkflowFactory

from aiida.common.files import md5_file
from aiida_sssp_workflow.workflows.delta_factor import helper_parse_upf, \
    get_standard_cif_filename_from_element, \
    helper_create_standard_cif_from_element, \
    RARE_EARTH_ELEMENTS, update_dict

PressureWorkChain = WorkflowFactory('sssp_workflow.pressure_evaluation')

PARA_ECUTWFC_LIST = lambda: orm.List(list=[
    20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 110,
    120, 130, 140, 160, 180, 200
])

PARA_ECUTRHO_LIST = lambda: orm.List(list=[
    160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680, 720,
    760, 800, 880, 960, 1040, 1120, 1280, 1440, 1600
])

RARE_EARTH_ELEMENTS = [
    'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er',
    'Tm', 'Yb', 'Lu'
]


@calcfunction
def helper_get_volume_from_pressure_birch_murnaghan(P, V0, B0, B1):
    """
    Knowing the pressure P and the Birch-Murnaghan equation of state
    parameters, gets the volume the closest to V0 (relatively) that is
    such that P_BirchMurnaghan(V)=P

    retrun unit is (%)
    """
    import numpy as np

    V0 = V0.value
    B0 = B0.value
    B1 = B1.value
    P = P.value
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

    return orm.Float((V - V0) / V0 * 100)


@calcfunction
def helper_get_v0_b0_b1(element: orm.Str):
    import re
    from aiida_sssp_workflow.calculations.wien2k_ref import WIEN2K_REF, WIEN2K_REN_REF

    if element.value == 'F':
        # Use SiF4 as reference of fluorine(F)
        return {
            'V0': orm.Float(19.3583),
            'B0': orm.Float(74.0411),
            'B1': orm.Float(4.1599),
        }

    if element.value in RARE_EARTH_ELEMENTS:
        element_str = f'{element.value}N'
    else:
        element_str = element.value

    regex = re.compile(
        rf"""{element_str}\s*
                        (?P<V0>\d*.\d*)\s*
                        (?P<B0>\d*.\d*)\s*
                        (?P<B1>\d*.\d*)""", re.VERBOSE)
    if element.value not in RARE_EARTH_ELEMENTS:
        match = regex.search(WIEN2K_REF)
        V0 = match.group('V0')
        B0 = match.group('B0')
        B1 = match.group('B1')
    else:
        match = regex.search(WIEN2K_REN_REF)
        V0 = match.group('V0')
        B0 = match.group('B0')
        B1 = match.group('B1')

    return {
        'V0': orm.Float(float(V0)),
        'B0': orm.Float(float(B0)),
        'B1': orm.Float(float(B1)),
    }


class ConvergencePressureWorkChain(WorkChain):
    """WorkChain to converge test on pressure of input structure"""

    _DEGUASS = 0.00735
    _OCCUPATIONS = 'smearing'
    _SMEARING = 'marzari-vanderbilt'
    _KDISTANCE = 0.15
    _CONV_THR = 1e-10

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
            'The output parameters include pressure information of the structures.'
        )
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

            z_valence_RE = upf_info['z_valence']  # pylint: disable=invalid-name
            z_valence_N = helper_parse_upf(upf_nitrogen)['z_valence']  # pylint: disable=invalid-name
            nbands = (z_valence_N + z_valence_RE) // 2
            nbands_factor = 2
            pw_parameters = {
                'SYSTEM': {
                    'nbnd': int(nbands * nbands_factor),
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

        inputs = AttributeDict({
            'code': self.inputs.code,
            'pseudos': self.ctx.pseudos,
            'structure': self.ctx.structure,
            'parameters': {
                'pw':
                orm.Dict(dict=update_dict(_PW_PARAS, self.ctx.pw_parameters)),
                'kpoints_distance': orm.Float(self._KDISTANCE),
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

        running = self.submit(PressureWorkChain, **inputs)

        self.report(f'launching reference PressureWorkChain<{running.pk}>.')

        return ToContext(ref_workchain=running)

    def run_all(self):
        """
        Running the calculation for other points
        """
        ref_workchain = self.ctx.ref_workchain

        if not ref_workchain.is_finished_ok:
            self.report(
                f'Reference run of PressureWorkChain failed with exit status {ref_workchain.exit_status}'
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED.format(
                pk=ref_workchain.pk)

        self.ctx.ref_pressure = ref_workchain.outputs.output_parameters[
            'hydrostatic_stress']

        for ecutwfc, ecutrho in zip(self.ctx.ecutwfc_list,
                                    self.ctx.ecutrho_list):
            inputs = self.get_inputs(ecutwfc, ecutrho)

            workchain = self.submit(PressureWorkChain, **inputs)
            self.report(
                f'submitting pressure evaluation {workchain.pk} on ecutwfc={ecutwfc} ecutrho={ecutrho}.'
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
        absolute_diffs = []
        ecutwfc_list = []
        ecutrho_list = []

        ref_pressure = self.ctx.ref_pressure
        for child in success_child:
            ecutwfc = child.inputs.parameters__pw['SYSTEM']['ecutwfc']
            ecutrho = child.inputs.parameters__pw['SYSTEM']['ecutrho']
            pressure = child.outputs.output_parameters['hydrostatic_stress']
            ecutwfc_list.append(ecutwfc)
            ecutrho_list.append(ecutrho)

            absolute_diff = pressure - ref_pressure

            element = self.inputs.pseudo.element
            res = helper_get_v0_b0_b1(orm.Str(element))
            V0, B0, B1 = res['V0'], res['B0'], res['B1']
            res = helper_get_volume_from_pressure_birch_murnaghan(
                orm.Float(absolute_diff), V0, B0, B1)
            relative_diff = res.value

            relative_diffs.append(relative_diff)
            absolute_diffs.append(absolute_diff)

        xy_data = orm.XyData()
        xy_data.set_x(np.array(ecutwfc_list), 'wavefunction cutoff', 'Rydberg')
        xy_data.set_y(np.array(relative_diffs),
                      'Relative values of phonon frequencies', '%')

        output_parameters = orm.Dict(
            dict={
                'ecutwfc_list': ecutwfc_list,
                'ecutrho_list': ecutrho_list,
                'relative_diff_list': relative_diffs,
                'absolute_diff_list': absolute_diffs,
                'cutoff_unit': 'Ry',
                'relative_unit': '%',
                'absolute_unit': 'GPascal',
            })
        self.out('output_parameters', output_parameters.store())
        self.out('xy_data', xy_data.store())
