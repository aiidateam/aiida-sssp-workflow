# -*- coding: utf-8 -*-
"""
Convergence test on cohesive energy of a given pseudopotential
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

CohesiveEnergyWorkChain = WorkflowFactory('sssp_workflow.cohesive_energy')

PARA_ECUTWFC_LIST = lambda: orm.List(list=[
    20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 110,
    120, 130, 140, 160, 180, 200
])

PARA_ECUTRHO_LIST = lambda: orm.List(list=[
    160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680, 720,
    760, 800, 880, 960, 1040, 1120, 1280, 1440, 1600
])


class ConvergenceCohesiveEnergyWorkChain(WorkChain):
    """WorkChain to converge test on cohisive energy of input structure"""
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
        spec.exit_code(
            510,
            'ERROR_DIFFERENT_SIZE_OF_ECUTOFF_PAIRS',
            message='The ecutwfc_list and ecutrho_list have incompatible size.'
        )

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

        self.ctx.structure = cif_data.get_structure()
        self.ctx.pseudos = pseudos
        self.ctx.pw_parameters = pw_parameters

    def get_inputs(self, ecutwfc, ecutrho):
        _PW_BULK_PARAS = {   # pylint: disable=invalid-name
            'SYSTEM': {
                'degauss': 0.01,
                'occupations': 'smearing',
                'smearing': 'marzari-vanderbilt',
                'ecutrho': ecutrho,
                'ecutwfc': ecutwfc,
            },
        }
        _PW_ATOM_PARAS = {   # pylint: disable=invalid-name
            'SYSTEM': {
                'degauss': 0.01,
                'occupations': 'smearing',
                'smearing': 'gaussian',
                'ecutrho': ecutrho,
                'ecutwfc': ecutwfc,
            },
        }
        inputs = AttributeDict({
            'code': self.inputs.code,
            'pseudos': self.ctx.pseudos,
            'structure': self.ctx.structure,
            'parameters': {
                'pw_bulk':
                orm.Dict(
                    dict=update_dict(_PW_BULK_PARAS, self.ctx.pw_parameters)),
                'pw_atom':
                orm.Dict(dict=_PW_ATOM_PARAS),
                'kpoints_distance':
                orm.Float(0.15),
                'vacuum_length':
                orm.Float(12.0),
            },
        })

        return inputs

    def run_ref(self):
        """
        Running the calculation for the reference point
        hard code to 200Ry at the moment
        """
        ecutwfc = 200
        ecutrho = 1600  # TODO if NC set to 800
        inputs = self.get_inputs(ecutwfc, ecutrho)

        running = self.submit(CohesiveEnergyWorkChain, **inputs)

        self.report(
            f'launching reference CohesiveEnergyWorkChain<{running.pk}>.')

        return ToContext(ref_workchain=running)

    def run_all(self):
        """
        Running the calculation for other points
        """
        ref_workchain = self.ctx.ref_workchain

        if not ref_workchain.is_finished_ok:
            self.report(
                f'Reference run of CohesiveEnergyWorkChain failed with exit status {ref_workchain.exit_status}'
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED.format(
                pk=ref_workchain.pk)

        self.ctx.ref_energy = ref_workchain.outputs.output_parameters[
            'cohesive_energy']

        for ecutwfc, ecutrho in zip(self.ctx.ecutwfc_list,
                                    self.ctx.ecutrho_list):
            inputs = self.get_inputs(ecutwfc, ecutrho)

            workchain = self.submit(CohesiveEnergyWorkChain, **inputs)
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
        cohesive_energies = []
        relative_values = []
        ecutwfc_list = []
        ecutrho_list = []

        ref_energy = self.ctx.ref_energy
        for child in success_child:
            ecutwfc = child.inputs.parameters__pw_bulk['SYSTEM']['ecutwfc']
            ecutrho = child.inputs.parameters__pw_bulk['SYSTEM']['ecutrho']
            energy = child.outputs.output_parameters['cohesive_energy']
            relative_value = (energy -
                              ref_energy) / ref_energy * 100.0  # unit %
            ecutwfc_list.append(ecutwfc)
            ecutrho_list.append(ecutrho)

            cohesive_energies.append(energy)
            relative_values.append(relative_value)

        xy_data = orm.XyData()
        xy_data.set_x(np.array(ecutwfc_list), 'wavefunction cutoff', 'Rydberg')
        xy_data.set_y(np.array(relative_values),
                      'Relative values of cohesive energy', '%')

        output_parameters = orm.Dict(
            dict={
                'ecutwfc_list': ecutwfc_list,
                'ecutrho_list': ecutrho_list,
                'cohesive_energies': cohesive_energies,
                'relative_values': relative_values,
                'cutoff_unit': 'Ry',
                'cohesive_energy_unit': 'eV/atom',
                'relative_unit': '%'
            })
        self.out('output_parameters', output_parameters.store())
        self.out('xy_data', xy_data.store())
