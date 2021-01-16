# -*- coding: utf-8 -*-
"""Workchain to calculate delta factor of specific psp"""
import pathlib
import yaml

from aiida import orm
from aiida.common import AttributeDict
from aiida.engine import WorkChain, ToContext, calcfunction

from aiida_sssp_workflow.utils import update_dict, \
    MAGNETIC_ELEMENTS, \
    RARE_EARTH_ELEMENTS, \
    helper_parse_upf
from aiida_sssp_workflow.helpers import get_pw_inputs_from_pseudo
from aiida_sssp_workflow.calculations.calculate_delta import calculate_delta
from aiida_sssp_workflow.workflows.eos import EquationOfStateWorkChain


@calcfunction
def helper_get_magnetic_inputs(structure: orm.StructureData):
    """
    To set initial magnet to the magnetic system, need to set magnetic order to
    every magnetic element site, with certain pw starting_mainetization parameters.
    """
    MAG_INIT_Mn = {"Mn1": 0.5, "Mn2": -0.3, "Mn3": 0.5, "Mn4": -0.3}  # pylint: disable=invalid-name
    MAG_INIT_O = {"O1": 0.5, "O2": 0.5, "O3": -0.5, "O4": -0.5}  # pylint: disable=invalid-name
    MAG_INIT_Cr = {"Cr1": 0.5, "Cr2": -0.5}  # pylint: disable=invalid-name

    mag_structure = orm.StructureData(cell=structure.cell, pbc=structure.pbc)
    kind_name = structure.get_kind_names()[0]

    parameters = orm.Dict(dict={
        'SYSTEM': {
            'nspin': 2,
        },
    })
    # ferromagnetic
    if kind_name in ['Fe', 'Co', 'Ni']:
        for i, site in enumerate(structure.sites):
            mag_structure.append_atom(position=site.position,
                                      symbols=kind_name)

        parameters = orm.Dict(dict={
            'SYSTEM': {
                'nspin': 2,
                'starting_magnetization': {
                    kind_name: 0.2
                },
            },
        })

    #
    if kind_name in ['Mn', 'O', 'Cr']:
        for i, site in enumerate(structure.sites):
            mag_structure.append_atom(position=site.position,
                                      symbols=kind_name,
                                      name=f'{kind_name}{i+1}')

        if kind_name == 'Mn':
            parameters = orm.Dict(dict={
                'SYSTEM': {
                    'nspin': 2,
                    'starting_magnetization': MAG_INIT_Mn,
                },
            })

        if kind_name == 'O':
            parameters = orm.Dict(dict={
                'SYSTEM': {
                    'nspin': 2,
                    'starting_magnetization': MAG_INIT_O,
                },
            })

        if kind_name == 'Cr':
            parameters = orm.Dict(dict={
                'SYSTEM': {
                    'nspin': 2,
                    'starting_magnetization': MAG_INIT_Cr,
                },
            })

    return {
        'structure': mag_structure,
        'parameters': parameters,
    }


class DeltaFactorWorkChain(WorkChain):
    """Workchain to calculate delta factor of specific pseudopotential"""
    # pylint: disable=too-many-instance-attributes

    _PW_PARAMETERS = {
        'SYSTEM': {
            'degauss': 0.00735,
            'occupations': 'smearing',
            'smearing': 'marzari-vanderbilt',
        },
        'ELECTRONS': {
            'conv_thr': 1e-10,
        },
    }

    _MAX_WALLCLOCK_SECONDS = 1800 * 3

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        super().define(spec)
        spec.input('code',
                   valid_type=orm.Code,
                   help='The `pw.x` code use for the `PwCalculation`.')
        spec.input('pseudo',
                   valid_type=orm.UpfData,
                   required=True,
                   help='Pseudopotential to be verified')
        spec.input(
            'structure',
            valid_type=orm.StructureData,
            required=False,
            help='Ground state structure which the verification perform')
        spec.input('protocol',
                   valid_type=orm.Str,
                   default=lambda: orm.Str('efficiency'),
                   help='The protocol to use for the workchain.')
        spec.input('options',
                   valid_type=orm.Dict,
                   required=False,
                   help='Optional `options` to use for the `PwCalculations`.')
        spec.input_namespace('parameters', help='Para')
        spec.input('parameters.pw',
                   valid_type=orm.Dict,
                   required=False,
                   help='parameters for pwscf.')
        spec.input(
            'parameters.ecutwfc',
            valid_type=(orm.Float, orm.Int),
            required=False,
            help=
            'The ecutwfc set for both atom and bulk calculation. Please also set ecutrho if ecutwfc is set.'
        )
        spec.input(
            'parameters.ecutrho',
            valid_type=(orm.Float, orm.Int),
            required=False,
            help=
            'The ecutrho set for both atom and bulk calculation.  Please also set ecutwfc if ecutrho is set.'
        )
        spec.input('parameters.kpoints_distance',
                   valid_type=orm.Float,
                   required=False,
                   help='Global kpoints setting.')
        spec.input('parameters.scale_count',
                   valid_type=orm.Int,
                   required=False,
                   help='Numbers of scale points in eos step.')
        spec.input('parameters.scale_increment',
                   valid_type=orm.Float,
                   required=False,
                   help='The scale increment in eos step.')
        spec.input(
            'clean_workdir',
            valid_type=orm.Bool,
            default=lambda: orm.Bool(False),
            help=
            'If `True`, work directories of all called calculation will be cleaned at the end of execution.'
        )
        spec.outline(
            cls.setup,
            cls.validate_structure_and_pseudo,
            cls.run_eos,
            cls.inspect_eos,
            cls.run_delta_calc,
            cls.results,
        )
        spec.output('output_eos_parameters',
                    valid_type=orm.Dict,
                    required=True,
                    help='The eos outputs.')
        spec.output('output_parameters',
                    valid_type=orm.Dict,
                    required=True,
                    help='The delta factor of the pseudopotential.')
        spec.output(
            'output_pseudo_header',
            valid_type=orm.Dict,
            required=True,
            help='The header(important parameters) of the pseudopotential.')
        spec.output('output_birch_murnaghan_fit',
                    valid_type=orm.Dict,
                    required=True,
                    help='The results V0, B0, B1 of Birch-Murnaghan fit.')
        spec.exit_code(
            201,
            'ERROR_SUB_PROCESS_FAILED_EOS',
            message='The `EquationOfStateWorkChain` sub process failed.')

    def _get_protocol(self):
        """Load and read protocol from faml file to a verbose dict"""
        with open(
                str(
                    pathlib.Path(__file__).resolve().parents[0] /
                    'protocol.yml')) as handle:
            self._protocol = yaml.safe_load(handle)  # pylint: disable=attribute-defined-outside-init

            return self._protocol

    def setup(self):
        """Input validation"""
        # pylint: disable=invalid-name, attribute-defined-outside-init

        # parse pseudo and output its header information
        upf_info = helper_parse_upf(self.inputs.pseudo)
        self.ctx.element = orm.Str(upf_info['element'])
        self.out('output_pseudo_header', orm.Dict(dict=upf_info).store())

        # Read from protocol if parameters not set from inputs
        protocol_name = self.inputs.protocol.value
        protocol = self._get_protocol()[protocol_name]
        protocol = protocol['delta_factor']
        self._DEGAUSS = protocol['degauss']
        self._OCCUPATIONS = protocol['occupations']
        self._SMEARING = protocol['smearing']
        self._CONV_THR = protocol['electron_conv_thr']

        if 'kpoints_distance' in self.inputs.parameters:
            self._KDISTANCE = self.inputs.parameters.kpoints_distance.value
        else:
            self._KDISTANCE = protocol['kpoints_distance']

        if 'ecutwfc' in self.inputs.parameters:
            self._ECUTWFC = self.inputs.parameters.ecutwfc.value
        else:
            self._ECUTWFC = protocol['ecutwfc']

        if 'scale_count' in self.inputs.parameters:
            self._SCALE_COUNT = self.inputs.parameters.scale_count.value
        else:
            self._SCALE_COUNT = protocol['scale_count']

        if 'scale_increment' in self.inputs.parameters:
            self._SCALE_INCREMENT = self.inputs.parameters.scale_increment.value
        else:
            self._SCALE_INCREMENT = protocol['scale_increment']

        pw_parameters = {
            'SYSTEM': {
                'degauss': self._DEGAUSS,
                'occupations': self._OCCUPATIONS,
                'smearing': self._SMEARING,
            },
            'ELECTRONS': {
                'conv_thr': self._CONV_THR,
            },
        }

        if 'pw' in self.inputs.parameters:
            pw_parameters = update_dict(pw_parameters,
                                        self.inputs.parameters.pw.get_dict())

        parameters = {
            'SYSTEM': {
                'ecutwfc': self._ECUTWFC,
            },
        }
        # set the ecutrho according to the type of pseudopotential
        # dual 4 for NC and 8 for all other type of PP.
        if 'ecutrho' in self.inputs.parameters:
            parameters['SYSTEM']['ecutrho'] = self.inputs.parameters.ecutrho
        else:
            upf_header = helper_parse_upf(self.inputs.pseudo)
            if upf_header['pseudo_type'] in ['NC', 'SL']:
                dual = 4.0
            else:
                dual = 8.0
            parameters['SYSTEM']['ecutrho'] = self._ECUTWFC * dual

        pw_parameters = update_dict(pw_parameters, parameters)

        self.ctx.pw_parameters = pw_parameters

        self.ctx.kpoints_distance = self._KDISTANCE

    def validate_structure_and_pseudo(self):
        """validate structure"""
        res = get_pw_inputs_from_pseudo(pseudo=self.inputs.pseudo,
                                        primitive_cell=False)

        structure = res['structure']
        base_pw_parameters = res['base_pw_parameters']
        self.ctx.pw_parameters = orm.Dict(
            dict=update_dict(self.ctx.pw_parameters, base_pw_parameters))
        self.ctx.pseudos = res['pseudos']

        if self.ctx.element.value in RARE_EARTH_ELEMENTS:
            extra_parameters = {
                'SYSTEM': {
                    'nspin': 2,
                    'starting_magnetization': {
                        self.ctx.element.value: 0.2,
                        'N': 0.0,
                    },
                },
            }
            self.ctx.pw_parameters = orm.Dict(dict=update_dict(
                self.ctx.pw_parameters.get_dict(), extra_parameters))

        if 'structure' not in self.inputs:
            if self.ctx.element.value not in MAGNETIC_ELEMENTS:
                self.ctx.structure = structure
            else:
                # Mn (antiferrimagnetic), O and Cr (antiferromagnetic), Fe, Co, and Ni (ferromagnetic).
                res = helper_get_magnetic_inputs(structure)
                self.ctx.structure = res['structure']
                parameters = res['parameters']
                self.ctx.pw_parameters = orm.Dict(dict=update_dict(
                    parameters.get_dict(), self.ctx.pw_parameters.get_dict()))

                # setting pseudos
                pseudos = {}
                pseudo = self.inputs.pseudo
                for kind_name in self.ctx.structure.get_kind_names():
                    pseudos[kind_name] = pseudo
                self.ctx.pseudos = pseudos

        else:
            self.ctx.structure = self.inputs.structure

    def run_eos(self):
        """run eos workchain"""
        inputs = AttributeDict({
            'structure':
            self.ctx.structure,
            'kpoints_distance':
            orm.Float(self._KDISTANCE),
            'scale_count':
            orm.Int(self._SCALE_COUNT),
            'scale_increment':
            orm.Float(self._SCALE_INCREMENT),
            'metadata': {
                'call_link_label': 'eos'
            },
            'scf': {
                'pw': {
                    'code': self.inputs.code,
                    'pseudos': self.ctx.pseudos,
                    'parameters': self.ctx.pw_parameters,
                    'metadata': {},
                },
            }
        })

        if 'options' in self.inputs:
            inputs.options = self.inputs.options
        else:
            from aiida_quantumespresso.utils.resources import get_default_options

            inputs.options = orm.Dict(dict=get_default_options(
                max_wallclock_seconds=self._MAX_WALLCLOCK_SECONDS,
                with_mpi=True))

        self.report(f'options is {inputs.options.attributes}')
        running = self.submit(EquationOfStateWorkChain, **inputs)

        self.report(f'launching EquationOfStateWorkChain<{running.pk}>')

        return ToContext(workchain_eos=running)

    def inspect_eos(self):
        """Inspect the results of EquationOfStateWorkChain
        and run the Birch-Murnaghan fit"""
        workchain = self.ctx.workchain_eos

        if not workchain.is_finished_ok:
            self.report(
                f'EquationOfStateWorkChain failed with exit status {workchain.exit_status}'
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_EOS

        self.out('output_eos_parameters', workchain.outputs.output_parameters)
        self.ctx.birch_murnaghan_fit_result = workchain.outputs.output_parameters[
            'birch_murnaghan_fit']
        # TODO report result and output it

    def run_delta_calc(self):
        """calculate the delta factor"""
        res = self.ctx.birch_murnaghan_fit_result

        inputs = {
            'element': self.ctx.element,
            'V0': orm.Float(res['V0']),
            'B0': orm.Float(res['B0']),
            'B1': orm.Float(res['B1']),
        }
        self.ctx.output_parameters = calculate_delta(**inputs)

        self.report(
            f'Birch-Murnaghan fit results are {self.ctx.birch_murnaghan_fit_result}'
        )
        self.out('output_birch_murnaghan_fit',
                 orm.Dict(dict=self.ctx.birch_murnaghan_fit_result).store())

    def results(self):
        """Attach the output parameters to the outputs."""
        self.out('output_parameters', self.ctx.output_parameters)

    def on_terminated(self):
        """Clean the working directories of all child calculations if `clean_workdir=True` in the inputs."""
        super().on_terminated()

        if self.inputs.clean_workdir.value is False:
            self.report('remote folders will not be cleaned')
            return

        cleaned_calcs = []

        for called_descendant in self.node.called_descendants:
            if isinstance(called_descendant, orm.CalcJobNode):
                try:
                    called_descendant.outputs.remote_folder._clean()  # pylint: disable=protected-access
                    cleaned_calcs.append(called_descendant.pk)
                except (IOError, OSError, KeyError):
                    pass

        if cleaned_calcs:
            self.report(
                f"cleaned remote folders of calculations: {' '.join(map(str, cleaned_calcs))}"
            )
