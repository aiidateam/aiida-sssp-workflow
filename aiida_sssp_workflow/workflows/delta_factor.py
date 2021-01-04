# -*- coding: utf-8 -*-
"""Workchain to calculate delta factor of specific psp"""
from aiida import orm
from aiida.common import AttributeDict
from aiida.engine import WorkChain, ToContext, calcfunction
from aiida.plugins import WorkflowFactory, CalculationFactory

# TODO concise import
from aiida_sssp_workflow.utils import update_dict, \
    MAGNETIC_ELEMENTS, \
    RARE_EARTH_ELEMENTS, \
    helper_parse_upf
from .helper import get_pw_inputs_from_pseudo

calculate_delta = CalculationFactory('sssp_workflow.calculate_delta')
EquationOfStateWorkChain = WorkflowFactory('sssp_workflow.eos')


@calcfunction
def helper_get_magnetic_inputs(structure: orm.StructureData):
    """
    docstring
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
            mag_structure.append_site(site=site)

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


PW_PARAS = lambda: orm.Dict(dict={
    'SYSTEM': {
        'ecutrho': 1600,
        'ecutwfc': 200,
    },
})


class DeltaFactorWorkChain(WorkChain):
    """Workchain to calculate delta factor of specific pseudopotential"""

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

    _MAX_WALLCLOCK_SECONDS = 1800 * 2

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
        spec.input('options',
                   valid_type=orm.Dict,
                   required=False,
                   help='Optional `options` to use for the `PwCalculations`.')
        spec.input_namespace('parameters', help='Para')
        spec.input('parameters.pw',
                   valid_type=orm.Dict,
                   default=PW_PARAS,
                   help='parameters for pwscf.')
        spec.input('parameters.kpoints_distance',
                   valid_type=orm.Float,
                   default=lambda: orm.Float(0.1),
                   help='Global kpoints setting.')
        spec.input('parameters.scale_count',
                   valid_type=orm.Int,
                   default=lambda: orm.Int(7),
                   help='Numbers of scale points in eos step.')
        spec.input('parameters.scale_increment',
                   valid_type=orm.Float,
                   default=lambda: orm.Float(0.02),
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
        spec.output('output_birch_murnaghan_fit',
                    valid_type=orm.Dict,
                    required=True,
                    help='The results V0, B0, B1 of Birch-Murnaghan fit.')
        spec.exit_code(
            201,
            'ERROR_SUB_PROCESS_FAILED_EOS',
            message='The `EquationOfStateWorkChain` sub process failed.')

    def setup(self):
        """Input validation"""
        # TODO set ecutwfc and ecutrho according to certain protocol

        pw_parameters = self._PW_PARAMETERS

        self.ctx.pw_parameters = orm.Dict(dict=update_dict(
            pw_parameters, self.inputs.parameters.pw.get_dict()))
        self.ctx.kpoints_distance = self.inputs.parameters.kpoints_distance

    def validate_structure_and_pseudo(self):
        """validate structure"""
        upf_info = helper_parse_upf(self.inputs.pseudo)
        self.ctx.element = orm.Str(upf_info['element'])

        res = get_pw_inputs_from_pseudo(pseudo=self.inputs.pseudo,
                                        primitive_cell=False)

        structure = res['structure']
        base_pw_parameters = res['base_pw_parameters']
        self.ctx.pw_parameters = orm.Dict(dict=update_dict(
            self.ctx.pw_parameters.get_dict(), base_pw_parameters))
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
                structure = self.ctx.structure
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
            'structure': self.ctx.structure,
            'kpoints_distance': self.ctx.kpoints_distance,
            'scale_count': self.inputs.parameters.scale_count,
            'scale_increment': self.inputs.parameters.scale_increment,
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
            'v0': orm.Float(res['v0']),
            'b0': orm.Float(res['b0']),
            'bp': orm.Float(res['bp']),
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
