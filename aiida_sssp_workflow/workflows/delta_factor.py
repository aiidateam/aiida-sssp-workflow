# -*- coding: utf-8 -*-
"""Workchain to calculate delta factor of specific psp"""
import importlib_resources
import collections.abc

from aiida import orm
from aiida.common import AttributeDict
from aiida.engine import WorkChain, ToContext, calcfunction
from aiida.plugins import WorkflowFactory, CalculationFactory

birch_murnaghan_fit = CalculationFactory('sssp_workflow.birch_murnaghan_fit')
calculate_delta = CalculationFactory('sssp_workflow.calculate_delta')
EquationOfStateWorkChain = WorkflowFactory('sssp_workflow.eos')

MAGNETIC_ELEMENTS = ['Mn', 'O', 'Cr', 'Fe', 'Co', 'Ni']

@calcfunction
def helper_parse_upf(upf):
    return orm.Str(upf.element)


@calcfunction
def helper_create_standard_cif_from_element(element: orm.Str) -> orm.CifData:
    filename = get_standard_cif_filename_from_element(element.value)
    cif_data, created = orm.CifData.get_or_create(filename)
    assert created is True

    return cif_data

@calcfunction
def helper_get_magnetic_inputs(structure: orm.StructureData):
    """
    docstring
    """
    MAG_INIT_Mn = {"Mn1":0.5,"Mn2":-0.3,"Mn3":0.5,"Mn4":-0.3}
    MAG_INIT_O = {"O1":0.5,"O2":0.5,"O3":-0.5,"O4":-0.5}
    MAG_INIT_Cr = {"Cr1":0.5,"Cr2":-0.5}

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
                'starting_magnetization': {kind_name: 0.2},
            },
        })

    #
    if kind_name in ['Mn', 'O', 'Cr']:
        for i, site in enumerate(structure.sites):
            mag_structure.append_atom(position=site.position,symbols=kind_name,name=f'{kind_name}{i+1}')

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



def get_standard_cif_filename_from_element(element: str) -> str:
    if element in RARE_EARTH_ELEMENTS:
        fpath = importlib_resources.path('aiida_sssp_workflow.REF.CIFs_REN',
                                         f'{element}N.cif')
    else:
        fpath = importlib_resources.path('aiida_sssp_workflow.REF.CIFs',
                                         f'{element}.cif')
    with fpath as path:
        filename = str(path)

    return filename


def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


PW_PARAS = lambda: orm.Dict(
    dict={
        'SYSTEM': {
            'ecutrho': 1600,
            'ecutwfc': 200,
        },
    })


RARE_EARTH_ELEMENTS = [
    'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er',
    'Tm', 'Yb', 'Lu'
]  # move to utils


class DeltaFactorWorkChain(WorkChain):
    """Workchain to calculate delta factor of specific pseudopotential"""
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
        spec.output('eos_parameters',
                    valid_type=orm.Dict,
                    required=True,
                    help='The eos outputs.')
        spec.output('output_parameters',
                    valid_type=orm.Dict,
                    required=True,
                    help='The delta factor of the pseudopotential.')
        spec.output('element',
                    valid_type=orm.Str,
                    required=True,
                    help='The element of the pseudopotential.')
        spec.output(
            'eos_initial_cif',
            valid_type=orm.CifData,
            required=False,
            help='The standard cif file provided in sssp_workflow package.')
        spec.output(
            'eos_initial_structure',
            valid_type=orm.StructureData,
            help='The initial input structure used for calculate delta factor.'
        )
        # TODO delta prime out
        spec.exit_code(
            201,
            'ERROR_SUB_PROCESS_FAILED_EOS',
            message='The `EquationOfStateWorkChain` sub process failed.')

    def setup(self):
        """Input validation"""
        # TODO set ecutwfc and ecutrho according to certain protocol


        pw_parameters = {
            'SYSTEM': {
                'degauss': 0.02,
                'occupations': 'smearing',
                'smearing': 'marzari-vanderbilt',
            },
            'ELECTRONS': {
                'conv_thr': 1e-10,
            },
        }

        self.ctx.pw_parameters = orm.Dict(dict=update(pw_parameters, self.inputs.parameters.pw.get_dict()))
        self.ctx.kpoints_distance = self.inputs.parameters.kpoints_distance

    def validate_structure_and_pseudo(self):
        """validate structure"""
        from aiida.common.files import md5_file

        self.ctx.element = helper_parse_upf(self.inputs.pseudo)

        pseudos = {self.ctx.element.value: self.inputs.pseudo}
        if self.ctx.element in RARE_EARTH_ELEMENTS:
            # If rare-earth add psp of N
            fpath = importlib_resources.path('aiida_sssp_workflow.REF.UPFs',
                                             'N.UPF')
            with fpath as path:
                filename = str(path)
                upf_nitrogen = orm.UpfData.get_or_create(filename)[0]
                pseudos['N'] = upf_nitrogen
            parameters = {
                'SYSTEM': {
                    'nspin': 2,
                },
            }
            self.ctx.pw_parameters = orm.Dict(dict=update(self.ctx.pw_parameters.get_dict(), parameters))
        self.ctx.pseudos = pseudos
        self.report(f'pseudos is {pseudos}')

        if not 'structure' in self.inputs:
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

            self.out('eos_initial_cif', cif_data)

            if not self.ctx.element.value in MAGNETIC_ELEMENTS:
                self.ctx.structure = cif_data.get_structure()
            else:
                # Mn (antiferrimagnetic), O and Cr (antiferromagnetic), Fe, Co, and Ni (ferromagnetic).
                structure = cif_data.get_structure()
                res = helper_get_magnetic_inputs(structure, self.inputs.pseudo)
                self.ctx.structure = res['structure']
                parameters = res['parameters']
                self.ctx.pw_parameters = orm.Dict(dict=update(parameters.get_dict(), self.ctx.pw_parameters.get_dict()))

                # setting pseudos
                pseudos = {}
                pseudo = self.inputs.pseudo
                for kind_name in self.ctx.structure.get_kind_names():
                    pseudos[kind_name] = pseudo
                self.ctx.pseudos = pseudos

        else:
            self.ctx.structure = self.inputs.structure

        self.out('eos_initial_structure', self.ctx.structure)



    def run_eos(self):
        """run eos workchain"""
        inputs = AttributeDict({
            'structure': self.ctx.structure,
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
                'kpoints_distance': self.ctx.kpoints_distance,
            }
        })

        if 'options' in self.inputs:
            inputs.scf.pw.metadata.options = self.inputs.options.get_dict()
        else:
            from aiida_quantumespresso.utils.resources import get_default_options

            inputs.scf.pw.metadata.options = get_default_options(max_wallclock_seconds=3600, with_mpi=True)

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

        volume_energy = workchain.outputs.output_parameters  # This keep the provenance
        self.out('eos_parameters', workchain.outputs.output_parameters)
        self.ctx.birch_murnaghan_fit_result = birch_murnaghan_fit(
            volume_energy)
        # TODO report result and output it

    def run_delta_calc(self):
        """calculate the delta factor"""
        res = self.ctx.birch_murnaghan_fit_result
        inputs = {
            'element': self.ctx.element,
            'v0': res['volume0'],
            'b0': res['bulk_modulus0'],
            'bp': res['bulk_deriv0'],
        }
        self.ctx.output_parameters = calculate_delta(**inputs)
        # TODO report

    def results(self):
        """Attach the output parameters to the outputs."""
        self.out('output_parameters', self.ctx.output_parameters)
        self.out('element', self.ctx.element)
        # TODO output the parameters used for eos and pw.

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
