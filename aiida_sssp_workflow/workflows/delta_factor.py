# -*- coding: utf-8 -*-
"""Workchain to calculate delta factor of specific psp"""
import yaml
import importlib_resources

from aiida import orm
from aiida.engine import WorkChain, ToContext, if_
from aiida.plugins import DataFactory

from aiida_sssp_workflow.utils import update_dict, \
    MAGNETIC_ELEMENTS, \
    RARE_EARTH_ELEMENTS, \
    helper_parse_upf, get_standard_cif_filename_from_element
from aiida_sssp_workflow.calculations.calculate_delta import calculate_delta
from aiida_sssp_workflow.workflows._eos import _EquationOfStateWorkChain

UpfData = DataFactory('pseudo.upf')


def helper_get_magnetic_inputs(structure: orm.StructureData):
    """
    To set initial magnet to the magnetic system, need to set magnetic order to
    every magnetic element site, with certain pw starting_mainetization parameters.
    """
    MAG_INIT_Mn = {'Mn1': 0.5, 'Mn2': -0.3, 'Mn3': 0.5, 'Mn4': -0.3}  # pylint: disable=invalid-name
    MAG_INIT_O = {'O1': 0.5, 'O2': 0.5, 'O3': -0.5, 'O4': -0.5}  # pylint: disable=invalid-name
    MAG_INIT_Cr = {'Cr1': 0.5, 'Cr2': -0.5}  # pylint: disable=invalid-name

    mag_structure = orm.StructureData(cell=structure.cell, pbc=structure.pbc)
    kind_name = structure.get_kind_names()[0]

    # ferromagnetic
    if kind_name in ['Fe', 'Co', 'Ni']:
        for i, site in enumerate(structure.sites):
            mag_structure.append_atom(position=site.position,
                                      symbols=kind_name)

        parameters = {
            'SYSTEM': {
                'nspin': 2,
                'starting_magnetization': {
                    kind_name: 0.2
                },
            },
        }

    #
    if kind_name in ['Mn', 'O', 'Cr']:
        for i, site in enumerate(structure.sites):
            mag_structure.append_atom(position=site.position,
                                      symbols=kind_name,
                                      name=f'{kind_name}{i+1}')

        if kind_name == 'Mn':
            parameters = {
                'SYSTEM': {
                    'nspin': 2,
                    'starting_magnetization': MAG_INIT_Mn,
                },
            }

        if kind_name == 'O':
            parameters = {
                'SYSTEM': {
                    'nspin': 2,
                    'starting_magnetization': MAG_INIT_O,
                },
            }

        if kind_name == 'Cr':
            parameters = {
                'SYSTEM': {
                    'nspin': 2,
                    'starting_magnetization': MAG_INIT_Cr,
                },
            }

    return mag_structure, parameters


class DeltaFactorWorkChain(WorkChain):
    """Workchain to calculate delta factor of specific pseudopotential"""
    # pylint: disable=too-many-instance-attributes

    _MAX_WALLCLOCK_SECONDS = 1800 * 3

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        # yapf: disable
        super().define(spec)
        spec.input('code', valid_type=orm.Code,
                    help='The `pw.x` code use for the `PwCalculation`.')
        spec.input('pseudo', valid_type=UpfData, required=True,
                    help='Pseudopotential to be verified')
        spec.input('protocol', valid_type=orm.Str, default=lambda: orm.Str('efficiency'),
                    help='The protocol to use for the workchain.')
        spec.input('options', valid_type=orm.Dict, required=False,
                    help='Optional `options` to use for the `PwCalculations`.')
        spec.input('parallelization', valid_type=orm.Dict, required=False,
                    help='Parallelization options for the `PwCalculations`.')
        spec.input('clean_workdir', valid_type=orm.Bool, default=lambda: orm.Bool(False),
                    help='If `True`, work directories of all called calculation will be cleaned at the end of execution.')
        spec.outline(
            cls.init_setup,
            if_(cls.is_rare_earth_element)(
                cls.extra_setup_for_rare_earth_element,
            ),
            if_(cls.is_magnetic_element)(
                cls.extra_setup_for_magnetic_element,
            ),
            cls.setup_pw_parameters_from_protocol,
            cls.setup_pw_resource_options,
            cls.run_eos,
            cls.inspect_eos,
            cls.calculate_delta_calc,
        )
        spec.expose_outputs(_EquationOfStateWorkChain,
                    namespace_options={'help': 'volume_energy and birch_murnaghan_fit result from EOS.'})
        spec.output('output_delta_factor', valid_type=orm.Dict, required=True,
                    help='The delta factor of the pseudopotential.')
        spec.exit_code(201, 'ERROR_SUB_PROCESS_FAILED_EOS',
                    message=f'The {_EquationOfStateWorkChain.__name__} sub process failed.')
        # yapf: enable

    def _get_protocol(self):
        """Load and read protocol from faml file to a verbose dict"""
        import_path = importlib_resources.path('aiida_sssp_workflow',
                                               'sssp_protocol.yml')
        with import_path as pp_path, open(pp_path, 'rb') as handle:
            self._protocol = yaml.safe_load(handle)  # pylint: disable=attribute-defined-outside-init

            return self._protocol

    def init_setup(self):
        """
        This step contains all preparation before actaul setup, e.g. set
        the context of element, base_structure, base pw_parameters and pseudos.
        """
        # parse pseudo and output its header information
        element = self.inputs.pseudo.element
        self.ctx.element = element

        self.ctx.pw_parameters = {}
        self.ctx.pseudos = {element: self.inputs.pseudo}

        # Structures for delta factor calculation as provided in
        # http:// molmod.ugent.be/deltacodesdft/
        # Exception for lanthanides use nitride structures from
        # https://doi.org/10.1016/j.commatsci.2014.07.030
        cif_file = get_standard_cif_filename_from_element(element)
        self.ctx.structure = orm.CifData.get_or_create(
            cif_file)[0].get_structure(primitive_cell=False)

    def is_rare_earth_element(self):
        """Check if the element is rare earth"""
        return self.ctx.element in RARE_EARTH_ELEMENTS

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

        extra_parameters = {
            'SYSTEM': {
                'nbnd': int(nbands * nbands_factor),
                'nspin': 2,
                'starting_magnetization': {
                    self.ctx.element: 0.2,
                    'N': 0.0,
                },
            },
        }
        self.ctx.pw_parameters = update_dict(self.ctx.pw_parameters,
                                             extra_parameters)

    def is_magnetic_element(self):
        """Check if the element is magnetic"""
        return self.ctx.element in MAGNETIC_ELEMENTS

    def extra_setup_for_magnetic_element(self):
        """Extra setup for magnetic element"""
        # Mn (antiferrimagnetic), O and Cr (antiferromagnetic), Fe, Co, and Ni (ferromagnetic).
        self.ctx.structure, extra_parameters = helper_get_magnetic_inputs(
            self.ctx.structure)
        self.ctx.pw_parameters = update_dict(self.ctx.pw_parameters,
                                             extra_parameters)

        # setting pseudos
        pseudos = {}
        pseudo = self.inputs.pseudo
        for kind_name in self.ctx.structure.get_kind_names():
            pseudos[kind_name] = pseudo
        self.ctx.pseudos = pseudos

    def setup_pw_parameters_from_protocol(self):
        """Input validation"""
        # pylint: disable=invalid-name, attribute-defined-outside-init

        # Read from protocol if parameters not set from inputs
        protocol_name = self.inputs.protocol.value
        protocol = self._get_protocol()[protocol_name]
        protocol = protocol['delta_factor']
        self._DEGAUSS = protocol['degauss']
        self._OCCUPATIONS = protocol['occupations']
        self._SMEARING = protocol['smearing']
        self._CONV_THR = protocol['electron_conv_thr']

        self._KDISTANCE = protocol['kpoints_distance']
        self._ECUTWFC = protocol['ecutwfc']
        self._SCALE_COUNT = protocol['scale_count']
        self._SCALE_INCREMENT = protocol['scale_increment']

        parameters = {
            'SYSTEM': {
                'degauss': self._DEGAUSS,
                'occupations': self._OCCUPATIONS,
                'smearing': self._SMEARING,
                'ecutwfc': self._ECUTWFC,
            },
            'ELECTRONS': {
                'conv_thr': self._CONV_THR,
            },
        }

        # set the ecutrho according to the type of pseudopotential
        # dual 4 for NC and 8 for all other type of PP.
        upf_header = helper_parse_upf(self.inputs.pseudo)
        if upf_header['pseudo_type'] in ['NC', 'SL']:
            dual = 4.0
        else:
            dual = 8.0
        parameters['SYSTEM']['ecutrho'] = self._ECUTWFC * dual

        self.ctx.pw_parameters = update_dict(self.ctx.pw_parameters,
                                             parameters)
        self.ctx.kpoints_distance = self._KDISTANCE

        self.report(
            f'The pw parameters for EOS step is: {self.ctx.pw_parameters}')

    def setup_pw_resource_options(self):
        """
        setup resource options and parallelization for `PwCalculation` from inputs
        """
        if 'options' in self.inputs:
            self.ctx.options = self.inputs.options.get_dict()
        else:
            from aiida_sssp_workflow.utils import get_default_options

            self.ctx.options = get_default_options(
                max_wallclock_seconds=self._MAX_WALLCLOCK_SECONDS,
                with_mpi=True)

        if 'parallelization' in self.inputs:
            self.ctx.parallelization = self.inputs.parallelization.get_dict()
        else:
            self.ctx.parallelization = {}

        self.report(f'resource options set to {self.ctx.options}')
        self.report(
            f'parallelization options set to {self.ctx.parallelization}')

    def run_eos(self):
        """run eos workchain"""
        # yapf: disable
        self.report(f'{self.ctx.pw_parameters}')
        inputs = {
            'structure': self.ctx.structure,
            'kpoints_distance': orm.Float(self._KDISTANCE),
            'scale_count': orm.Int(self._SCALE_COUNT),
            'scale_increment': orm.Float(self._SCALE_INCREMENT),
            'metadata': {
                'call_link_label': 'eos'
            },
            'scf': {
                'pw': {
                    'code': self.inputs.code,
                    'pseudos': self.ctx.pseudos,
                    'parameters': orm.Dict(dict=self.ctx.pw_parameters),
                    'metadata': {
                        'options': self.ctx.options
                    },
                    'parallelization': orm.Dict(dict=self.ctx.parallelization),
                },
            }
        }
        # yapf: enable

        running = self.submit(_EquationOfStateWorkChain, **inputs)
        self.report(f'launching _EquationOfStateWorkChain<{running.pk}>')

        return ToContext(workchain_eos=running)

    def inspect_eos(self):
        """Inspect the results of _EquationOfStateWorkChain"""
        workchain = self.ctx.workchain_eos

        if not workchain.is_finished_ok:
            self.report(
                f'_EquationOfStateWorkChain failed with exit status {workchain.exit_status}'
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_EOS

        self.out_many(
            self.exposed_outputs(
                workchain,
                _EquationOfStateWorkChain,
            ))

        self.ctx.output_birch_murnaghan_fit = workchain.outputs.output_birch_murnaghan_fit

    def calculate_delta_calc(self):
        """calculate the delta factor"""
        output_bmf = self.ctx.output_birch_murnaghan_fit

        inputs = {
            'element': orm.Str(self.ctx.element),
            'V0': orm.Float(output_bmf['volume0']),
            'B0': orm.Float(output_bmf['bulk_modulus0']),
            'B1': orm.Float(output_bmf['bulk_deriv0']),
        }
        output_delta_factor = calculate_delta(**inputs)
        self.out('output_delta_factor', output_delta_factor)

        self.report(
            f'The birch murnaghan fitting results are: {output_delta_factor.get_dict()}'
        )

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
