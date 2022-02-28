# -*- coding: utf-8 -*-
"""Workchain to calculate delta factor of specific psp"""
import yaml
import importlib_resources

from aiida import orm
from aiida.engine import WorkChain, ToContext, if_, append_
from aiida.plugins import DataFactory

from aiida_sssp_workflow.utils import update_dict, \
    MAGNETIC_ELEMENTS, \
    RARE_EARTH_ELEMENTS, \
    get_standard_cif_filename_from_element, \
    get_standard_cif_filename_dict_from_element, \
    helper_get_magnetic_inputs
from aiida_sssp_workflow.calculations.calculate_delta import delta_analyze
from aiida_sssp_workflow.workflows._eos import _EquationOfStateWorkChain
from pseudo_parser.upf_parser import parse_pseudo_type, parse_element

UpfData = DataFactory('pseudo.upf')


class DeltaFactorWorkChain(WorkChain):
    """Workchain to calculate delta factor of specific pseudopotential"""
    # pylint: disable=too-many-instance-attributes

    _MAX_WALLCLOCK_SECONDS = 1800 * 3
    _OXIDE_STRUCTURES = ['XO', 'XO2','XO3', 'X2O', 'X2O3', 'X2O5']

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        # yapf: disable
        super().define(spec)
        spec.input('code', valid_type=orm.Code,
                    help='The `pw.x` code use for the `PwCalculation`.')
        spec.input('pseudo', valid_type=UpfData, required=True,
                    help='Pseudopotential to be verified')
        spec.input('protocol_calculation', valid_type=orm.Str, default=lambda: orm.Str('theos'),
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
            # To compatible with common-wf, the magnism is not on. 
            # if_(cls.is_magnetic_element)(
            #     cls.extra_setup_for_magnetic_element,
            # ),
            cls.setup_pw_parameters_from_protocol,
            cls.setup_pw_resource_options,
            cls.run_eos,
            cls.inspect_eos,
            cls.run_delta_analyze,
        )
        spec.expose_outputs(_EquationOfStateWorkChain, namespace='X',
                    namespace_options={'help': 'volume_energy and birch_murnaghan_fit result from X EOS.'})
        spec.expose_outputs(_EquationOfStateWorkChain, namespace='XO',
                    namespace_options={'help': 'volume_energy and birch_murnaghan_fit result from XO EOS.'})
        spec.expose_outputs(_EquationOfStateWorkChain, namespace='XO2',
                    namespace_options={'help': 'volume_energy and birch_murnaghan_fit result from XO2 EOS.'})
        spec.expose_outputs(_EquationOfStateWorkChain, namespace='XO3',
                    namespace_options={'help': 'volume_energy and birch_murnaghan_fit result from XO3 EOS.'})
        spec.expose_outputs(_EquationOfStateWorkChain, namespace='X2O',
                    namespace_options={'help': 'volume_energy and birch_murnaghan_fit result from X2O EOS.'})
        spec.expose_outputs(_EquationOfStateWorkChain, namespace='X2O3',
                    namespace_options={'help': 'volume_energy and birch_murnaghan_fit result from X2O3 EOS.'})
        spec.expose_outputs(_EquationOfStateWorkChain, namespace='X2O5',
                    namespace_options={'help': 'volume_energy and birch_murnaghan_fit result from X2O5 EOS.'})
        spec.output_namespace('output_delta_analyze', dynamic=True,
                    help='The output of delta factor and other measures to describe the accuracy of EOS compare '
                        ' with the AE equation of state.')
        spec.exit_code(201, 'ERROR_SUB_PROCESS_FAILED_EOS',
                    message=f'The {_EquationOfStateWorkChain.__name__} sub process failed.')
        # yapf: enable

    def _get_protocol(self):
        """Load and read protocol from faml file to a verbose dict"""
        import_path = importlib_resources.path('aiida_sssp_workflow',
                                               'PROTOCOL_CALC.yml')
        with import_path as pp_path, open(pp_path, 'rb') as handle:
            self._protocol = yaml.safe_load(handle)  # pylint: disable=attribute-defined-outside-init

            return self._protocol

    def init_setup(self):
        """
        This step contains all preparation before actaul setup, e.g. set
        the context of element, base_structure, base pw_parameters and pseudos.
        """
        # parse pseudo and output its header information
        content = self.inputs.pseudo.get_content()
        element = parse_element(content)
        pseudo_type = parse_pseudo_type(content)
        self.ctx.element = element
        self.ctx.pseudo_type = pseudo_type
        self.ctx.pseudos_elementary = {element: self.inputs.pseudo}
        
        # Import oxygen pseudopotential file and set the pseudos
        import_path = importlib_resources.path('aiida_sssp_workflow.REF.UPFs',
                                               'O.pbe-n-kjpaw_psl.0.1.UPF')
        with import_path as pp_path, open(pp_path, 'rb') as stream:
            pseudo_O = UpfData(stream)
            
        self.ctx.pseudos_oxide = {
            element: self.inputs.pseudo, 
            'O': pseudo_O,
        }

        self.ctx.pw_parameters = {}

        # Structures for delta factor calculation as provided in
        # http:// molmod.ugent.be/deltacodesdft/
        # Exception for lanthanides use nitride structures from
        # https://doi.org/10.1016/j.commatsci.2014.07.030 and from 
        # common-workflow set from
        # https://github.com/aiidateam/commonwf-oxides-scripts/tree/main/0-preliminary-do-not-run/cifs-set2
        cif_dict = get_standard_cif_filename_dict_from_element(element)
        
        # keys here are: X, (BCC), (FCC), XO, XO2, (XO3), (X2O), (X2O3), (X2O5)
        # parentatheses means not supported yet.
        self.ctx.structures = {}
        for key in cif_dict.keys():
            self.ctx.structures[key] = orm.CifData.get_or_create(
                cif_dict[key], use_first=True)[0].get_structure(primitive_cell=False)

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
                # Althrough magnetic off for mag element
                # still turn on for Lanths for converge reason
                # Furthure investigation needed in future.
                'nspin': 2,
                'starting_magnetization': {
                    self.ctx.element: 0.2,
                    'N': 0.0,
                },
            },
        }
        self.ctx.pw_parameters = update_dict(self.ctx.pw_parameters,
                                             extra_parameters)

    # def is_magnetic_element(self):
    #     """Check if the element is magnetic"""
    #     return self.ctx.element in MAGNETIC_ELEMENTS

    # def extra_setup_for_magnetic_element(self):
    #     """Extra setup for magnetic element"""
    #     # Mn (antiferrimagnetic), O and Cr (antiferromagnetic), Fe, Co, and Ni (ferromagnetic).
    #     self.ctx.structure, extra_parameters = helper_get_magnetic_inputs(
    #         self.ctx.structure)
    #     self.ctx.pw_parameters = update_dict(self.ctx.pw_parameters,
    #                                          extra_parameters)

    #     # setting pseudos
    #     pseudos = {}
    #     pseudo = self.inputs.pseudo
    #     for kind_name in self.ctx.structure.get_kind_names():
    #         pseudos[kind_name] = pseudo
    #     self.ctx.pseudos = pseudos

    def setup_pw_parameters_from_protocol(self):
        """Input validation"""
        # pylint: disable=invalid-name, attribute-defined-outside-init

        # Read from protocol if parameters not set from inputs
        protocol_name = self.inputs.protocol_calculation.value
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

        # # set the ecutrho according to the type of pseudopotential
        # # dual 4 for NC and 8 for all other type of PP.
        # if self.ctx.pseudo_type in ['NC', 'SL']:
        #     dual = 4.0
        # else:
        #     dual = 8.0

        # TBD: Always use dual=8 since pseudo_O here is non-NC
        parameters['SYSTEM']['ecutrho'] = self._ECUTWFC * 8

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
        
    def _get_inputs(self, name, structure):
        if 'O' in name:
            # pseudos for oxides
            pseudos = self.ctx.pseudos_oxide
        else:
            pseudos = self.ctx.pseudos_elementary
        # yapf: disable
        inputs = {
            'structure': structure,
            'kpoints_distance': orm.Float(self._KDISTANCE),
            'scale_count': orm.Int(self._SCALE_COUNT),
            'scale_increment': orm.Float(self._SCALE_INCREMENT),
            'metadata': {
                'call_link_label': f'{name}_EOS'
            },
            'scf': {
                'pw': {
                    'code': self.inputs.code,
                    'pseudos': pseudos,
                    'parameters': orm.Dict(dict=self.ctx.pw_parameters),
                    'metadata': {
                        'options': self.ctx.options
                    },
                    'parallelization': orm.Dict(dict=self.ctx.parallelization),
                },
            }
        }
        # yapf: enable
        
        return inputs

    def run_eos(self):
        """run eos workchain"""
        self.report(f'{self.ctx.pw_parameters}')

        for name, structure in self.ctx.structures.items():
            inputs = self._get_inputs(name, structure)

            future = self.submit(_EquationOfStateWorkChain, **inputs)
            self.report(f'launching _EquationOfStateWorkChain<{future.pk}> for {name} structure.')

            self.to_context(**{f'{name}_eos': future})

    def inspect_eos(self):
        """Inspect the results of _EquationOfStateWorkChain"""
        failed = []
        for key in self.ctx.structures.keys():
            workchain = self.ctx[f'{key}_eos']

            if not workchain.is_finished_ok:
                self.report(
                    f'_EquationOfStateWorkChain of {key} failed with exit status {workchain.exit_status}'
                )
                failed.append(key)

            self.out_many(
                self.exposed_outputs(
                    workchain,
                    _EquationOfStateWorkChain,
                    namespace=key,
                ))
        
        if failed:
            pass
            # TODO ERROR

    def run_delta_analyze(self):
        """calculate the delta factor"""
        
        for structure_name in ['X'] + self._OXIDE_STRUCTURES:
            try:
                output_bmf = self.outputs[structure_name].get('output_birch_murnaghan_fit')
            except KeyError:
                self.report(f'Can not get the key {structure_name} from outputs.')
                continue
            
            if 'O' in structure_name:
                # delta analyze of unitary structures, V0 is per atoms, B1 unit is GPa
                V0 = orm.Float(output_bmf['volume0'])
            else:
                # delta analyze of oxide structures, V0 is whole structure, B1 unit is eV A^3
                V0 = orm.Float(output_bmf['volume0']/output_bmf['num_of_atoms'])
            inputs = {
                'element': orm.Str(self.ctx.element),
                'structure': orm.Str(structure_name),
                'V0': V0,
                'B0': orm.Float(output_bmf['bulk_modulus0']),
                'B1': orm.Float(output_bmf['bulk_deriv0']),
            }

            self.out(f'output_delta_analyze.output_{structure_name}', delta_analyze(**inputs))

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
