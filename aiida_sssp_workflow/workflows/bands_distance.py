# -*- coding: utf-8 -*-
"""
Bands distance of many input pseudos
"""
import yaml
import importlib_resources

from aiida import orm
from aiida.engine import WorkChain, if_, append_
from aiida.plugins import DataFactory

from aiida_sssp_workflow.utils import NONMETAL_ELEMENTS, update_dict, \
    RARE_EARTH_ELEMENTS, \
    get_standard_cif_filename_from_element
from aiida_sssp_workflow.workflows.evaluate._bands import BandsWorkChain
from aiida_sssp_workflow.calculations.calculate_bands_distance import calculate_bands_distance

UpfData = DataFactory('pseudo.upf')


def validate_input_pseudos(d_pseudos, _):
    """Validate that all input pseudos map to same element"""
    element = set(pseudo.element for pseudo in d_pseudos.values())

    if len(element) > 1:
        return f'The pseudos corespond to different elements {element}.'


class BandsDistanceWorkChain(WorkChain):
    """WorkChain to do bands comparision of different pseudos"""

    _MAX_WALLCLOCK_SECONDS = 1800 * 3

    _LARGE_DUAL_ELEMENTS = ['Fe', 'Hf']

    _INIT_NBANDS_FACTOR = 3.0
    _RY_TO_EV = 13.6056980659

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        super().define(spec)
        # yapf: disable
        spec.input('code', valid_type=orm.Code,
                    help='The `pw.x` code use for the `PwCalculation`.')
        spec.input_namespace('input_pseudos', valid_type=UpfData, dynamic=True,
                    validator=validate_input_pseudos,
                    help='A mapping of `UpfData` node to be verified onto file name:upf.')
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
                cls.extra_setup_for_rare_earth_element, ),
            cls.setup_code_parameters_from_protocol,
            cls.setup_code_resource_options,
            cls.run_bands_evaluation,
            cls.calculate_bands_distance,
        )

        spec.output('output_bands_distance', valid_type=orm.Dict, required=True,
            help='the distance of each pair of pseudopotentials.')

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
        self.ctx.pw_parameters = {}
        self.ctx.extra_parameters = {}

        # First idx since already being validate to make sure all mapping to same element
        element = [pseudo.element for pseudo in self.inputs.input_pseudos.values()][0]
        self.ctx.element = element

        # Structures for convergence verification are all primitive structures
        # the original conventional structure comes from the same CIF files of
        # http:// molmod.ugent.be/deltacodesdft/
        # EXCEPT that for the element fluorine the `SiF4.cif` used for convergence
        # reason. But we do the structure setup for SiF4 in the following step:
        # `cls.extra_setup_for_fluorine_element`
        cif_file = get_standard_cif_filename_from_element(element)
        self.ctx.structure = orm.CifData.get_or_create(
            cif_file)[0].get_structure(primitive_cell=True)

        # extra setting for bands convergence
        self.ctx.is_metal = element not in NONMETAL_ELEMENTS

    def is_rare_earth_element(self):
        """Check if the element is rare earth"""
        return self.ctx.element in RARE_EARTH_ELEMENTS

    def extra_setup_for_rare_earth_element(self):
        """Extra setup for rare earth element"""
        import_path = importlib_resources.path('aiida_sssp_workflow.REF.UPFs',
                                               'N.pbe-n-radius_5.UPF')
        with import_path as pp_path, open(pp_path, 'rb') as stream:
            upf_nitrogen = UpfData(stream)
            self.ctx.pseudo_N = upf_nitrogen

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

    def setup_code_parameters_from_protocol(self):
        """Input validation"""
        # pylint: disable=invalid-name, attribute-defined-outside-init

        # Read from protocol if parameters not set from inputs
        protocol_name = self.inputs.protocol.value
        protocol = self._get_protocol()[protocol_name]
        protocol = protocol['bands_distance']
        self._DEGAUSS = protocol['degauss']
        self._OCCUPATIONS = protocol['occupations']
        self._SMEARING = protocol['smearing']
        self._CONV_THR = protocol['electron_conv_thr']
        self._KDISTANCE = protocol['kpoints_distance']

        self.ctx.ecutwfc = protocol['ecutwfc']
        self.ctx.kpoints_distance = self._KDISTANCE
        self.ctx.degauss = self._DEGAUSS

        # set large ecutrho for specific elements
        if self.ctx.element in self._LARGE_DUAL_ELEMENTS:
            self.ctx.ecutrho = protocol['ecutrho'] * 2
        else:
            self.ctx.ecutrho = protocol['ecutrho']

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

        self.report(
            f'The pw parameters for convergence is: {self.ctx.pw_parameters}'
        )

    def setup_code_resource_options(self):
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

    def _get_inputs(self, element, pseudos):
        """
        get inputs for the bands evaluation with given pseudo
        """
        if element in RARE_EARTH_ELEMENTS:
            pseudos['N'] = self.ctx.pseudo_N

        # yapf: disable
        inputs = {
            'code': self.inputs.code,
            'pseudos': pseudos,
            'structure': self.ctx.structure,
            'pw_base_parameters': orm.Dict(dict=self.ctx.pw_parameters),
            'ecutwfc': orm.Float(self.ctx.ecutwfc),
            'ecutrho': orm.Float(self.ctx.ecutrho),
            'kpoints_distance': orm.Float(self.ctx.kpoints_distance),
            'init_nbands_factor': orm.Float(self._INIT_NBANDS_FACTOR),
            'options': orm.Dict(dict=self.ctx.options),
            'parallelization': orm.Dict(dict=self.ctx.parallelization),
            'clean_workdir': orm.Bool(False),   # will leave the workdir clean to outer most wf
        }
        # yapf: enable

        return inputs

    def run_bands_evaluation(self):
        """run bands evaluation of pp in inputs list"""
        for name, pseudo in self.inputs.input_pseudos.items():
            pseudos = {self.ctx.element: pseudo}
            inputs = self._get_inputs(self.ctx.element, pseudos)

            running = self.submit(BandsWorkChain, **inputs)

            # set the name to indicate the pp
            running.set_extra('name', name)
            self.report(
                f'launching pseudo >{name}<: {pseudo} BandsWorkChain<{running.pk}>'
            )

            self.to_context(children=append_(running))

    def calculate_bands_distance(self):
        """calculate bands distance of every pair of pp"""
        from itertools import combinations

        children = self.ctx.children
        success_children = [
            child for child in children if child.is_finished_ok
        ]

        d_output_parameters = {}

        for idx, (wca, wcb) in enumerate(combinations(success_children, 2)):
            d_ab = {
                'workchain_name_a': wca.extras['name'],
                'workchain_name_b': wcb.extras['name']
            }

            wca_bands_parameters = wca.outputs.output_bands_parameters
            wcb_bands_parameters = wcb.outputs.output_bands_parameters
            wca_bands_structure = wca.outputs.output_bands_structure
            wcb_bands_structure = wcb.outputs.output_bands_structure

            res = calculate_bands_distance(
                bands_structure_a=wca_bands_structure,
                bands_parameters_a=wca_bands_parameters,
                bands_structure_b=wcb_bands_structure,
                bands_parameters_b=wcb_bands_parameters,
                smearing=orm.Float(self.ctx.degauss * self._RY_TO_EV),
                is_metal=orm.Bool(self.ctx.is_metal))

            d_ab.update(res.get_dict())
            d_output_parameters[f'dist_{idx}'] = d_ab

        self.out('output_bands_distance',
                 orm.Dict(dict=d_output_parameters).store())
