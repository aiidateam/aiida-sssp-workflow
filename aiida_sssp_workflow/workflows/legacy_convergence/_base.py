# -*- coding: utf-8 -*-
"""
Base legacy work chain
"""
from abc import ABCMeta, abstractmethod
import importlib_resources
import yaml

from aiida import orm
from aiida.engine import WorkChain, if_
from aiida.plugins import DataFactory

from aiida_sssp_workflow.utils import RARE_EARTH_ELEMENTS, \
    MAGNETIC_ELEMENTS, \
    get_standard_cif_filename_from_element, \
    update_dict, \
    helper_get_magnetic_inputs

UpfData = DataFactory('pseudo.upf')


class BaseLegacyWorkChain(WorkChain):
    """Base legacy workchain"""
    # pylint: disable=too-many-instance-attributes
    __metaclass__ = ABCMeta

    _MAX_WALLCLOCK_SECONDS = 1800 * 3

    # ecutwfc evaluate list, the normal reference 200Ry not included
    # since reference will anyway included at final inspect step
    _ECUTWFC_LIST = [
        30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 90, 100, 120, 150
    ]

    @classmethod
    def define(cls, spec):
        super().define(spec)
        # yapf: disable
        spec.input('pseudo', valid_type=UpfData, required=True,
                    help='Pseudopotential to be verified')
        spec.input('protocol', valid_type=orm.Str, default=lambda: orm.Str('theos'),
                    help='The protocol to use for the workchain.')
        spec.input('options', valid_type=orm.Dict, required=False,
                    help='Optional `options` to use for the `PwCalculations`.')
        spec.input('parallelization', valid_type=orm.Dict, required=False,
                    help='Parallelization options for the `PwCalculations`.')
        spec.input('clean_workdir', valid_type=orm.Bool, default=lambda: orm.Bool(False),
                    help='If `True`, work directories of all called calculation will be cleaned at the end of execution.')

        spec.outline(
            cls.init_setup,
            if_(cls.is_magnetic_element)(
                cls.extra_setup_for_magnetic_element, ),
            if_(cls.is_rare_earth_element)(
                cls.extra_setup_for_rare_earth_element, ),
            if_(cls.is_fluorine_element)(
                cls.extra_setup_for_fluorine_element, ),
            cls.setup_code_parameters_from_protocol,
            cls.setup_code_resource_options,
            cls.run_reference,
            cls.run_samples_fix_dual,
            cls.inspect_fix_dual,
            cls.run_samples_fix_wfc_cutoff,
            cls.inspect_fix_wfc_cutoff,
            cls.final_results,
        )

        spec.output('fix_dual_output_parameters', valid_type=orm.Dict, required=True,
                    help='The output parameters include results of all calculations.')
        spec.output('fix_wfc_cutoff_output_parameters', valid_type=orm.Dict, required=True,
                    help='The output parameters include results of all calculations.')
        spec.output('final_output_parameters', valid_type=orm.Dict, required=True,
                    help='The output parameters of two stage convergence test.')

        spec.exit_code(401, 'ERROR_SUB_PROCESS_FAILED',
            message='The sub process for `{label}` did not finish successfully.')
        # yapy: enable

    def _get_protocol(self):
        """Load and read protocol from faml file to a verbose dict"""
        import_path = importlib_resources.path('aiida_sssp_workflow',
                                               'CALC_PROTOCOL.yml')
        with import_path as pp_path, open(pp_path, 'rb') as handle:
            self._protocol = yaml.safe_load(handle)  # pylint: disable=attribute-defined-outside-init

            return self._protocol

    def init_setup(self):
        """
        This step contains all preparation before actaul setup, e.g. set
        the context of element, base_structure, base pw_parameters and pseudos.
        """
        # parse pseudo and output its header information
        from pseudo_parser.upf_parser import parse_element, parse_pseudo_type

        self.ctx.extra_pw_parameters = {}
        content = self.inputs.pseudo.get_content()
        element = parse_element(content)
        pseudo_type = parse_pseudo_type(content)
        self.ctx.element = element
        self.ctx.pseudo_type = pseudo_type

        # set the ecutrho according to the type of pseudopotential
        # dual 4 for NC and 10 for all other type of PP.
        if self.ctx.pseudo_type in ['NC', 'SL']:
            self.ctx.init_dual = 4.0
            self.ctx.min_dual = 2.0
        else:
            # the initial dual set to 10 to make sure it is enough and converged
            # In the follow up steps will converge on ecutrho
            self.ctx.init_dual = 10.0
            self.ctx.min_dual = 4.0

        # TODO: for extrem high dual elements: O Fe Hf etc.

        self.ctx.pseudos = {element: self.inputs.pseudo}

        # Structures for convergence verification are all primitive structures
        # the original conventional structure comes from the same CIF files of
        # http:// molmod.ugent.be/deltacodesdft/
        # EXCEPT that for the element fluorine the `SiF4.cif` used for convergence
        # reason. But we do the structure setup for SiF4 in the following step:
        # `cls.extra_setup_for_fluorine_element`
        cif_file = get_standard_cif_filename_from_element(element)
        self.ctx.cif = orm.CifData.get_or_create(cif_file, use_first=True)[0]
        self.ctx.structure = self.ctx.cif.get_structure(primitive_cell=True)

    def is_magnetic_element(self):
        """Check if the element is magnetic"""
        return self.ctx.element in MAGNETIC_ELEMENTS

    def extra_setup_for_magnetic_element(self):
        """Extra setup for magnetic element"""
        self.ctx.structure = self.ctx.cif.get_structure(primitive_cell=False)

        self.ctx.structure, self.ctx.magnetic_extra_parameters = helper_get_magnetic_inputs(
            self.ctx.structure)
        self.ctx.extra_pw_parameters = update_dict(self.ctx.extra_pw_parameters, self.ctx.magnetic_extra_parameters)

        # setting pseudos
        pseudos = {}
        pseudo = self.inputs.pseudo
        for kind_name in self.ctx.structure.get_kind_names():
            pseudos[kind_name] = pseudo
        self.ctx.pseudos = pseudos

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

        extra_pw_parameters = {
            'SYSTEM': {
                'nbnd': int(nbands * nbands_factor),
                'nspin': 2,
                'starting_magnetization': {
                    self.ctx.element: 0.2,
                    'N': 0.0,
                },
            },
            'ELECTRONS': {
                'diagonalization': 'cg',
            }
        }
        self.ctx.extra_pw_parameters = update_dict(self.ctx.extra_pw_parameters,
                                             extra_pw_parameters)

    def is_fluorine_element(self):
        """Check if the element is magnetic"""
        return self.ctx.element == 'F'

    @abstractmethod
    def extra_setup_for_fluorine_element(self):
        """Extra setup for fluorine element"""

    @abstractmethod
    def setup_code_parameters_from_protocol(self):
        """Input validation"""

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

    @abstractmethod
    def run_reference(self):
        """
        run on reference calculation
        """

    @abstractmethod
    def run_samples_fix_dual(self):
        """
        run on all other evaluation sample points
        """

    @abstractmethod
    def inspect_fix_dual(self):
        """
        inspect the convergence run of fix dual and set the converge cutoff for follows.
        """

    def inspect_fix_wfc_cutoff(self):
        """
        run on different rho cutoffs
        """

    @abstractmethod
    def helper_compare_result_extract_fun(self, sample_node, reference_node, **kwargs) -> dict:
        """
        Must be implemented for specific convergence workflow to extrac the result
        Expected to return a dict of result.

        Get the node of sample and reference as input. Extract the parameters for
        properties extract helper function.
        """

    @abstractmethod
    def get_result_metadata(self):
        """
        define a dict of which is the metadata of the results, e.g. the unit of the properties
        return a list type

        for example:

        return {
            'absolute_unit': 'eV/atom',
            'relative_unit': '%',
        }
        """

    @abstractmethod
    def run_samples_fix_wfc_cutoff(self):
        """set results of sub-workflows to output ports"""

    def result_general_process(self, reference_node, sample_nodes, **kwargs) -> dict:
        """set results of sub-workflows to output ports"""
        children = sample_nodes
        success_children = [
            child for child in children if child.is_finished_ok
        ]

        ecutwfc_list = []
        ecutrho_list = []
        d_output_parameters = {}

        for key, value in self.get_result_metadata().items():
            d_output_parameters[key] = value

        for child_node in success_children:
            ecutwfc_list.append(child_node.inputs.ecutwfc.value)
            ecutrho_list.append(child_node.inputs.ecutrho.value)

            res = self.helper_compare_result_extract_fun(child_node,
                                                    reference_node, **kwargs)

            for key, value in res.items():
                if key not in self.get_result_metadata():
                    lst = d_output_parameters.get(key, [])
                    lst.append(value)
                    d_output_parameters.update({key: lst})

        d_output_parameters['ecutwfc'] = ecutwfc_list
        d_output_parameters['ecutrho'] = ecutrho_list

        return d_output_parameters

    @abstractmethod
    def final_results(self):
        """If more to parse for the final"""

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
