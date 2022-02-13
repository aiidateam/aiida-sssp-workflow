# -*- coding: utf-8 -*-
"""
Base legacy work chain
"""
from abc import ABCMeta, abstractmethod
from argon2 import extract_parameters
import importlib_resources
import yaml

from aiida import orm
from aiida.engine import WorkChain, if_, append_
from aiida.plugins import DataFactory

from aiida_sssp_workflow.utils import RARE_EARTH_ELEMENTS, \
    MAGNETIC_ELEMENTS, \
    get_standard_cif_filename_from_element, \
    update_dict, \
    helper_get_magnetic_inputs, \
    convergence_analysis

UpfData = DataFactory('pseudo.upf')

class abstract_attribute(object):
    """lazy variable check: https://stackoverflow.com/a/32536493"""
    def __get__(self, obj, type):   
        for cls in type.__mro__:
            for name, value in cls.__dict__.items():
                if value is self:
                    this_obj = obj if obj else type
                    raise NotImplementedError(
                         "%r does not have the attribute %r "
                         "(abstract from class %r)" %
                             (this_obj, name, cls.__name__))

        raise NotImplementedError(
            "%s does not set the abstract attribute <unknown>", type.__name__)

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

    _EVALUATE_WORKCHAIN = abstract_attribute()
    _MEASURE_OUT_PROPERTY = abstract_attribute()

    @classmethod
    def define(cls, spec):
        super().define(spec)
        # yapf: disable
        spec.input('pseudo', valid_type=UpfData, required=True,
                    help='Pseudopotential to be verified')
        spec.input('protocol_calculation', valid_type=orm.Str, default=lambda: orm.Str('theos'),
                    help='The calculation protocol to use for the workchain.')
        spec.input('protocol_criteria', valid_type=orm.Str, default=lambda: orm.Str('theos'),
                    help='The criteria protocol to use for the workchain.')
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
            cls.setup_criteria_parameters_from_protocol,
            cls.setup_code_resource_options,
            cls.run_reference,
            cls.inspect_reference,
            cls.run_wfc_convergence_test,
            cls.inspect_wfc_convergence_test,
            cls.run_rho_convergence_test,
            cls.inspect_rho_convergence_test,
            cls.final_results,
        )

        spec.output('output_parameters_wfc_test', valid_type=orm.Dict, required=True,
                    help='The output parameters include results of all wfc test calculations.')
        spec.output('output_parameters_rho_test', valid_type=orm.Dict, required=True,
                    help='The output parameters include results of all rho test calculations.')
        spec.output('final_output_parameters', valid_type=orm.Dict, required=True,
                    help='The output parameters of two stage convergence test.')

        spec.exit_code(401, 'ERROR_SUB_PROCESS_FAILED',
            message='The sub process for `{label}` did not finish successfully.')
        # yapy: enable

    def _get_protocol(self, ptype):
        """Load and read protocol from faml file to a verbose dict"""
        if ptype == 'calculation':
            filename = 'PROTOCOL_CALC.yml'
        else:
            filename = 'PROTOCOL_CRI.yml'

        import_path = importlib_resources.path('aiida_sssp_workflow', filename)
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
            self.ctx.max_dual = 4.0
            self.ctx.dual_scan_list = [2.0, 2.5, 3.0, 3.5, 4.0]
        else:
            # the initial dual set to 10 to make sure it is enough and converged
            # In the follow up steps will converge on ecutrho
            self.ctx.init_dual = 8.0
            self.ctx.min_dual = 6.0
            
            # For the non-NC pseudos we should be careful that high charge density cutoff 
            # is needed. 
            # We set the scan range from dual=8.0 to dual=18.0 to find the best 
            # charge density cutoff. 
            self.ctx.max_dual = self.ctx.init_dual + 10
            self.ctx.dual_scan_list = [6.0, 6.5, 7.0, 7.5, 8.0, 9.0, 10.0, 12.0, 15.0, 18.0]

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
        self.ctx.structure = self.ctx.cif.get_structure(primitive_cell=False)

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

    def extra_setup_for_fluorine_element(self):
        """Extra setup for fluorine element"""
        cif_file = get_standard_cif_filename_from_element('SiF4')
        self.ctx.structure = orm.CifData.get_or_create(
            cif_file, use_first=True)[0].get_structure(primitive_cell=True)

        # setting pseudos
        import_path = importlib_resources.path(
            'aiida_sssp_workflow.REF.UPFs', 'Si.pbe-n-rrkjus_psl.1.0.0.UPF')
        with import_path as pp_path, open(pp_path, 'rb') as stream:
            upf_silicon = UpfData(stream)
            self.ctx.pseudos['Si'] = upf_silicon

    def setup_code_parameters_from_protocol(self):
        """Input validation"""
        protocol_name = self.inputs.protocol_calculation.value
        self.ctx.protocol_calculation = self._get_protocol(ptype='calculation')[protocol_name]

    def setup_criteria_parameters_from_protocol(self):
        """Input validation"""
        protocol_name = self.inputs.protocol_criteria.value
        self.ctx.protocol_criteria = self._get_protocol(ptype='criteria')[protocol_name]

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

    def run_reference(self):
        """
        run on reference calculation
        """
        ecutwfc = self.ctx.reference_ecutwfc
        ecutrho = ecutwfc * self.ctx.init_dual
        inputs = self._get_inputs(ecutwfc=ecutwfc, ecutrho=ecutrho)

        running = self.submit(self._EVALUATE_WORKCHAIN, **inputs)
        self.report(f'launching reference {running.process_label}<{running.pk}>')

        self.to_context(reference=running)
        
    def inspect_reference(self):
        try:
            workchain = self.ctx.reference
        except AttributeError as exc:
            raise RuntimeError('Reference evaluation is not triggered') from exc

        if not workchain.is_finished_ok:
            self.report(
                f'{workchain.process_label} pk={workchain.pk} for reference run is failed with exit_code={workchain.exit_status}.'
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED.format(
                label=f'reference')

    def run_wfc_convergence_test(self):
        """
        run on all other evaluation sample points
        """
        ecutrho = self._REFERENCE_ECUTWFC * self.ctx.init_dual
        
        for idx in range(self.ctx.max_evaluate):
            ecutwfc = self._ECUTWFC_LIST[idx]
            inputs = self._get_inputs(ecutwfc=ecutwfc, ecutrho=ecutrho)

            running = self.submit(self._EVALUATE_WORKCHAIN, **inputs)
            self.report(
                f'launching fix ecutrho={ecutrho} [ecutwfc={ecutwfc}] {running.process_label}<{running.pk}>')

            self.to_context(children_wfc=append_(running))

    def inspect_wfc_convergence_test(self):
        # include reference node in the last
        sample_nodes = self.ctx.children_wfc + [self.ctx.reference]
        
        if 'extra_parameters' in self.ctx:
            output_parameters = self.result_general_process(
                self.ctx.reference, 
                sample_nodes,
                extra_parameters=self.ctx.extra_parameters
            )
        else:
            output_parameters = self.result_general_process(
                self.ctx.reference, sample_nodes)

        self.out('output_parameters_wfc_test',
                 orm.Dict(dict=output_parameters).store())

        # from the fix dual result find the converge wfc cutoff
        x = output_parameters['ecutwfc']
        y = output_parameters[self._MEASURE_OUT_PROPERTY]
        criteria = self.ctx.criteria['wfc_test']
        res = convergence_analysis(orm.List(list=list(zip(x, y))),
                                   orm.Dict(dict=criteria))

        self.ctx.wfc_cutoff, y_value = res['cutoff'].value, res['value'].value

        self.report(
            f'The wfc convergence at {self.ctx.wfc_cutoff} with value={y_value}'
        )
        
    def run_rho_convergence_test(self):
        """
        run on all other evaluation sample points
        """
        import numpy as np

        ecutwfc = self.ctx.wfc_cutoff
        for dual in self.ctx.dual_scan_list:
            ecutrho = ecutwfc * dual
            inputs = self._get_inputs(ecutwfc=ecutwfc, ecutrho=ecutrho)

            running = self.submit(self._EVALUATE_WORKCHAIN, **inputs)
            self.report(
                f'launching fix ecutwfc={ecutwfc} [ecutrho={ecutrho}] {running.process_label}<{running.pk}>'
            )

            self.to_context(children_rho=append_(running))
        
    def inspect_rho_convergence_test(self):
        sample_nodes = self.ctx.children_rho
        
        if 'extra_parameters' in self.ctx:
            output_parameters = self.result_general_process(
                self.ctx.reference, 
                sample_nodes,
                extra_parameters=self.ctx.extra_parameters
            )
        else:
            output_parameters = self.result_general_process(
                self.ctx.reference, sample_nodes)

        self.out('output_parameters_rho_test',
                    orm.Dict(dict=output_parameters).store())

        # from the fix wfc cutoff result find the converge rho cutoff
        x = output_parameters['ecutrho']
        y = output_parameters[self._MEASURE_OUT_PROPERTY]
        criteria = self.ctx.criteria['rho_test']
        res = convergence_analysis(orm.List(list=list(zip(x, y))),
                                    orm.Dict(dict=criteria))
        self.ctx.rho_cutoff, y_value = res['cutoff'].value, res[
            'value'].value

        self.report(
            f'The rho convergence at {self.ctx.rho_cutoff} with value={y_value}'
        )

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

    def final_results(self):
        output_parameters = {
            'wfc_cutoff': self.ctx.wfc_cutoff,
            'rho_cutoff': self.ctx.rho_cutoff,
        }

        self.out('final_output_parameters',
                 orm.Dict(dict=output_parameters).store())

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
