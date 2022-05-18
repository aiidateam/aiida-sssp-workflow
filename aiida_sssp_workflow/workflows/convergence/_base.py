# -*- coding: utf-8 -*-
"""
Base legacy work chain
"""
from abc import ABCMeta, abstractmethod

from aiida import orm
from aiida.engine import append_, if_
from aiida.plugins import DataFactory

from aiida_sssp_workflow.utils import (
    MAGNETIC_ELEMENTS,
    RARE_EARTH_ELEMENTS,
    convergence_analysis,
    get_magnetic_inputs,
    get_protocol,
    get_standard_structure,
    reset_pseudos_for_magnetic,
    update_dict,
)
from aiida_sssp_workflow.workflows import SelfCleanWorkChain
from aiida_sssp_workflow.workflows.common import (
    get_extra_parameters_and_pseudos_for_lanthanides,
)

UpfData = DataFactory('pseudo.upf')


class abstract_attribute(object):
    """lazy variable check: https://stackoverflow.com/a/32536493"""
    def __get__(self, obj, type):
        for cls in type.__mro__:
            for name, value in cls.__dict__.items():
                if value is self:
                    this_obj = obj if obj else type
                    raise NotImplementedError(
                        f'{this_obj!r} does not have the attribute {name!r} (abstract from class {cls.__name__!r})'
                    )

        raise NotImplementedError(
            '%s does not set the abstract attribute <unknown>', type.__name__)


class BaseConvergenceWorkChain(SelfCleanWorkChain):
    """Base legacy workchain"""
    # pylint: disable=too-many-instance-attributes
    __metaclass__ = ABCMeta

    _PROPERTY_NAME = abstract_attribute()   # used to get convergence protocol
    _EVALUATE_WORKCHAIN = abstract_attribute()
    _MEASURE_OUT_PROPERTY = abstract_attribute()

    # Default set to True, override it in subclass to turn it off
    _RUN_WFC_TEST = True
    _RUN_RHO_TEST = True

    @classmethod
    def define(cls, spec):
        super().define(spec)
        # yapf: disable
        spec.input('pw_code', valid_type=orm.Code,
                    help='The `pw.x` code use for the `PwCalculation`.')
        spec.input('pseudo', valid_type=UpfData, required=True,
                    help='Pseudopotential to be verified')
        spec.input('protocol', valid_type=orm.Str, required=True,
                    help='The calculation protocol to use for the workchain.')
        spec.input('cutoff_control', valid_type=orm.Str, required=True,
                    help='The cutoff control list to use for the workchain.')
        spec.input('criteria', valid_type=orm.Str, required=True,
                    help='Criteria for convergence measurement to give recommend cutoff pair.')
        spec.input('options', valid_type=orm.Dict, required=False,
                    help='Optional `options`.')
        spec.input('parallelization', valid_type=orm.Dict, required=False,
                    help='Parallelization options')

        spec.outline(
            cls.init_setup,
            if_(cls.is_magnetic_element)(
                cls.extra_setup_for_magnetic_element,
            ),
            if_(cls.is_rare_earth_element)(
                cls.extra_setup_for_rare_earth_element,
            ),
            cls.setup_code_parameters_from_protocol,
            cls.setup_criteria_parameters_from_protocol,
            cls.setup_code_resource_options,
            cls.run_reference,
            cls.inspect_reference,
            if_(cls._is_run_wfc_convergence_test)(
                cls.run_wfc_convergence_test,
                cls.inspect_wfc_convergence_test,
            ),
            if_(cls._is_run_rho_convergence_test)(
                cls.run_rho_convergence_test,
                cls.inspect_rho_convergence_test,
            ),
            cls.finalize,
        )

        spec.output('output_parameters_wfc_test', valid_type=orm.Dict, required=False,
                    help='The output parameters include results of all wfc test calculations.')
        spec.output('output_parameters_rho_test', valid_type=orm.Dict, required=False,
                    help='The output parameters include results of all rho test calculations.')
        spec.output('final_output_parameters', valid_type=orm.Dict, required=False,
                    help='The output parameters of two stage convergence test.')

        spec.exit_code(401, 'ERROR_REFERENCE_CALCULATION_FAILED',
            message='The reference calculation failed.')
        spec.exit_code(401, 'ERROR_SUB_PROCESS_FAILED',
            message='The sub process for `{label}` did not finish successfully.')
        # yapy: enable

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
        return a list type.

        This dict will be merged into the final `output_parameters` results.
        For different convergence workflow, you may want to add different metadata
        into output.

        for example:

        return {
            'absolute_unit': 'meV/atom',
            'relative_unit': '%',
        }
        """

    @abstractmethod
    def _get_inputs(self, ecutwfc: int, ecutrho: int) -> dict:
        """generate inputs for the evaluate workflow
        Must compatible with inputs format of every evaluate workflow.
        This is used in actual submit of reference and convergence tests.

        Since the ecutwfc and ecutrho has less point set to be a float with decimals.
        Therefore, always pass round to nearst int inputs for ecutwfc and ecutrho.
        It also bring the advantage that the inputs to final calcjob always the same
        for these two inputs and caching will properly triggered.
        """
        # TODO: there can be a validation of inputs and the evaluate workflow inputs ports

    def _is_run_wfc_convergence_test(self):
        """If running wavefunction convergence test
        default True, override class attribute `_RUN_WFC_TEST`
        in subclass to supress running it"""
        return self._RUN_WFC_TEST

    def _is_run_rho_convergence_test(self):
        """If running charge density convergence test
        default True, override class attribute `_RUN_RHO_TEST` in
        subclass to supress running it"""
        return self._RUN_RHO_TEST

    def init_setup(self):
        """
        This step contains all preparation before actaul setup, e.g. set
        the context of element, base_structure, base pw_parameters and pseudos.
        """
        # parse pseudo and output its header information
        from pseudo_parser.upf_parser import parse_element, parse_pseudo_type

        # init output_parameters to store output
        self.ctx.output_parameters = {}

        cutoff_control = get_protocol(category='control', name=self.inputs.cutoff_control.value)
        self.ctx.ecutwfc_list = self._ECUTWFC_LIST = cutoff_control['wfc_scan']

        # use the last cutoff as reference
        # IMPORTANT to convert to float since the value should have
        # the same tye so the caching will correctly activated.
        self.ctx.reference_ecutwfc = self._ECUTWFC_LIST[-1]

        self.ctx.extra_pw_parameters = {}
        content = self.inputs.pseudo.get_content()
        element = parse_element(content)
        pseudo_type = parse_pseudo_type(content)
        self.ctx.element = element
        self.ctx.pseudo_type = pseudo_type

        # set the ecutrho according to the type of pseudopotential
        # dual 4 for NC and 10 for all other type of PP.
        if self.ctx.pseudo_type in ['NC', 'SL']:
            self.ctx.dual = 4.0
            self.ctx.dual_scan_list = cutoff_control['nc_dual_scan']
        else:
            # the initial dual set to 10 to make sure it is enough and converged
            # In the follow up steps will converge on ecutrho
            self.ctx.dual = 8.0

            # For the non-NC pseudos we should be careful that high charge density cutoff
            # is needed.
            # We recommond to set the scan wide range from to find the best
            # charge density cutoff.
            self.ctx.dual_scan_list = cutoff_control['nonnc_dual_scan']

        # TODO: for extrem high dual elements: O Fe Hf etc.

        self.ctx.pseudos = {element: self.inputs.pseudo}

        # Please check README for what and why we use configuration set 'convergence'
        # for convergence verification.
        self.ctx.structure = get_standard_structure(element, prop='convergence')

    def is_magnetic_element(self):
        """Check if the element is magnetic"""
        return self.ctx.element in MAGNETIC_ELEMENTS

    def extra_setup_for_magnetic_element(self):
        """
        Extra setup for magnetic element, set starting magnetization
        and reset pseudos to correspont elements name.
        """
        self.ctx.structure, magnetic_extra_parameters = get_magnetic_inputs(self.ctx.structure)
        self.ctx.extra_pw_parameters = update_dict(self.ctx.extra_pw_parameters, magnetic_extra_parameters)

        # override pseudos setting
        # required for O, Mn, Cr where the kind names varies for sites
        self.ctx.pseudos = reset_pseudos_for_magnetic(self.inputs.pseudo, self.ctx.structure)

    def is_rare_earth_element(self):
        """Check if the element is rare earth"""
        return self.ctx.element in RARE_EARTH_ELEMENTS

    def extra_setup_for_rare_earth_element(self):
        """
        Extra setup for rare-earth element same as magnetic elements

        We use nitrdes configuration for the convergence verification of rare-earth elements.
        Otherwise it is hard to get converged in scf calculation.
        """
        self.ctx.extra_pw_parameters, self.ctx.pseudos = \
            get_extra_parameters_and_pseudos_for_lanthanides(
                self.ctx.element,
                pseudo_RE=self.inputs.pseudo
            )

    def setup_code_parameters_from_protocol(self):
        """unzip and parse protocol parameters to context"""
        protocol = get_protocol(category='converge', name=self.inputs.protocol.value)
        self.ctx.protocol = {
            **protocol['base'],
            **protocol.get(self._PROPERTY_NAME, dict()),    # if _PROPERTY_NAME not set, simply use base
        }

    def _get_pw_base_parameters(self, degauss, occupations, smearing, conv_thr):
        """Return base pw parameters dict for all convengence bulk workflow
        Unchanged dict for caching purpose

        TODO: move this method out of class
        """
        parameters = {
            'SYSTEM': {
                'degauss': degauss,
                'occupations': occupations,
                'smearing': smearing,
            },
            'ELECTRONS': {
                'conv_thr': conv_thr,
            },
            'CONTROL': {
                'calculation': 'scf',
                'wf_collect': True,
                'tstress': True,
            },
        }

        # update with extra pw params, for magnetic and lanthenides
        if self.ctx.extra_pw_parameters:
            parameters = update_dict(parameters, self.ctx.extra_pw_parameters)

        return parameters

    def setup_criteria_parameters_from_protocol(self):
        """Input validation"""
        self.ctx.criteria = get_protocol(
            category='criteria', name=self.inputs.criteria.value
        )[self._PROPERTY_NAME]

    def setup_code_resource_options(self):
        """
        setup resource options and parallelization for `PwCalculation` from inputs
        """
        if 'options' in self.inputs:
            self.ctx.options = self.inputs.options.get_dict()
        else:
            from aiida_sssp_workflow.utils import get_default_options

            self.ctx.options = get_default_options(with_mpi=True)

        if 'parallelization' in self.inputs:
            self.ctx.parallelization = self.inputs.parallelization.get_dict()
        else:
            self.ctx.parallelization = {}

    def run_reference(self):
        """
        run on reference calculation
        """
        ecutwfc = self.ctx.reference_ecutwfc
        ecutrho = ecutwfc * self.ctx.dual
        inputs = self._get_inputs(ecutwfc=round(ecutwfc), ecutrho=round(ecutrho))

        running = self.submit(self._EVALUATE_WORKCHAIN, **inputs)
        self.report(f'launching reference calculation: {running.process_label}<{running.pk}>')

        self.to_context(reference=running)

    def inspect_reference(self):
        try:
            workchain = self.ctx.reference
        except AttributeError as e:
            raise RuntimeError('Reference evaluation is not triggered') from e

        if not workchain.is_finished_ok:
            self.report(
                f'{workchain.process_label} pk={workchain.pk} for reference run is failed.'
            )
            return self.exit_codes.ERROR_REFERENCE_CALCULATION_FAILED

    def run_wfc_convergence_test(self):
        """
        run on all other evaluation sample points
        """
        self.ctx.max_ecutrho = ecutrho = self.ctx.reference_ecutwfc * self.ctx.dual

        for ecutwfc in self.ctx.ecutwfc_list[:-1]: # The last one is reference
            ecutwfc, ecutrho = round(ecutwfc), round(ecutrho)
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
        res = convergence_analysis(orm.List(list=list(zip(x, y))),
                                   orm.Dict(dict=self.ctx.criteria))

        self.ctx.wfc_cutoff, y_value = res['cutoff'].value, res['value'].value
        self.ctx.output_parameters['wavefunction_cutoff'] = self.ctx.wfc_cutoff

        self.logger.info(
            f'The wfc convergence at {self.ctx.wfc_cutoff} with value={y_value}'
        )

    def run_rho_convergence_test(self):
        """
        run rho converge test on fix wfc cutoff
        """

        ecutwfc = self.ctx.wfc_cutoff
        # Only run rho test when ecutrho less than the max reference
        # otherwise meaningless for the exceeding cutoff test
        for ecutrho in [dual * ecutwfc for dual in self.ctx.dual_scan_list if dual * ecutwfc < self.ctx.max_ecutrho]:
            ecutwfc, ecutrho = round(ecutwfc), round(ecutrho)
            inputs = self._get_inputs(ecutwfc=ecutwfc, ecutrho=ecutrho)

            running = self.submit(self._EVALUATE_WORKCHAIN, **inputs)
            self.report(
                f'launching fix ecutwfc={ecutwfc} [ecutrho={ecutrho}] {running.process_label}<{running.pk}>'
            )

            self.to_context(children_rho=append_(running))

    def inspect_rho_convergence_test(self):
        sample_nodes = self.ctx.children_rho + [self.ctx.reference]

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
        res = convergence_analysis(orm.List(list=list(zip(x, y))),
                                    orm.Dict(dict=self.ctx.criteria))
        self.ctx.rho_cutoff, y_value = res['cutoff'].value, res[
            'value'].value
        self.ctx.output_parameters['chargedensity_cutoff'] = self.ctx.rho_cutoff

        self.logger.info(
            f'The rho convergence at {self.ctx.rho_cutoff} with value={y_value}'
        )

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

    def finalize(self):
        # store output_parameters
        self.out('final_output_parameters',
                 orm.Dict(dict=self.ctx.output_parameters).store())
