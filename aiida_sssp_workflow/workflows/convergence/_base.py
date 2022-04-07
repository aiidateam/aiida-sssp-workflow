# -*- coding: utf-8 -*-
"""
Base legacy work chain
"""
from abc import ABCMeta, abstractmethod

from aiida import orm
from aiida.engine import WorkChain, append_, if_
from aiida.plugins import DataFactory

from aiida_sssp_workflow.utils import (
    MAGNETIC_ELEMENTS,
    RARE_EARTH_ELEMENTS,
    convergence_analysis,
    get_protocol,
    get_standard_structure,
    reset_pseudos_for_magnetic,
    update_dict,
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


class BaseLegacyWorkChain(WorkChain):
    """Base legacy workchain"""
    # pylint: disable=too-many-instance-attributes
    __metaclass__ = ABCMeta

    _MAX_WALLCLOCK_SECONDS = 1800 * 3

    _PROPERTY_NAME = abstract_attribute()   # used to get convergence protocol
    _EVALUATE_WORKCHAIN = abstract_attribute()
    _MEASURE_OUT_PROPERTY = abstract_attribute()

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
        spec.input('clean_workdir', valid_type=orm.Bool, default=lambda: orm.Bool(False),
                    help='If `True`, work directories of all called calculation will be cleaned at the end of execution.')

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

    def init_setup(self):
        """
        This step contains all preparation before actaul setup, e.g. set
        the context of element, base_structure, base pw_parameters and pseudos.
        """
        # parse pseudo and output its header information
        from pseudo_parser.upf_parser import parse_element, parse_pseudo_type

        cutoff_control = get_protocol(category='control', name=self.inputs.cutoff_control.value)
        self._ECUTWFC_LIST = cutoff_control['wfc_scan']
        self._REFERENCE_ECUTWFC = self._ECUTWFC_LIST[-1]    # use the last cutoff as reference

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

        # Structures for convergence verification are all primitive structures
        # the original conventional structure comes from the same CIF files of
        # http:// molmod.ugent.be/deltacodesdft/
        # EXCEPT that for the element fluorine the `SiF4.cif` used for convergence
        # reason. But we do the structure setup for SiF4 in the following step:
        # `cls.extra_setup_for_fluorine_element`
        self.ctx.structure = get_standard_structure(element, prop='convergence')

    def is_magnetic_element(self):
        """Check if the element is magnetic"""
        return self.ctx.element in MAGNETIC_ELEMENTS

    @staticmethod
    def _get_extra_parameters_and_pseudos_for_mag_on(structure, pseudo):
        """
        Return extra parameters and magnetic pseudos setting with given
        structure data and pseudo data.
        """
        mag_structure = orm.StructureData(cell=structure.cell, pbc=structure.pbc)
        element = kind_name = structure.get_kind_names()[0]

        for i, site in enumerate(structure.sites):
            mag_structure.append_atom(
                position=site.position, symbols=kind_name, name=f"{kind_name}{i+1}"
            )

        extra_parameters = {
            "SYSTEM": {
                "nspin": 2,
                "starting_magnetization": {
                    f'{element}1': 0.5,
                    f'{element}2': -0.4,
                },
            },
        }
        # override pseudos setting for two sites of diamond cell
        pseudos = reset_pseudos_for_magnetic(pseudo, mag_structure)

        return extra_parameters, pseudos, mag_structure

    def extra_setup_for_magnetic_element(self):
        """
        Extra setup for magnetic element

        We use diamond configuration for the convergence verification.
        It contains two atoms in the cell. For the magnetic elements, it makes
        more sense that the two atom sites are distinguished so that the symmetry
        is broken.
        The starting magnetizations are set to [0.5, -0.4] for two sites.
        """
        extra_parameters, self.ctx.pseudos, self.ctx.structure = self._get_extra_parameters_and_pseudos_for_mag_on(
            self.ctx.structure, self.inputs.pseudo)

        self.ctx.extra_pw_parameters = update_dict(self.ctx.extra_pw_parameters, extra_parameters)

    def is_rare_earth_element(self):
        """Check if the element is rare earth"""
        return self.ctx.element in RARE_EARTH_ELEMENTS

    def extra_setup_for_rare_earth_element(self):
        """
        Extra setup for rare-earth element same as magnetic elements

        We use diamond configuration for the convergence verification.
        It contains two atoms in the cell. For the magnetic elements, it makes
        more sense that the two atom sites are distinguished so that the symmetry
        is broken.
        The starting magnetizations are set to [0.5, -0.4] for two sites.
        """
        extra_parameters, self.ctx.pseudos, self.ctx.structure = self._get_extra_parameters_and_pseudos_for_mag_on(
            self.ctx.structure, self.inputs.pseudo)

        self.ctx.extra_pw_parameters = update_dict(self.ctx.extra_pw_parameters, extra_parameters)


    def setup_code_parameters_from_protocol(self):
        """unzip and parse protocol parameters to context"""
        protocol = get_protocol(category='converge', name=self.inputs.protocol.value)
        self.ctx.protocol = {
            **protocol['base'],
            **protocol[self._PROPERTY_NAME]
        }

    def _get_pw_base_parameters(self, degauss, occupations, smearing, conv_thr):
        """Return base pw parameters dict for all convengence bulk workflow
        Unchanged dict for caching purpose"""
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

        # update with extra pw params, for magnetic ane lanthenides
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
        ecutwfc = self._REFERENCE_ECUTWFC
        ecutrho = ecutwfc * self.ctx.dual
        inputs = self._get_inputs(ecutwfc=ecutwfc, ecutrho=ecutrho)

        running = self.submit(self._EVALUATE_WORKCHAIN, **inputs)
        self.report(f'launching reference {running.process_label}<{running.pk}>')

        self.to_context(reference=running)

    def inspect_reference(self):
        try:
            workchain = self.ctx.reference
        except AttributeError as e:
            raise RuntimeError('Reference evaluation is not triggered') from e

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
        self.ctx.max_ecutrho = ecutrho = self._REFERENCE_ECUTWFC * self.ctx.dual

        for ecutwfc in self._ECUTWFC_LIST[:-1]: # The last one is reference
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

        self.report(
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
