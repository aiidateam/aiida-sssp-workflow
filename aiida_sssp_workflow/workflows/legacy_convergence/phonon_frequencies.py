# -*- coding: utf-8 -*-
"""
Convergence test on phonon frequencies of a given pseudopotential
"""
import yaml
import importlib_resources

from aiida.engine import calcfunction
from aiida import orm
from aiida.engine import WorkChain, if_, append_, ToContext
from aiida.plugins import DataFactory

from aiida_sssp_workflow.utils import update_dict, \
    RARE_EARTH_ELEMENTS, \
    helper_parse_upf, get_standard_cif_filename_from_element
from aiida_sssp_workflow.workflows.evaluate._phonon_frequencies import PhononFrequenciesWorkChain

UpfData = DataFactory('pseudo.upf')


@calcfunction
def helper_phonon_frequencies_difference(input_parameters: orm.Dict,
                                         ref_parameters: orm.Dict) -> orm.Dict:
    """
    doc
    """
    import numpy as np

    input_frequencies = input_parameters['dynamical_matrix_0']['frequencies']
    ref_frequencies = ref_parameters['dynamical_matrix_0']['frequencies']
    diffs = np.array(input_frequencies) - np.array(ref_frequencies)
    weights = np.array(ref_frequencies)

    absolute_diff = np.mean(diffs)
    absolute_max_diff = np.amax(diffs)

    relative_diff = np.sqrt(np.mean((diffs / weights)**2)) * 100
    relative_max_diff = np.amax(diffs / weights) * 100

    return orm.Dict(
        dict={
            'relative_diff': relative_diff,
            'relative_max_diff': relative_max_diff,
            'absolute_diff': absolute_diff,
            'absolute_max_diff': absolute_max_diff,
            'absolute_unit': 'cm-1',
            'relative_unit': '%'
        })


class ConvergencePhononFrequenciesWorkChain(WorkChain):
    """WorkChain to converge test on cohisive energy of input structure"""
    # pylint: disable=too-many-instance-attributes

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
        spec.input('pw_code', valid_type=orm.Code,
                    help='The `pw.x` code use for the `PwCalculation`.')
        spec.input('ph_code', valid_type=orm.Code,
                    help='The `ph.x` code use for the `PhCalculation`.')
        spec.input('pseudo', valid_type=UpfData, required=True,
                    help='Pseudopotential to be verified')
        spec.input('protocol', valid_type=orm.Str, default=lambda: orm.Str('efficiency'),
                    help='The protocol to use for the workchain.')
        spec.input('dual', valid_type=orm.Float,
                    help='The dual to derive ecutrho from ecutwfc.(only for legacy convergence wf).')
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
            if_(cls.is_fluorine_element)(
                cls.extra_setup_for_fluorine_element, ),
            cls.setup_code_parameters_from_protocol,
            cls.setup_code_resource_options,
            cls.run_reference,
            cls.run_samples,
            cls.results,
        )

        spec.output('output_parameters', valid_type=orm.Dict, required=True,
                    help='The output parameters include results of all calculations.')

        spec.exit_code(401, 'ERROR_SUB_PROCESS_FAILED',
            message='The sub process for `{label}` did not finish successfully.')
        # yapy: enable

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

        self.ctx.pseudos = {element: self.inputs.pseudo}

        # Structures for convergence verification are all primitive structures
        # the original conventional structure comes from the same CIF files of
        # http:// molmod.ugent.be/deltacodesdft/
        # EXCEPT that for the element fluorine the `SiF4.cif` used for convergence
        # reason. But we do the structure setup for SiF4 in the following step:
        # `cls.extra_setup_for_fluorine_element`
        cif_file = get_standard_cif_filename_from_element(element)
        self.ctx.structure = orm.CifData.get_or_create(
            cif_file)[0].get_structure(primitive_cell=True)

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
            },
        }
        self.ctx.bulk_parameters = update_dict(self.ctx.pw_parameters,
                                               extra_parameters)

    def is_fluorine_element(self):
        """Check if the element is magnetic"""
        return self.ctx.element == 'F'

    def extra_setup_for_fluorine_element(self):
        """Extra setup for fluorine element"""
        cif_file = get_standard_cif_filename_from_element('SiF4')
        self.ctx.structure = orm.CifData.get_or_create(
            cif_file)[0].get_structure(primitive_cell=True)

        # setting pseudos
        import_path = importlib_resources.path(
            'aiida_sssp_workflow.REF.UPFs', 'Si.pbe-n-rrkjus_psl.1.0.0.UPF')
        with import_path as pp_path, open(pp_path, 'rb') as stream:
            upf_silicon = UpfData(stream)
            self.ctx.pseudos['Si'] = upf_silicon

    def setup_code_parameters_from_protocol(self):
        """Input validation"""
        # pylint: disable=invalid-name, attribute-defined-outside-init

        # Read from protocol if parameters not set from inputs
        protocol_name = self.inputs.protocol.value
        protocol = self._get_protocol()[protocol_name]
        protocol = protocol['convergence']['phonon_frequencies']
        self._DEGAUSS = protocol['degauss']
        self._OCCUPATIONS = protocol['occupations']
        self._SMEARING = protocol['smearing']
        self._CONV_THR = protocol['electron_conv_thr']
        self._KDISTANCE = protocol['kpoints_distance']
        self._QPOINTS_LIST = protocol['qpoints_list']

        self._PH_EPSILON = protocol['ph']['epsilon']
        self._PH_TR2_PH = protocol['ph']['tr2_ph']

        self._MAX_EVALUATE = protocol['max_evaluate']
        self._REFERENCE_ECUTWFC = protocol['reference_ecutwfc']

        self.ctx.qpoints_list = self._QPOINTS_LIST

        self.ctx.max_evaluate = self._MAX_EVALUATE
        self.ctx.reference_ecutwfc = self._REFERENCE_ECUTWFC

        self.ctx.pw_parameters = {
            'SYSTEM': {
                'degauss': self._DEGAUSS,
                'occupations': self._OCCUPATIONS,
                'smearing': self._SMEARING,
            },
            'ELECTRONS': {
                'conv_thr': self._CONV_THR,
            },
            'CONTROL': {
                'calculation': 'scf',
                'wf_collect': True,
                'tstress': True,
            },
        }

        self.ctx.ph_parameters = {
            'INPUTPH': {
                'tr2_ph': self._PH_TR2_PH,
                'epsil': self._PH_EPSILON,
            }
        }

        # set the ecutrho according to the type of pseudopotential
        # dual 4 for NC and 8 for all other type of PP.
        upf_header = helper_parse_upf(self.inputs.pseudo)
        if upf_header['pseudo_type'] in ['NC', 'SL']:
            dual = 4.0
        else:
            dual = 8.0

        if 'dual' in self.inputs:
            dual = self.inputs.dual

        self.ctx.dual = dual

        self.ctx.kpoints_distance = self._KDISTANCE

        self.report(
            f'The pw parameters for convergence is: {self.ctx.pw_parameters}'
        )
        self.report(
            f'The ph parameters for convergence is: {self.ctx.ph_parameters}'
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

    def _get_inputs(self, ecutwfc, ecutrho):
        """
        get inputs for the evaluation CohesiveWorkChain by provide ecutwfc and ecutrho,
        all other parameters are fixed for the following steps
        """
        # yapf: disable
        inputs = {
            'pw_code': self.inputs.pw_code,
            'ph_code': self.inputs.ph_code,
            'pseudos': self.ctx.pseudos,
            'structure': self.ctx.structure,
            'pw_base_parameters': orm.Dict(dict=self.ctx.pw_parameters),
            'ph_base_parameters': orm.Dict(dict=self.ctx.ph_parameters),
            'ecutwfc': orm.Float(ecutwfc),
            'ecutrho': orm.Float(ecutrho),
            'qpoints': orm.List(list=self.ctx.qpoints_list),
            'kpoints_distance': orm.Float(self.ctx.kpoints_distance),
            'options': orm.Dict(dict=self.ctx.options),
            'parallelization': orm.Dict(dict=self.ctx.parallelization),
            'clean_workdir': orm.Bool(False),   # will leave the workdir clean to outer most wf
        }
        # yapf: enable

        return inputs

    def run_reference(self):
        """
        run on reference calculation
        """
        ecutwfc = self.ctx.reference_ecutwfc
        ecutrho = ecutwfc * self.ctx.dual
        inputs = self._get_inputs(ecutwfc=ecutwfc, ecutrho=ecutrho)

        running = self.submit(PhononFrequenciesWorkChain, **inputs)
        self.report(
            f'launching reference PhononFrequenciesWorkChain<{running.pk}>')

        return ToContext(reference=running)

    def run_samples(self):
        """
        run on all other evaluation sample points
        """
        workchain = self.ctx.reference

        if not workchain.is_finished_ok:
            self.report(
                f'PwBaseWorkChain pk={workchain.pk} for reference run is failed.'
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED.format(
                label='reference')

        for idx in range(self.ctx.max_evaluate):
            ecutwfc = self._ECUTWFC_LIST[idx]
            ecutrho = ecutwfc * self.ctx.dual
            inputs = self._get_inputs(ecutwfc=ecutwfc, ecutrho=ecutrho)

            running = self.submit(PhononFrequenciesWorkChain, **inputs)
            self.report(
                f'launching [{idx}] PhononFrequenciesWorkChain<{running.pk}>')

            self.to_context(children=append_(running))

    def results(self):
        """
        results
        """
        reference = self.ctx.reference
        reference_parameters = reference.outputs.output_parameters

        children = self.ctx.children
        success_children = [
            child for child in children if child.is_finished_ok
        ]

        ecutwfc_list = []
        ecutrho_list = []
        absolute_diff_list = []
        d_output_parameters = {}

        for child in success_children:
            ecutwfc_list.append(child.inputs.ecutwfc.value)
            ecutrho_list.append(child.inputs.ecutrho.value)

            child_parameters = child.outputs.output_parameters
            res = helper_phonon_frequencies_difference(child_parameters,
                                                       reference_parameters)

            absolute_diff_list.append(res['absolute_diff'])

        d_output_parameters['ecutwfc_list'] = ecutwfc_list
        d_output_parameters['ecutrho_list'] = ecutrho_list
        d_output_parameters['absolute_list'] = absolute_diff_list

        self.out('output_parameters',
                 orm.Dict(dict=d_output_parameters).store())

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
