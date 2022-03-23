# -*- coding: utf-8 -*-
"""
Bands distance of many input pseudos
"""
import importlib

from aiida import orm
from aiida.engine import ToContext, WorkChain, if_
from aiida.plugins import DataFactory

from aiida_sssp_workflow.utils import (
    NONMETAL_ELEMENTS,
    RARE_EARTH_ELEMENTS,
    get_protocol,
    get_standard_cif_filename_from_element,
    update_dict,
)
from aiida_sssp_workflow.workflows.evaluate._bands import BandsWorkChain

UpfData = DataFactory('pseudo.upf')


def validate_input_pseudos(d_pseudos, _):
    """Validate that all input pseudos map to same element"""
    element = set(pseudo.element for pseudo in d_pseudos.values())

    if len(element) > 1:
        return f'The pseudos corespond to different elements {element}.'


class BandsMeasureWorkChain(WorkChain):
    """
    WorkChain to run bands measure,
    run without sym for distance compare and band structure along the path
    """
    _MAX_WALLCLOCK_SECONDS = 1800 * 3

    _LARGE_DUAL_ELEMENTS = ['Fe', 'Hf']
    _RY_TO_EV = 13.6056980659

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        super().define(spec)
        # yapf: disable
        spec.input('pw_code', valid_type=orm.Code,
                    help='The `pw.x` code use for the `PwCalculation`.')
        spec.input('pseudo', valid_type=UpfData, required=True,
                    help='Pseudopotential to be verified')
        spec.input('protocol', valid_type=orm.Str, required=True,
                    help='The protocol which define input calculation parameters.')
        spec.input('cutoff_control', valid_type=orm.Str, default=lambda: orm.Str('standard'),
                    help='The control protocol where define max_wfc.')
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
            cls.setup_pw_parameters_from_protocol,
            cls.setup_pw_resource_options,
            cls.run_bands_evaluation,
            cls.finalize,
        )

        spec.expose_outputs(BandsWorkChain)

    def init_setup(self):
        """
        This step contains all preparation before actaul setup, e.g. set
        the context of element, base_structure, base pw_parameters and pseudos.
        """
        # parse pseudo and output its header information
        self.ctx.pw_parameters = {}
        self.ctx.extra_parameters = {}

        element = self.ctx.element = self.inputs.pseudo.element

        # Structures for convergence verification are all primitive structures
        # the original conventional structure comes from the same CIF files of
        # http://molmod.ugent.be/deltacodesdft/
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
        import_path = importlib.resources.path('aiida_sssp_workflow.statics.UPFs',
                                               'N.pbe-n-radius_5.upf')
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

    def setup_pw_parameters_from_protocol(self):
        """Input validation"""
        # pylint: disable=invalid-name, attribute-defined-outside-init

        # Read from protocol if parameters not set from inputs
        protocol = get_protocol(category="bands", name=self.inputs.protocol.value)
        self._DEGAUSS = protocol['degauss']
        self._OCCUPATIONS = protocol['occupations']
        self._SMEARING = protocol['smearing']
        self._CONV_THR = protocol['electron_conv_thr']
        self._KDISTANCE = protocol['kpoints_distance']

        self._INIT_NBANDS_FACTOR = protocol['init_nbands_factor']
        self._BANDS_SHIFT = protocol['bands_shift']

        cutoff_control = get_protocol(
            category="control", name=self.inputs.cutoff_control.value
        )
        self._ECUTWFC = cutoff_control["max_wfc"]

        self.ctx.ecutwfc = self._ECUTWFC
        self.ctx.kpoints_distance = self._KDISTANCE
        self.ctx.init_nbands_factor = self._INIT_NBANDS_FACTOR
        self.ctx.bands_shift = self._BANDS_SHIFT

        parameters = {
            "SYSTEM": {
                "degauss": self._DEGAUSS,
                "occupations": self._OCCUPATIONS,
                "smearing": self._SMEARING,
            },
            "ELECTRONS": {
                "conv_thr": self._CONV_THR,
            },
        }

        # TBD: Always use dual=8 since pseudo_O here is non-NC
        self.ctx.ecutwfc = self._ECUTWFC
        self.ctx.ecutrho = self._ECUTWFC * 8

        self.ctx.pw_parameters = update_dict(self.ctx.pw_parameters, parameters)

        self.report(
            f'The pw parameters for convergence is: {self.ctx.pw_parameters}'
        )

    def setup_pw_resource_options(self):
        """
        setup resource options and parallelization for `PwCalculation` from inputs
        """
        if "options" in self.inputs:
            self.ctx.options = self.inputs.options.get_dict()
        else:
            from aiida_sssp_workflow.utils import get_default_options

            self.ctx.options = get_default_options(
                max_wallclock_seconds=self._MAX_WALLCLOCK_SECONDS, with_mpi=True
            )

        if "parallelization" in self.inputs:
            self.ctx.parallelization = self.inputs.parallelization.get_dict()
        else:
            self.ctx.parallelization = {}

        self.report(f"resource options set to {self.ctx.options}")
        self.report(f"parallelization options set to {self.ctx.parallelization}")

    def _get_inputs(self, element, pseudos):
        """
        get inputs for the bands evaluation with given pseudo
        """
        if element in RARE_EARTH_ELEMENTS:
            pseudos['N'] = self.ctx.pseudo_N

        inputs = {
            'code': self.inputs.pw_code,
            'pseudos': pseudos,
            'structure': self.ctx.structure,
            'pw_base_parameters': orm.Dict(dict=self.ctx.pw_parameters),
            'ecutwfc': orm.Float(self.ctx.ecutwfc),
            'ecutrho': orm.Float(self.ctx.ecutrho),
            'kpoints_distance': orm.Float(self.ctx.kpoints_distance),
            'init_nbands_factor': orm.Float(self.ctx.init_nbands_factor),
            'bands_shift': orm.Float(self.ctx.bands_shift),
            'should_run_bands_structure': orm.Bool(True),
            'options': orm.Dict(dict=self.ctx.options),
            'parallelization': orm.Dict(dict=self.ctx.parallelization),
            'clean_workdir': orm.Bool(False),   # will leave the workdir clean to outer most wf
        }

        return inputs

    def run_bands_evaluation(self):
        """run bands evaluation of psp in inputs list"""
        pseudos = {self.ctx.element: self.inputs.pseudo}

        inputs = self._get_inputs(self.ctx.element, pseudos)

        running = self.submit(BandsWorkChain, **inputs)

        self.report(
            f'launching pseudo >_<: {self.inputs.pseudo} BandsWorkChain<{running.pk}>'
        )

        return ToContext(bands=running)

    def finalize(self):
        """inspect bands run results"""
        self.out_many(
            self.exposed_outputs(self.ctx.bands, BandsWorkChain)
        )
