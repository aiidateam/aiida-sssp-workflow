# -*- coding: utf-8 -*-
"""
Bands distance of many input pseudos
"""

from aiida import orm
from aiida.engine import ToContext, if_
from aiida.plugins import DataFactory

from aiida_sssp_workflow.utils import (
    HIGH_DUAL_ELEMENTS,
    MAGNETIC_ELEMENTS,
    NONMETAL_ELEMENTS,
    RARE_EARTH_ELEMENTS,
    get_magnetic_inputs,
    get_protocol,
    get_standard_structure,
    reset_pseudos_for_magnetic,
    update_dict,
)
from aiida_sssp_workflow.workflows.common import (
    get_extra_parameters_for_lanthanides,
    get_pseudo_element_and_type,
    get_pseudo_N,
)
from aiida_sssp_workflow.workflows.evaluate._bands import BandsWorkChain
from aiida_sssp_workflow.workflows.measure import _BaseMeasureWorkChain

UpfData = DataFactory('pseudo.upf')


def validate_input_pseudos(d_pseudos, _):
    """Validate that all input pseudos map to same element"""
    element = set(pseudo.element for pseudo in d_pseudos.values())

    if len(element) > 1:
        return f'The pseudos corespond to different elements {element}.'


class BandsMeasureWorkChain(_BaseMeasureWorkChain):
    """
    WorkChain to run bands measure,
    run without sym for distance compare and band structure along the path
    """
    _RY_TO_EV = 13.6056980659

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        # yapf: disable
        super().define(spec)

        spec.outline(
            cls.init_setup,
            if_(cls.is_magnetic_element)(
                cls.extra_setup_for_magnetic_element,
            ),
            if_(cls.is_rare_earth_element)(
                cls.extra_setup_for_rare_earth_element, ),
            cls.setup_pw_parameters_from_protocol,
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

        element = self.ctx.element = self.inputs.pseudo.element
        self.ctx.pseudos = {self.ctx.element: self.inputs.pseudo}

        self.ctx.structure = get_standard_structure(
            element,
            prop="bands",
        )

        # extra setting for bands convergence
        self.ctx.is_metal = element not in NONMETAL_ELEMENTS

    def is_magnetic_element(self):
        """Check if the element is magnetic"""
        return self.ctx.element in MAGNETIC_ELEMENTS

    def extra_setup_for_magnetic_element(self):
        """Extra setup for magnetic element"""
        self.ctx.structure, magnetic_extra_parameters = get_magnetic_inputs(self.ctx.structure)
        self.ctx.pw_parameters = update_dict(self.ctx.pw_parameters, magnetic_extra_parameters)

        # override pseudos setting
        # required for O, Mn, Cr where the kind names varies for sites
        self.ctx.pseudos = reset_pseudos_for_magnetic(self.inputs.pseudo, self.ctx.structure)


    def is_rare_earth_element(self):
        """Check if the element is rare earth"""
        return self.ctx.element in RARE_EARTH_ELEMENTS

    def extra_setup_for_rare_earth_element(self):
        """Extra setup for rare earth element"""
        nbnd_factor = 2.0
        pseudo_N = get_pseudo_N()
        self.ctx.pseudos['N'] = pseudo_N
        pseudo_RE = self.inputs.pseudo
        nbnd = nbnd_factor * (pseudo_N.z_valence + pseudo_RE.z_valence)
        pw_parameters = get_extra_parameters_for_lanthanides(self.ctx.element, nbnd)

        self.ctx.pw_parameters = update_dict(self.ctx.pw_parameters, pw_parameters)

    def setup_pw_parameters_from_protocol(self):
        """Input validation"""
        # pylint: disable=invalid-name, attribute-defined-outside-init

        # Read from protocol if parameters not set from inputs
        protocol = get_protocol(category="bands", name=self.inputs.protocol.value)
        self._DEGAUSS = protocol['degauss']
        self._OCCUPATIONS = protocol['occupations']
        self._SMEARING = protocol['smearing']
        self._CONV_THR = protocol['electron_conv_thr']

        self._INIT_NBANDS_FACTOR = protocol['init_nbands_factor']
        self._FERMI_SHIFT = protocol['fermi_shift']

        cutoff_control = get_protocol(
            category="control", name=self.inputs.cutoff_control.value
        )
        self._ECUTWFC = cutoff_control["max_wfc"]

        self.ctx.ecutwfc = self._ECUTWFC
        self.ctx.init_nbands_factor = self._INIT_NBANDS_FACTOR
        self.ctx.fermi_shift = self._FERMI_SHIFT

        self.ctx.kpoints_distance_scf = protocol['kpoints_distance_scf']
        self.ctx.kpoints_distance_bands = protocol['kpoints_distance_bands']
        self.ctx.kpoints_distance_band_structure = protocol['kpoints_distance_band_structure']


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

        self.ctx.ecutwfc = self._ECUTWFC

        self.ctx.pw_parameters = update_dict(self.ctx.pw_parameters, parameters)

        self.logger.info(
            f'The pw parameters for convergence is: {self.ctx.pw_parameters}'
        )

    def _get_inputs(self, pseudos):
        """
        get inputs for the bands evaluation with given pseudo
        """
        element, pseudo_type = get_pseudo_element_and_type(self.inputs.pseudo)
        if pseudo_type in ['NC', 'SL']:
            ecutrho = self.ctx.ecutwfc * 4
        else:
            ecutrho = self.ctx.ecutwfc * 8

        if element in HIGH_DUAL_ELEMENTS and pseudo_type not in ['NC', 'SL']:
            ecutrho = self.ctx.ecutwfc * 18

        if element in RARE_EARTH_ELEMENTS:
            # since nitrides is used, the pseudo of N is non-NC
            # The N.us.z_5.ld1.theose.v0 is used so set dual equal to 8
            ecutrho = self.ctx.ecutwfc * 8

        parameters = {
            "SYSTEM": {
                "ecutwfc": round(self.ctx.ecutwfc),
                "ecutrho": round(ecutrho),
            },
        }
        parameters = update_dict(parameters, self.ctx.pw_parameters)

        parameters_bands = update_dict(parameters, {})
        parameters_bands["SYSTEM"].pop("nbnd", None)

        inputs = {
            "structure": self.ctx.structure,
            "scf": {
                "pw": {
                    "code": self.inputs.code,
                    "pseudos": pseudos,
                    "parameters": orm.Dict(dict=parameters),
                    "metadata": {
                        "options": self.inputs.options.get_dict(),
                    },
                    "parallelization": self.inputs.parallelization,
                },
                "kpoints_distance": orm.Float(self.ctx.kpoints_distance_scf),
            },
            "bands": {
                "pw": {
                    "code": self.inputs.code,
                    "pseudos": pseudos,
                    "parameters": orm.Dict(dict=parameters_bands),
                    "metadata": {
                        "options": self.inputs.options.get_dict(),
                    },
                    "parallelization": self.inputs.parallelization,
                },
            },
            "kpoints_distance_bands": orm.Float(self.ctx.kpoints_distance_bands),
            "init_nbands_factor": orm.Float(self.ctx.init_nbands_factor),
            "fermi_shift": orm.Float(self.ctx.fermi_shift),
            "run_bands_structure": orm.Bool(True),
            "kpoints_distance_band_structure": orm.Float(self.ctx.kpoints_distance_band_structure),
            "clean_workchain": self.inputs.clean_workchain,
        }

        return inputs

    def run_bands_evaluation(self):
        """run bands evaluation of psp in inputs list"""
        inputs = self._get_inputs(self.ctx.pseudos)

        running = self.submit(BandsWorkChain, **inputs)

        self.report(
            f'launching BandsWorkChain<{running.pk}>'
        )

        return ToContext(bands=running)

    def finalize(self):
        """inspect bands run results"""
        self.out_many(
            self.exposed_outputs(self.ctx.bands, BandsWorkChain)
        )
