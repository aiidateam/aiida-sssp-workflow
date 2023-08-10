# -*- coding: utf-8 -*-
"""
Convergence test on bands of a given pseudopotential
"""

from aiida import orm
from aiida.engine import calcfunction
from aiida.plugins import DataFactory

from aiida_sssp_workflow.calculations.calculate_bands_distance import get_bands_distance
from aiida_sssp_workflow.utils import MAGNETIC_ELEMENTS, NONMETAL_ELEMENTS, update_dict
from aiida_sssp_workflow.workflows.convergence._base import _BaseConvergenceWorkChain
from aiida_sssp_workflow.workflows.evaluate._bands import BandsWorkChain

UpfData = DataFactory("pseudo.upf")


@calcfunction
def helper_bands_distence_difference(
    band_structure_a: orm.BandsData,
    band_parameters_a: orm.Dict,
    band_structure_b: orm.BandsData,
    band_parameters_b: orm.Dict,
    smearing: orm.Float,
    fermi_shift: orm.Float,
    do_smearing: orm.Bool,
    spin: orm.Bool,
):
    """The helper function to calculate the bands distance between two band structures.
    The function is called in last step of convergence workflow to get the bands distance.
    """

    # The raw implementation of `get_bands_distance` is in `aiida_sssp_workflow/calculations/bands_distance.py`
    bandsdata_a = {
        "number_of_electrons": band_parameters_a["number_of_electrons"],
        "number_of_bands": band_parameters_a["number_of_bands"],
        "fermi_level": band_parameters_a["fermi_energy"],
        "bands": band_structure_a.get_bands(),
        "kpoints": band_structure_a.get_kpoints(),
        "weights": band_structure_a.get_array("weights"),
    }
    bandsdata_b = {
        "number_of_electrons": band_parameters_b["number_of_electrons"],
        "number_of_bands": band_parameters_b["number_of_bands"],
        "fermi_level": band_parameters_b["fermi_energy"],
        "bands": band_structure_b.get_bands(),
        "kpoints": band_structure_b.get_kpoints(),
        "weights": band_structure_b.get_array("weights"),
    }
    res = get_bands_distance(
        bandsdata_a,
        bandsdata_b,
        smearing=smearing.value,
        fermi_shift=fermi_shift.value,
        do_smearing=do_smearing.value,
        spin=spin.value,
    )
    eta = res.get("eta_c", None)
    shift = res.get("shift_c", None)
    max_diff = res.get("max_diff_c", None)
    units = res.get("units", None)

    return orm.Dict(
        dict={
            "eta_c": eta,
            "shift_c": shift,
            "max_diff_c": max_diff,
            "bands_unit": units,
        }
    )


class ConvergenceBandsWorkChain(_BaseConvergenceWorkChain):
    """WorkChain to converge test on cohisive energy of input structure"""

    # pylint: disable=too-many-instance-attributes

    _PROPERTY_NAME = "bands"
    _EVALUATE_WORKCHAIN = BandsWorkChain
    _MEASURE_OUT_PROPERTY = "eta_c"

    def init_setup(self):
        super().init_setup()
        self.ctx.extra_pw_parameters = {
            "CONTROL": {
                "disk_io": "low",
            },
        }

    def setup_code_parameters_from_protocol(self):
        """Input validation"""
        # pylint: disable=invalid-name, attribute-defined-outside-init

        # Read from protocol if parameters not set from inputs
        super().setup_code_parameters_from_protocol()

        # parse protocol
        protocol = self.ctx.protocol
        self.ctx.degauss = self._DEGAUSS = protocol["degauss"]
        self._OCCUPATIONS = protocol["occupations"]
        self._SMEARING = protocol["smearing"]
        self._CONV_THR_PER_ATOM = protocol["conv_thr_per_atom"]
        self.ctx.kpoints_distance_scf = protocol["kpoints_distance_scf"]
        self.ctx.kpoints_distance_bands = protocol["kpoints_distance_bands"]

        # Set context parameters
        self.ctx.pw_parameters = super()._get_pw_base_parameters(
            self._DEGAUSS, self._OCCUPATIONS, self._SMEARING, self._CONV_THR_PER_ATOM
        )

        self.ctx.fermi_shift = protocol["fermi_shift"]
        self.ctx.init_nbands_factor = protocol["init_nbands_factor"]
        self.ctx.is_metal = self.ctx.element not in NONMETAL_ELEMENTS

        self.logger.info(f"The parameters for convergence is: {self.ctx.pw_parameters}")

    def _get_inputs(self, ecutwfc, ecutrho):
        """
        get inputs for the evaluation CohesiveWorkChain by provide ecutwfc and ecutrho,
        all other parameters are fixed for the following steps
        """
        parameters = {
            "SYSTEM": {
                "ecutwfc": ecutwfc,
                "ecutrho": ecutrho,
            },
        }

        parameters = update_dict(parameters, self.ctx.pw_parameters)

        parameters_bands = update_dict(parameters, {})
        parameters_bands["SYSTEM"].pop("nbnd", None)
        parameters_bands["CONTROL"].pop("tstress", None)
        parameters_bands["CONTROL"]["calculation"] = "bands"

        inputs = {
            "structure": self.ctx.structure,
            "scf": {
                "pw": {
                    "code": self.inputs.code,
                    "pseudos": self.ctx.pseudos,
                    "parameters": orm.Dict(dict=parameters),
                    "metadata": {
                        "options": self.ctx.options,
                    },
                    "parallelization": orm.Dict(dict=self.ctx.parallelization),
                },
                "kpoints_distance": orm.Float(self.ctx.kpoints_distance_scf),
            },
            "bands": {
                "pw": {
                    "code": self.inputs.code,
                    "pseudos": self.ctx.pseudos,
                    "parameters": orm.Dict(dict=parameters_bands),
                    "metadata": {
                        "options": self.ctx.options,
                    },
                    "parallelization": orm.Dict(dict=self.ctx.parallelization),
                },
            },
            "kpoints_distance_bands": orm.Float(self.ctx.kpoints_distance_bands),
            "init_nbands_factor": orm.Float(self.ctx.init_nbands_factor),
            "fermi_shift": orm.Float(self.ctx.fermi_shift),
            "run_bands_structure": orm.Bool(
                False
            ),  # for convergence with no band structure evaluate
            "clean_workdir": self.inputs.clean_workdir,
        }

        return inputs

    def helper_compare_result_extract_fun(self, sample_node, reference_node, **kwargs):
        """implement"""
        sample_band_parameters = sample_node.outputs.bands.band_parameters
        reference_band_parameters = reference_node.outputs.bands.band_parameters

        sample_band_structure = sample_node.outputs.bands.band_structure
        reference_band_structure = reference_node.outputs.bands.band_structure

        spin = self.ctx.element in MAGNETIC_ELEMENTS

        # Always process smearing to find fermi level even for non-metal elements.
        res = helper_bands_distence_difference(
            sample_band_structure,
            sample_band_parameters,
            reference_band_structure,
            reference_band_parameters,
            smearing=orm.Float(self.ctx.degauss),
            fermi_shift=orm.Float(self.ctx.fermi_shift),
            do_smearing=orm.Bool(True),
            spin=orm.Bool(spin),
        ).get_dict()

        return res

    def get_result_metadata(self):
        return {
            "fermi_shift": self.ctx.fermi_shift,
            "bands_unit": "meV",
        }
