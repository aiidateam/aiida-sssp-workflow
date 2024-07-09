# -*- coding: utf-8 -*-
"""
Convergence test on pressure of a given pseudopotential
"""

from pathlib import Path
from typing import Union, Any

from aiida import orm
from aiida.engine import ProcessBuilder
from aiida_pseudo.data.pseudo import UpfData

from aiida_sssp_workflow.utils import get_default_mpi_options
from aiida_sssp_workflow.workflows.convergence._base import _BaseConvergenceWorkChain
from aiida_sssp_workflow.workflows.evaluate._eos import _EquationOfStateWorkChain
from aiida_sssp_workflow.workflows.evaluate._pressure import PressureWorkChain
from aiida_sssp_workflow.workflows.convergence.report import ConvergenceReport


class ConvergencePressureWorkChain(_BaseConvergenceWorkChain):
    """WorkChain to converge test on pressure of input structure"""

    _PROPERTY_NAME = "pressure"
    _EVALUATE_WORKCHAIN = PressureWorkChain

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input(
            "code",
            valid_type=orm.AbstractCode,
            help="The `pw.x` code use for the `PwCalculation`.",
        )
        spec.input(
            "parallelization",
            valid_type=orm.Dict,
            required=False,
            help="The parallelization settings for the `PwCalculation` calculation.",
        )
        spec.input(
            "mpi_options",
            valid_type=orm.Dict,
            required=False,
            help="The MPI options for the `PwCalculation` calculation.",
        )

    @classmethod
    def get_builder(
        cls,
        code: orm.AbstractCode,
        pseudo: Union[Path, UpfData],
        protocol: str,
        cutoff_list: list,
        configuration: str | None = None,
        parallelization: dict | None = None,
        mpi_options: dict | None = None,
        clean_workdir: bool = True,  # clean workdir by default
    ) -> ProcessBuilder:
        """Return a builder to run this pressure convergence workchain"""
        builder = super().get_builder(pseudo, protocol, cutoff_list, configuration)

        builder.metadata.call_link_label = "convergence_pressure"
        builder.clean_workdir = orm.Bool(clean_workdir)
        builder.code = code

        if parallelization:
            builder.parallelization = orm.Dict(parallelization)
        else:
            builder.parallelization = orm.Dict()

        if mpi_options:
            builder.mpi_options = orm.Dict(mpi_options)
        else:
            builder.mpi_options = orm.Dict(get_default_mpi_options())

        return builder

    def prepare_evaluate_builder(self, ecutwfc, ecutrho):
        """Prepare input builder for running the inner pressure evaluation workchain"""
        protocol = self.protocol
        natoms = len(self.structure.sites)

        builder = self._EVALUATE_WORKCHAIN.get_builder()

        builder.clean_workdir = (
            self.inputs.clean_workdir
        )  # sync with the main workchain
        builder.pseudos = self.pseudos
        builder.structure = self.structure

        pw_parameters = {
            "SYSTEM": {
                "degauss": protocol["degauss"],
                "occupations": protocol["occupations"],
                "smearing": protocol["smearing"],
                "ecutwfc": ecutwfc,  # <-- Here set the ecutwfc
                "ecutrho": ecutrho,  # <-- Here set the ecutrho
            },
            "ELECTRONS": {
                "conv_thr": protocol["conv_thr_per_atom"] * natoms,
                "mixing_beta": protocol["mixing_beta"],
            },
            "CONTROL": {
                "calculation": "scf",
                "tstress": True,
                "disk_io": "nowf",  # not store wavefunction file to save inodes
            },
        }

        builder.kpoints_distance = orm.Float(protocol["kpoints_distance"])
        builder.metadata.call_link_label = "pressure_scf"
        builder.pw["code"] = self.inputs.code
        builder.pw["parameters"] = orm.Dict(dict=pw_parameters)
        builder.pw["parallelization"] = self.inputs.parallelization
        builder.pw["metadata"]["options"] = self.inputs.mpi_options.get_dict()

        return builder

    def run_reference(self):
        """Beside running in reference point the pressure calculation, adding process on running
        EOS at the reference cutoff, the result of EOS of reference is used in compute the residual pressure.
        """
        super().run_reference()

        protocol = self.protocol
        natoms = len(self.structure.sites)

        ecutwfc, ecutrho = self.inputs.cutoff_list[-1]
        ecutwfc, ecutrho = round(ecutwfc), round(ecutrho)

        pw_parameters = {
            "SYSTEM": {
                "degauss": protocol["degauss"],
                "occupations": protocol["occupations"],
                "smearing": protocol["smearing"],
                "ecutwfc": ecutwfc,  # <-- Here set the ecutwfc
                "ecutrho": ecutrho,  # <-- Here set the ecutrho
            },
            "ELECTRONS": {
                "conv_thr": protocol["conv_thr_per_atom"] * natoms,
                "mixing_beta": protocol["mixing_beta"],
            },
            "CONTROL": {
                "calculation": "scf",
                "disk_io": "nowf",  # not store wavefunction file to save inodes
            },
        }

        # EOS builder
        builder = _EquationOfStateWorkChain.get_builder()

        builder.metadata.call_link_label = "EOS_for_pressure_ref"
        builder.structure = self.structure
        builder.kpoints_distance = orm.Float(protocol["kpoints_distance"])
        builder.scale_count = orm.Int(protocol["scale_count"])
        builder.scale_increment = orm.Float(protocol["scale_increment"])

        # pw
        builder.pw["code"] = self.inputs.code
        builder.pw["pseudos"] = self.pseudos
        builder.pw["parameters"] = orm.Dict(dict=pw_parameters)
        builder.pw["parallelization"] = self.inputs.parallelization
        builder.pw["metadata"]["options"] = self.inputs.mpi_options.get_dict()

        running = self.submit(builder)
        self.report(
            f"launching EOS calculation for pressure convergence at reference point pk = <{running.pk}>"
        )

        self.to_context(extra_reference=running)

    def inspect_reference(self):
        """After doing the regular inspect to get the pressure results, also parse the extra reference
        compute for EOS at reference in order to get data for residual data compute.
        """
        super().inspect_reference()

        workchain = self.ctx.extra_reference
        if not workchain.is_finished_ok:
            self.logger.warning(
                f"{workchain.process_label} pk={workchain.pk} for extra reference of "
                "pressure convergence run is failed with exit_code={workchain.exit_status}."
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED.format(
                label="extra_reference"
            )

        extra_reference = self.ctx.extra_reference
        extra_reference_parameters = extra_reference.outputs.output_birch_murnaghan_fit

        V0 = extra_reference_parameters["volume0"]
        B0 = extra_reference_parameters["bulk_modulus0"]  # The unit is eV/angstrom^3
        B1 = extra_reference_parameters["bulk_deriv0"]

        self.ctx.extra_parameters = {
            "V0": orm.Float(V0),
            "B0": orm.Float(B0),
            "B1": orm.Float(B1),
        }

def _helper_get_volume_from_pressure_birch_murnaghan(P, V0, B0, B1):
    """
    Knowing the pressure P and the Birch-Murnaghan equation of state
    parameters, gets the volume the closest to V0 (relatively) that is
    such that P_BirchMurnaghan(V)=P

    retrun unit is (%)

    !! The unit of P and B0 must be compatible. We use eV/angs^3 here.
    Therefore convert P from GPa to eV/angs^3
    """
    import numpy as np

    # convert P from GPa to eV/angs^3
    P = P / 160.21766208

    # coefficients of the polynomial in x=(V0/V)^(1/3) (aside from the
    # constant multiplicative factor 3B0/2)
    polynomial = [
        3.0 / 4.0 * (B1 - 4.0),
        0,
        1.0 - 3.0 / 2.0 * (B1 - 4.0),
        0,
        3.0 / 4.0 * (B1 - 4.0) - 1.0,
        0,
        0,
        0,
        0,
        -2 * P / (3.0 * B0),
    ]
    V = min(
        [
            V0 / (x.real**3)
            for x in np.roots(polynomial)
            if abs(x.imag) < 1e-8 * abs(x.real)
        ],
        key=lambda V: abs(V - V0) / float(V0),
    )

    return abs(V - V0) / V0 * 100

def compute_xy(
    node: orm.Node,
) -> dict[str, Any]:
    """From report calculate the xy data, xs are cutoffs and ys are residual pressue from reference"""
    outgoing = node.base.links.get_outgoing()
    EOS_ref_node = outgoing.get_node_by_label('EOS_for_pressure_ref')
    extra_ref_parameters = EOS_ref_node.outputs.output_birch_murnaghan_fit.get_dict()

    V0 = extra_ref_parameters["volume0"]
    B0 = extra_ref_parameters["bulk_modulus0"]  # The unit is eV/angstrom^3
    B1 = extra_ref_parameters["bulk_deriv0"]

    report_dict = node.outputs.report.get_dict()
    report = ConvergenceReport.construct(**report_dict)

    reference_node = orm.load_node(report.reference.uuid)
    output_parameters_r: orm.Dict = reference_node.outputs.output_parameters
    y_ref = output_parameters_r['hydrostatic_stress']

    xs = []
    ys = []
    for node_point in report.convergence_list:
        if node_point.exit_status != 0:
            # TODO: log to a warning file for where the node is not finished_okay
            continue
        
        x = node_point.wavefunction_cutoff
        xs.append(x)

        node = orm.load_node(node_point.uuid)
        output_parameters_p: orm.Dict = node.outputs.output_parameters

        y_p = output_parameters_p["hydrostatic_stress"]

        # calculate the diff
        absolute_diff = abs(y_p - y_ref)
        relative_diff = _helper_get_volume_from_pressure_birch_murnaghan(
            absolute_diff, V0, B0, B1,
        )
        
        y = relative_diff
        ys.append(y)

    return {
        'x': xs,
        'y': ys,
        'metadata': {
            'unit': '%',
        }
    }

