# -*- coding: utf-8 -*-
"""
Convergence test on pressure of a given pseudopotential
"""

from pathlib import Path
from typing import Union

from aiida import orm
from aiida.engine import ProcessBuilder
from aiida_pseudo.data.pseudo import UpfData

from aiida_sssp_workflow.utils import get_default_mpi_options
from aiida_sssp_workflow.workflows.convergence._base import _BaseConvergenceWorkChain
from aiida_sssp_workflow.workflows.evaluate._eos import _EquationOfStateWorkChain
from aiida_sssp_workflow.workflows.evaluate._pressure import PressureWorkChain


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
        pseudo: Union[Path, UpfData],
        protocol: str,
        cutoff_list: list,
        configuration: str,
        code: orm.AbstractCode,
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
