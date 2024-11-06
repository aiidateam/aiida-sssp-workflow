# -*- coding: utf-8 -*-
"""
Convergence test on pressure of a given pseudopotential
"""

from pathlib import Path
from typing import Union, Any, Tuple

from aiida import orm
from aiida.engine import ProcessBuilder, append_
from aiida_pseudo.data.pseudo import UpfData

from aiida_sssp_workflow.utils import get_default_mpi_options
from aiida_sssp_workflow.utils.element import MAGNETIC_ELEMENTS, UNSUPPORTED_ELEMENTS
from aiida_sssp_workflow.utils.protocol import get_protocol
from aiida_sssp_workflow.utils.pseudo import extract_pseudo_info
from aiida_sssp_workflow.utils.structure import get_standard_structure, scale_structure
from aiida_sssp_workflow.workflows import SelfCleanWorkChain
from aiida_sssp_workflow.workflows.evaluate._magnetization import MagnetizationWorkChain
from aiida_sssp_workflow.workflows.list_run.report import ListReport, PointRunReportEntry


class MagnetizationChangeWorkChain(SelfCleanWorkChain):
    """Workflow to get magnetization on different volumes"""

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
        spec.input(
            "pseudo",
            valid_type=UpfData,
            required=True,
            help="Pseudopotential to be verified",
        )
        spec.input(
            "protocol",
            valid_type=orm.Str,
            required=True,
            help="The calculation protocol to use for the workchain.",
        )
        spec.input(
            "lattice_constant_scale_list",
            valid_type=orm.List,
            required=True,
            help="The list of lattice constant for the structure to test, in Angstrom ",
        )
        spec.input(
            "wavefunction_cutoff",
            valid_type=orm.Int,
            required=True,
            help="The wavefunction cutoff.",
        )
        spec.input(
            "charge_density_cutoff",
            valid_type=orm.Int,
            required=True,
            help="The charge density cutoff.",
        )
        spec.input(
            "configuration",
            valid_type=orm.Str,
            required=True,
            help="The configuration to use for the workchain, can be DC/BCC/FCC/SC.",
        )

        spec.outline(
            cls._setup,
            cls.run_list,
            cls.inspect_list,
            cls._finalize,
        )

        spec.output(
            "report",
            valid_type=orm.Dict,
            required=True,
            help="The output report of convergence verification, it is a dict contains the full information of convergence test, the mapping of cutoffs to the UUID of the evaluation workchain etc.",
        )
        spec.output(
            "extra_report",
            valid_type=orm.Dict,
            required=False,
            help="Extra report for specific convergence test or additional reference calculation, it is currently only used in pressure convergence workflow.",
        )
        spec.output(
            "configuration",
            valid_type=orm.Str,
            required=True,
            help="The configuration used for the convergence.",
        )
        spec.output(
            "structure",
            valid_type=orm.StructureData,
            required=True,
            help="The structure used for the convergence.",
        )
        spec.output(
            "success_rate",
            valid_type=orm.Float,
            required=False,
            help="The success rate of convergence tests.",
        )

        spec.exit_code(
            401,
            "ERROR_REFERENCE_CALCULATION_FAILED",
            message="The reference calculation failed.",
        )
        spec.exit_code(
            402,
            "ERROR_SUB_PROCESS_FAILED",
            message="The sub process for `{label}` did not finish successfully.",
        )
        spec.exit_code(
            404,
            "ERROR_UNSUPPORTED_ELEMENT",
            message="The PP of element {element} is not yet supported to be verified.",
        )
        spec.exit_code(
            811,
            "ERROR_NOT_ENOUGH_CONVERGENCE_TEST",
            message="Not enough convergence test, the success rate is {rate:.1%}, expect > 80%",
        )
        spec.exit_code(
            1801,
            "WARNING_REFERENCE_CALCULATION_FAILED",
            message="The reference calculation failed with warning exit_status {warning_exit_status}",
        )

    @classmethod
    def get_builder(
        cls,
        code: orm.AbstractCode,
        pseudo: Union[Path, UpfData],
        cutoffs: Tuple[int, int],
        protocol: str,
        scale_list: list,
        configuration: str | None = None,
        parallelization: dict | None = None,
        mpi_options: dict | None = None,
        clean_workdir: bool = True,  # clean workdir by default
    ) -> ProcessBuilder:
        """Return a builder to run this pressure convergence workchain"""
        builder = super().get_builder()

        builder.protocol = orm.Str(protocol)

        builder.configuration = orm.Str(configuration)
        configuration_name = configuration

        # Set the default label and description
        # The default label is set to be the base file name of PP
        # The description include which configuration and which protocol is using.
        builder.metadata.label = (
            pseudo.filename if isinstance(pseudo, UpfData) else pseudo.name
        )
        builder.metadata.description = (
            f"Run on protocol '{protocol}' and configuration '{configuration_name}'"
        )

        if isinstance(pseudo, Path):
            builder.pseudo = UpfData.get_or_create(pseudo)
        else:
            builder.pseudo = pseudo

        builder.lattice_constant_scale_list = orm.List(list=scale_list)
        builder.wavefunction_cutoff = cutoffs[0]
        builder.charge_density_cutoff = cutoffs[1]
        builder.clean_workdir = orm.Bool(clean_workdir)

        builder.metadata.call_link_label = "magnetization_scale"
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

    def _setup(self):
        """Setup"""
        # set pseudo
        pseudo_info = extract_pseudo_info(
            self.inputs.pseudo.get_content(),
        )
        self.ctx.element = pseudo_info.element
        self.ctx.pseudos = {self.ctx.element: self.inputs.pseudo}

        # set structure
        if self.ctx.element not in MAGNETIC_ELEMENTS:
            return self.exit_codes.ERROR_UNSUPPORTED_ELEMENT.format(
                element=self.ctx.element
            )

        self.ctx.configuration = self.inputs.configuration

        structure = get_standard_structure(
            self.ctx.element, configuration=self.ctx.configuration
        )
        kind_name = structure.get_kind_names()[0]

        mag_structure = orm.StructureData(cell=structure.cell, pbc=structure.pbc)

        for _, site in enumerate(structure.sites):
            mag_structure.append_atom(position=site.position, symbols=kind_name)

        self.ctx.structure = mag_structure

        self.out("configuration", self.ctx.configuration)
        self.out("structure", self.ctx.structure)

        # set protocol
        self.ctx.protocol = get_protocol(
            category="magnetization", name=self.inputs.protocol.value
        )

    def prepare_evaluate_builder(self, scale):
        """Prepare input builder for running the inner pressure evaluation workchain"""
        protocol = self.ctx.protocol
        natoms = len(self.ctx.structure.sites)

        ecutwfc = self.inputs.wavefunction_cutoff.value
        ecutrho = self.inputs.charge_density_cutoff.value

        structure = self.ctx.structure
        kind_name = structure.get_kind_names()[0]

        # scale structure

        builder = MagnetizationWorkChain.get_builder()

        builder.clean_workdir = (
            self.inputs.clean_workdir
        )  # sync with the main workchain
        builder.pseudos = self.ctx.pseudos
        builder.structure = scale_structure(structure, orm.Float(scale))


        pw_parameters = {
            "SYSTEM": {
                "degauss": protocol["degauss"],
                "occupations": protocol["occupations"],
                "smearing": protocol["smearing"],
                "ecutwfc": ecutwfc,  # <-- Here set the ecutwfc
                "ecutrho": ecutrho,  # <-- Here set the ecutrho
                "nspin": 2,
                "starting_magnetization": {kind_name: 0.2},
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

    def run_list(self):
        """
        run on all other evaluation sample points
        """
        for scale in self.inputs.lattice_constant_scale_list:
            # The last one is reference
            builder = self.prepare_evaluate_builder(scale)

            if scale < 0:
                sign = "minus"
            else:
                sign = "plus"

            v = str(abs(scale)).replace('.', '_')
            # Add link to the called workchain by 'scale'
            builder.metadata.call_link_label = f"scale_{sign}_{v}"

            running = self.submit(builder)
            self.report(
                f"launching fix scale={scale} [scale={scale}] {running.process_label}<{running.pk}>"
            )

            # add scale as extras of running node
            running.base.extras.set("lattice_constant_scale", scale)

            self.to_context(children_convergence=append_(running))

    def inspect_list(self):
        reports = []
        for child in self.ctx.children_convergence:
            # The convergence runs are evaluate workchain inhrited from _BaseEvaluateWorkChain
            # So the results are sure to have scale as outputs.
            scale = child.base.extras.get("lattice_constant_scale")

            if child.exit_status == 0:
                self.report(
                    f"{child.process_label} pk={child.pk} finished successfully with cutoffs scale={scale}."
                )
            else:
                self.report(
                    f"{child.process_label} pk={child.pk} failed with exit status {child.exit_status}"
                )

            _report = {
                "uuid": child.uuid,
                "x": scale,
                "exit_status": child.exit_status,
            }

            reports.append(_report)

        try:
            reports = ListReport.build(
                reports,
            )
            self.report("Convergence report is validated.")
        except Exception as e:
            self.report(f"Convergence report is not validated: {e}")
            raise e
        else:
            self.out(
                "report",
                orm.Dict(dict=reports.model_dump()).store(),
            )

        # Pass to ctx for _finalize
        self.ctx.reports = reports.report_list

    def _finalize(self):
        """Construct a summary report from the list report.
        It will contains the analysis of the convergence test, such as the ratio of succuessful runs.
        """
        # Do two things:
        # 1. if the reference raise a warning, happened in EOS convergence where the birch marnaghan fit failed for Hg. The exit code -> 1801
        # 2. if the convergence points don't have enough success (must > 80%), exit code -> 811

        total = len(self.ctx.reports)
        success_count = 0
        for i in self.ctx.reports:
            if i.exit_status != 0:
                continue

            success_count += 1

        # success_count = len([i for i in self.ctx.reports if i.exit_status == 0])


        rate = round(float(success_count / total), 3)
        if rate < 0.8:
            return self.exit_codes.ERROR_NOT_ENOUGH_CONVERGENCE_TEST.format(rate=rate)

        self.out("success_rate", orm.Float(rate).store())


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
    EOS_ref_node = outgoing.get_node_by_label("EOS_for_pressure_ref")
    extra_ref_parameters = EOS_ref_node.outputs.output_birch_murnaghan_fit.get_dict()

    V0 = extra_ref_parameters["volume0"]
    B0 = extra_ref_parameters["bulk_modulus0"]  # The unit is eV/angstrom^3
    B1 = extra_ref_parameters["bulk_deriv0"]

    report_dict = node.outputs.report.get_dict()
    report = ConvergenceReport.construct(**report_dict)

    reference_node = orm.load_node(report.reference.uuid)
    output_parameters_r: orm.Dict = reference_node.outputs.output_parameters
    y_ref = output_parameters_r["hydrostatic_stress"]

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
        diff = y_p - y_ref
        relative_diff = _helper_get_volume_from_pressure_birch_murnaghan(
            diff,
            V0,
            B0,
            B1,
        )

        y = relative_diff
        ys.append(y)

    return {
        "xs": xs,
        "ys": ys,
        "metadata": {
            "unit": "%",
        },
    }
