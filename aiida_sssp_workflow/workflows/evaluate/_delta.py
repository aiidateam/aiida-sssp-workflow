# -*- coding: utf-8 -*-
"""
WorkChain calculate the delta for certain pseudopotential
"""
from aiida import orm
from aiida.engine import WorkChain
from aiida.plugins import DataFactory

from aiida_sssp_workflow.calculations.calculate_delta import delta_analyze
from aiida_sssp_workflow.utils import update_dict
from aiida_sssp_workflow.workflows.evaluate._eos import _EquationOfStateWorkChain

UpfData = DataFactory("pseudo.upf")


class DeltaWorkChain(WorkChain):
    """WorkChain calculate the bands for certain pseudopotential"""

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        # yapf: disable
        super().define(spec)
        spec.input('code', valid_type=orm.Code,
                    help='The `pw.x` code use for the `PwCalculation`.')
        spec.input_namespace('pseudos', valid_type=UpfData, dynamic=True,
                    help='A mapping of `UpfData` nodes onto the kind name to which they should apply.')
        spec.input('structure', valid_type=orm.StructureData,
                    help='Ground state structure which the verification perform')
        spec.input('element', valid_type=orm.Str,
                    help='element')
        spec.input('configuration', valid_type=orm.Str,
                    help='Configuration name of structure, BCC, FCC, SC and Diamond and name for oxides')
        spec.input('pw_base_parameters', valid_type=orm.Dict,
                    help='parameters for pwscf of calculation.')
        spec.input('ecutwfc', valid_type=orm.Float,
                    help='The ecutwfc set for both atom and bulk calculation. Please also set ecutrho if ecutwfc is set.')
        spec.input('ecutrho', valid_type=orm.Float,
                    help='The ecutrho set for both atom and bulk calculation.  Please also set ecutwfc if ecutrho is set.')
        spec.input('kpoints_distance', valid_type=orm.Float,
                    help='Kpoints distance setting for bulk energy calculation.')
        spec.input('scale_count', valid_type=orm.Int, default=lambda: orm.Int(7),
                    help='The number of points to compute for the equation of state.')
        spec.input('scale_increment', valid_type=orm.Float, default=lambda: orm.Float(0.02),
                    help='The relative difference between consecutive scaling factors.')
        spec.input('options', valid_type=orm.Dict, required=False,
                    help='Optional `options` to use for the `PwCalculations`.')
        spec.input('parallelization', valid_type=orm.Dict, required=False,
                    help='Parallelization options for the `PwCalculations`.')
        spec.input('clean_workdir', valid_type=orm.Bool, default=lambda: orm.Bool(False),
                    help='If `True`, work directories of all called calculation will be cleaned at the end of execution.')
        spec.outline(
            cls.setup_base_parameters,
            cls.validate_structure,
            cls.setup_code_resource_options,
            cls.run_eos,
            cls.inspect_eos,
            cls.finalize,
        )
        spec.expose_outputs(_EquationOfStateWorkChain, namespace='eos',
                    namespace_options={'help': f'volume_energy and birch_murnaghan_fit result from EOS.'})

        spec.output('output_parameters', required=True,
                    help='The output of delta factor and other measures to describe the accuracy of EOS compare '
                        ' with the AE equation of state.')
        spec.exit_code(201, 'ERROR_SUB_PROCESS_FAILED_BANDS',
                    message='The `PwBandsWorkChain` sub process failed.')
        # yapf: enable

    def setup_base_parameters(self):
        """Input validation"""
        pw_parameters = self.inputs.pw_base_parameters.get_dict()

        parameters = {
            "SYSTEM": {
                "ecutwfc": self.inputs.ecutwfc,
                "ecutrho": self.inputs.ecutrho,
            },
        }
        pw_parameters = update_dict(pw_parameters, parameters)

        self.ctx.pw_parameters = pw_parameters

        self.ctx.kpoints_distance = self.inputs.kpoints_distance

    def validate_structure(self):
        """doc"""
        self.ctx.pseudos = self.inputs.pseudos

    def setup_code_resource_options(self):
        """
        setup resource options and parallelization for `PwCalculation` from inputs
        """
        if "options" in self.inputs:
            self.ctx.options = self.inputs.options.get_dict()
        else:
            from aiida_sssp_workflow.utils import get_default_options

            self.ctx.options = get_default_options(with_mpi=True)

        if "parallelization" in self.inputs:
            self.ctx.parallelization = self.inputs.parallelization.get_dict()
        else:
            self.ctx.parallelization = {}

        self.report(f"resource options set to {self.ctx.options}")
        self.report(f"parallelization options set to {self.ctx.parallelization}")

    def _get_inputs(self):
        inputs = {
            "structure": self.inputs.structure,
            "kpoints_distance": self.inputs.kpoints_distance,
            "scale_count": self.inputs.scale_count,
            "scale_increment": self.inputs.scale_increment,
            "metadata": {"call_link_label": f"EOS"},
            "scf": {
                "pw": {
                    "code": self.inputs.code,
                    "pseudos": self.ctx.pseudos,
                    "parameters": orm.Dict(dict=self.ctx.pw_parameters),
                    "metadata": {"options": self.ctx.options},
                    "parallelization": orm.Dict(dict=self.ctx.parallelization),
                },
            },
        }

        return inputs

    def run_eos(self):
        """run eos workchain"""
        self.report(f"{self.ctx.pw_parameters}")

        inputs = self._get_inputs()

        future = self.submit(_EquationOfStateWorkChain, **inputs)
        self.report(f"launching _EquationOfStateWorkChain<{future.pk}>")

        self.to_context(**{f"eos": future})

    def inspect_eos(self):
        """Inspect the results of _EquationOfStateWorkChain"""
        workchain = self.ctx.eos

        if not workchain.is_finished_ok:
            self.report(
                f"_EquationOfStateWorkChain failed with exit status {workchain.exit_status}"
            )

        self.out_many(
            self.exposed_outputs(
                workchain,
                _EquationOfStateWorkChain,
                namespace="eos",
            )
        )

    def finalize(self):
        """result"""
        output_bmf = self.outputs["eos"].get("output_birch_murnaghan_fit")

        V0 = orm.Float(output_bmf["volume0"])

        inputs = {
            "element": self.inputs.element,
            "configuration": self.inputs.configuration,
            "V0": V0,
            "B0": orm.Float(output_bmf["bulk_modulus0"]),
            "B1": orm.Float(output_bmf["bulk_deriv0"]),
        }

        self.out(f"output_parameters", delta_analyze(**inputs))
