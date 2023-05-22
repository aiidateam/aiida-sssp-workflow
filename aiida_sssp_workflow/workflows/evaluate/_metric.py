# -*- coding: utf-8 -*-
"""
WorkChain calculate the metric (from EOS results) for certain pseudopotential
"""
from aiida import orm
from aiida.plugins import DataFactory
from plumpy import ToContext

from aiida_sssp_workflow.calculations.calculate_metric import metric_analyze
from aiida_sssp_workflow.workflows.evaluate._eos import _EquationOfStateWorkChain

from . import _BaseEvaluateWorkChain

UpfData = DataFactory("pseudo.upf")


class MetricWorkChain(_BaseEvaluateWorkChain):
    """WorkChain calculate the bands for certain pseudopotential"""

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        # yapf: disable
        super().define(spec)
        spec.expose_inputs(_EquationOfStateWorkChain, namespace='eos')
        spec.input('element', valid_type=orm.Str,
                    help='element')
        spec.input('configuration', valid_type=orm.Str,
                    help='Configuration name of structure, determine with AE reference data to use.')

        spec.outline(
            cls.run_eos,
            cls.inspect_eos,
            cls.finalize,
        )
        spec.expose_outputs(_EquationOfStateWorkChain, namespace='eos',
                    namespace_options={'help': f'volume_energy and birch_murnaghan_fit result from EOS.'})

        spec.output('output_parameters', required=True,
                    help='The output of metric factor measures to describe the precision of EOS compare '
                        ' with the AE equation of state.')

        # yapf: enable

    def _get_inputs(self):
        inputs = self.exposed_inputs(_EquationOfStateWorkChain, namespace="eos")

        return inputs

    def run_eos(self):
        """run eos workchain"""
        inputs = self._get_inputs()

        running = self.submit(_EquationOfStateWorkChain, **inputs)
        self.report(f"launching _EquationOfStateWorkChain<{running.pk}>")

        return ToContext(eos=running)

    def inspect_eos(self):
        """Inspect the results of _EquationOfStateWorkChain"""
        workchain = self.ctx.eos

        if not workchain.is_finished_ok:
            self.logger.warning(
                f"_EquationOfStateWorkChain failed with exit status {workchain.exit_status}"
            )

        self.ctx.ecutwfc = workchain.inputs.pw.parameters["SYSTEM"]["ecutwfc"]
        self.ctx.ecutrho = workchain.inputs.pw.parameters["SYSTEM"]["ecutrho"]

        self.out_many(
            self.exposed_outputs(
                workchain,
                _EquationOfStateWorkChain,
                namespace="eos",
            )
        )

    def finalize(self):
        """result"""
        # set ecutwfc and ecutrho
        self.out("ecutwfc", orm.Int(self.ctx.ecutwfc).store())
        self.out("ecutrho", orm.Int(self.ctx.ecutrho).store())

        output_bmf = self.outputs["eos"].get("output_birch_murnaghan_fit")

        if output_bmf is not None:
            inputs = {
                "element": self.inputs.element,
                "configuration": self.inputs.configuration,
                "V0": orm.Float(output_bmf["volume0"]),
                "B0": orm.Float(output_bmf["bulk_modulus0"]),
                "B1": orm.Float(output_bmf["bulk_deriv0"]),
                "natoms": orm.Int(output_bmf["num_of_atoms"]),
            }

            output_parameters = metric_analyze(**inputs)
            self.out("output_parameters", output_parameters)
        else:
            self.report(
                "birch_murnaghan_fit result not found in eos workchain outputs, fit failed."
            )
            output_parameters = orm.Dict(dict={})
            self.out("output_parameters", output_parameters.store())
