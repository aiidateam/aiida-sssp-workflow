# -*- coding: utf-8 -*-
"""
A calcfunctian create_isolate_atom
Create the structure of isolate atom
"""

from aiida import orm
from aiida.engine import ToContext, calcfunction
from aiida.plugins import DataFactory, WorkflowFactory

from . import _BaseEvaluateWorkChain

PwBaseWorkChain = WorkflowFactory("quantumespresso.pw.base")
UpfData = DataFactory("pseudo.upf")


@calcfunction
def helper_get_hydrostatic_stress(output_trajectory, output_parameters):
    """
    doc
    """
    import numpy as np

    output_stress = output_trajectory.get_array("stress")[0]
    stress_unit = output_parameters["stress_units"]
    hydrostatic_stress = np.trace(output_stress) / 3.0
    return orm.Dict(
        dict={
            "stress_unit": stress_unit,
            "hydrostatic_stress": hydrostatic_stress,
        }
    )


class PressureWorkChain(_BaseEvaluateWorkChain):
    """WorkChain to calculate cohisive energy of input structure"""

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        # yapf: disable
        super().define(spec)
        spec.input('structure', valid_type=orm.StructureData,
                    help='Ground state structure which the verification perform')
        spec.input_namespace('pseudos', valid_type=UpfData, dynamic=True,
                    help='A mapping of `UpfData` nodes onto the kind name to which they should apply.')
        spec.expose_inputs(PwBaseWorkChain, exclude=["pw.structure", "pw.pseudos"])

        spec.outline(
            cls.run_scf,
            cls.inspect_scf,
            cls.finalize,
        )
        spec.output('output_parameters', valid_type=orm.Dict, required=True,
                    help='The output parameters include pressure of the structure.')

        # yapf: enable

    def run_scf(self):
        """
        set the inputs and submit scf
        """
        inputs = self.exposed_inputs(PwBaseWorkChain)
        inputs["pw"]["structure"] = self.inputs.structure
        inputs["pw"]["pseudos"] = self.inputs.pseudos

        running = self.submit(PwBaseWorkChain, **inputs)
        self.report(f"Running pw scf calculation pk={running.pk}")

        return ToContext(workchain_scf=running)

    def inspect_scf(self):
        """inspect the result of scf calculation."""
        workchain = self.ctx.workchain_scf

        if workchain.is_finished:
            self._disable_cache(workchain)

        if not workchain.is_finished_ok:
            self.report(
                f"PwBaseWorkChain for pressure evaluation failed with exit status {workchain.exit_status}"
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_SCF

        output_trajectory = workchain.outputs.output_trajectory
        output_parameters = workchain.outputs.output_parameters

        # Return the output parameters of current workchain
        output_parameters = helper_get_hydrostatic_stress(
            output_trajectory, output_parameters
        )

        self.out("output_parameters", output_parameters)

    def finalize(self):
        """set ecutwfc and ecutrho"""
