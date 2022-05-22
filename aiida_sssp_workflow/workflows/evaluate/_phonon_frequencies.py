# -*- coding: utf-8 -*-
"""
WorkChain calculate phonon frequencies at Gamma
"""
from aiida import orm
from aiida.common import NotExistentAttributeError
from aiida.engine import ToContext, while_
from aiida.plugins import DataFactory, WorkflowFactory

from . import _BaseEvaluateWorkChain

PwBaseWorkflow = WorkflowFactory("quantumespresso.pw.base")
PhBaseWorkflow = WorkflowFactory("quantumespresso.ph.base")
UpfData = DataFactory("pseudo.upf")


class PhononFrequenciesWorkChain(_BaseEvaluateWorkChain):
    """WorkChain to calculate cohisive energy of input structure"""

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        # yapf: disable
        super().define(spec)
        spec.expose_inputs(PwBaseWorkflow, namespace='scf', include=['metadata', 'pw', 'kpoints_distance'])
        spec.expose_inputs(PhBaseWorkflow, namespace='phonon', exclude=['ph.parent_folder'])

        spec.outline(
            cls.setup,
            while_(cls.is_pw_not_ready_for_ph)(
                cls.run_scf,
                cls.inspect_scf,
            ),
            cls.run_ph,
            cls.inspect_ph,
            cls.finalize,
        )
        spec.output('output_parameters', valid_type=orm.Dict, required=True,
                    help='The output parameters include phonon frequencies.')
        spec.exit_code(211, 'ERROR_NO_REMOTE_FOLDER',
                    message='The remote folder node not exist')
        spec.exit_code(202, 'ERROR_SUB_PROCESS_FAILED_PH',
                    message='The `PhBaseWorkChain` sub process failed.')
        # yapf: enable

    def setup(self):
        self.ctx.not_ready_for_ph = True

    def is_pw_not_ready_for_ph(self):
        """used to check if the remote folder is not empty, otherwise rerun pw with caching off"""
        return self.ctx.not_ready_for_ph

    def run_scf(self):
        """
        set the inputs and submit scf
        """
        inputs = self.exposed_inputs(PwBaseWorkflow, namespace="scf")

        running = self.submit(PwBaseWorkflow, **inputs)
        self.report(f"Running pw calculation pk={running.pk}")
        return ToContext(workchain_scf=running)

    def inspect_scf(self):
        """inspect the result of scf calculation if fail do not continue."""
        workchain = self.ctx.workchain_scf

        if not workchain.is_finished_ok:
            self.logger.warning(
                f"PwBaseWorkChain failed with exit status {workchain.exit_status}"
            )
            # set condition to False to break loop
            self.ctx.not_ready_for_ph = False

            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_SCF

        try:
            remote_folder = self.ctx.scf_remote_folder = workchain.outputs.remote_folder

            if not remote_folder.is_empty:
                # when the remote_folder is not empty we regard it is ready for ph
                # even if the subsequent ph is successful
                self.ctx.not_ready_for_ph = False
            else:
                # set all same node to caching off and rerun
                pw_node = [
                    c for c in workchain.called if isinstance(c, orm.CalcJobNode)
                ][0]
                all_same_nodes = pw_node.get_all_same_nodes()
                for node in all_same_nodes:
                    node.delete_extra("_aiida_hash")

        except NotExistentAttributeError:
            # set condition to False to break loop
            self.ctx.not_ready_for_ph = False
            return self.exit_codes.ERROR_NO_REMOTE_FOLDER

        self.ctx.ecutwfc = workchain.inputs.pw.parameters["SYSTEM"]["ecutwfc"]
        self.ctx.ecutrho = workchain.inputs.pw.parameters["SYSTEM"]["ecutrho"]

        self.ctx.calc_time = workchain.outputs.output_parameters["wall_time_seconds"]

    def run_ph(self):
        """
        set the inputs and submit ph calculation to get quantities for phonon evaluation
        """

        inputs = self.exposed_inputs(PhBaseWorkflow, namespace="phonon")
        inputs["ph"]["parent_folder"] = self.ctx.scf_remote_folder

        running = self.submit(PhBaseWorkflow, **inputs)
        self.report(f"Running ph calculation pk={running.pk}")
        return ToContext(workchain_ph=running)

    def inspect_ph(self):
        """inspect the result of ph calculation."""
        workchain = self.ctx.workchain_ph

        if not workchain.is_finished_ok:
            self.report(
                f"PhBaseWorkChain for pressure evaluation failed with exit status {workchain.exit_status}"
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_PH

        self.ctx.calc_time += workchain.outputs.output_parameters["wall_time_seconds"]
        output_parameters = workchain.outputs.output_parameters.get_dict()
        output_parameters.update(
            {
                "total_calc_time": self.ctx.calc_time,
                "time_unit": "s",
            }
        )
        self.out("output_parameters", orm.Dict(dict=output_parameters).store())

    def finalize(self):
        """set ecutwfc and ecutrho"""
        self.out("ecutwfc", orm.Int(self.ctx.ecutwfc).store())
        self.out("ecutrho", orm.Int(self.ctx.ecutrho).store())
