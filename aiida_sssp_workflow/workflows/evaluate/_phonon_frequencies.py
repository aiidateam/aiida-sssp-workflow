# -*- coding: utf-8 -*-
"""
WorkChain calculate phonon frequencies at Gamma
"""
from aiida import orm
from aiida.common import NotExistentAttributeError
from aiida.engine import ToContext, while_
from aiida.plugins import DataFactory, WorkflowFactory

from aiida_sssp_workflow.workflows.common import clean_workdir, operate_calcjobs

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
            while_(cls.should_run_ph)(
                cls.run_scf,
                cls.inspect_scf,
                cls.run_ph,
                cls.inspect_ph,
            ),
            cls.finalize,
        )
        spec.output('output_parameters', valid_type=orm.Dict, required=True,
                    help='The output parameters include phonon frequencies.')
        spec.exit_code(211, 'ERROR_NO_REMOTE_FOLDER_OUTPUT_OF_SCF',
                    message='The remote folder node not exist')
        spec.exit_code(202, 'ERROR_SUB_PROCESS_FAILED_PH',
                    message='The `PhBaseWorkChain` sub process failed.')
        spec.exit_code(203, 'ERROR_SUB_PROCESS_FAILED_SCF',
                    message='The `PwBaseWorkChain` sub process failed.')
        # yapf: enable

    def setup(self):
        self.ctx.should_run_ph = True

        # The list of nodes that are invalided the caching of the node and re-run scf calculation. When the remote folder of parent calculation is empty, the caching is invalid, but since invalid caching is applied to all_same_nodes, we need to keep track of the invalid nodes and bring caching valid back to the node so the next run will still using the caching, otherwise the next parent calculation will be re-run again, which is not what we want.
        # Warning!! experimental feature and will potentially cause "race condition" like behavior between workflows.
        self.ctx.cache_invalid_list = []

    def should_run_ph(self):
        """will be True if ph calculation is successful, otherwise will rerun scf
        The ph calculation can be failed due to the following reasons:
        - the scf calculation is not successful
        - the ph calculation is not successful
        - the scf calculation is successful but the remote folder is not found and the ph can not be submitted
        """
        return self.ctx.should_run_ph

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

        if workchain.is_finished:
            self._disable_cache(workchain)

        if not workchain.is_finished_ok:
            self.logger.warning(
                f"PwBaseWorkChain failed with exit status {workchain.exit_status}"
            )

            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_SCF

        try:
            remote_folder = self.ctx.scf_remote_folder = workchain.outputs.remote_folder

            self.report(
                f"Is remote folder cleaned? {remote_folder.extras.get('cleaned', False)}"
            )
        except NotExistentAttributeError:
            # set condition to False to break loop
            return self.exit_codes.ERROR_NO_REMOTE_FOLDER_OUTPUT_OF_SCF

        # When change caching below, also check _bands and _caching_wise_bandes
        # the remote folder is set so can bring the caching invalid back to the node
        while self.ctx.cache_invalid_list:
            node_pk = self.ctx.cache_invalid_list.pop()
            node = orm.load_node(node_pk)
            node.is_valid_cache = True

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
            # if the remote folder is empty, invalid the caching of the node and re-run scf calculation
            is_cleaned = self.ctx.scf_remote_folder.base.extras.get("cleaned", False)
            if is_cleaned:
                self.logger.warning(
                    f"PhBaseWorkChain failed because the remote folder is empty with exit status {workchain.exit_status}, invalid the caching of the node and re-run scf calculation."
                )
                # invalid the caching of the node and re-run scf calculation
                workchain_scf = self.ctx.workchain_scf
                pw_node = [
                    c for c in workchain_scf.called if isinstance(c, orm.CalcJobNode)
                ][0]
                all_same_nodes = pw_node.base.caching.get_all_same_nodes()
                for node in all_same_nodes:
                    node.is_valid_cache = False
                    self.ctx.cache_invalid_list.append(node.pk)
                return
            else:
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

        self.ctx.should_run_ph = False

    def finalize(self):
        """set ecutwfc and ecutrho"""
        self.out("ecutwfc", orm.Int(self.ctx.ecutwfc).store())
        self.out("ecutrho", orm.Int(self.ctx.ecutrho).store())

        if self.inputs.clean_workdir.value is True:
            cleaned_calcs = operate_calcjobs(
                self.node, operator=clean_workdir, all_same_nodes=False
            )

            if cleaned_calcs:
                self.report(
                    f"cleaned remote folders of calculations: {' '.join(map(str, cleaned_calcs))}"
                )

        else:
            self.report(f"{type(self)}: remote folders will not be cleaned")
