from aiida.engine import WorkChain
from aiida.orm import Bool

from aiida_sssp_workflow.workflows.common import clean_workdir, operate_calcjobs


class SelfCleanWorkChain(WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input(
            "clean_workdir",
            valid_type=Bool,
            required=True,
            help="If `True`, work directories of all called non-cached calculations will be cleaned"
            " at the end of execution, and cached calculations will invalid from cache.",
        )

    def on_terminated(self):
        """Clean the working directories of all child calculations if `clean_worchain=True` in the inputs.
        If clean, important to also invalid calcjob nodes for caching. Since otherwise there will be chance
        that not the caching node but cleaned node used for caching in bands and phonon that will lead to the
        parent_folder empty issue.
        In big verification workflow, only _caching workflow is not process clean step at last but purge remote
        folder after the whole verfication workflow finished.
        hardcode the invalid_caching.
        """
        super().on_terminated()

        if not self.inputs.clean_workdir.value:
            self.report(f"{type(self)}: remote folders will not be cleaned")
            return

        try:
            cleaned_calcs = operate_calcjobs(
                self.node, operator=clean_workdir, all_same_nodes=False
            )
        except RuntimeError as exc:
            self.logger.warning(
                f"clean remote workdir folder {self.inputs.clean_workir} failed: {exc}"
            )
        else:
            self.report(
                f"cleaned remote folders of calculations: {' '.join(map(str, cleaned_calcs))}"
            )
