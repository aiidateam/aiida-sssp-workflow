from aiida.engine import WorkChain
from aiida.orm import Bool, CalcJobNode

from aiida_sssp_workflow.workflows.common import clean_workdir, operate_calcjobs


class SelfCleanWorkChain(WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input(
            "clean_workdir",
            valid_type=Bool,
            default=lambda: Bool(True),
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

        # The clen_wordir is modified to not clean the remote folder of the caching node.
        cleaned_calcs = operate_calcjobs(
            self.node, operator=clean_workdir, all_same_nodes=False
        )

        if cleaned_calcs:
            self.report(
                f"cleaned remote folders of calculations: {' '.join(map(str, cleaned_calcs))}"
            )

        # This is not turned on since it will make all finished workflow not cacheable.
        # I need to find a way to properly work around the parent_folder empty issue.
        # cache_invalid_calcs = operate_calcjobs(
        #     self.node, operator=invalid_cache, all_same_nodes=False
        # )

        # if cache_invalid_calcs:
        #     self.report(
        #         f"Invalid cache of cached calculations: {' '.join(map(str, cache_invalid_calcs))}"
        #     )
