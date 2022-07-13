from aiida import orm, plugins
from aiida.common.datastructures import CalcInfo, CodeInfo
from aiida.common.folders import Folder
from aiida.engine import CalcJob
from aiida.engine.processes.process_spec import CalcJobProcessSpec

UpfData = plugins.DataFactory("pseudo.upf")


class Ld1Calculation(CalcJob):
    @classmethod
    def define(cls, spec: CalcJobProcessSpec):
        """Define the specification"""

        super().define(spec)
        # atom_info part read from configuration cards
        spec.input("parameters", valid_type=orm.SinglefileData)
        spec.output("output_pseudo", valid_type=UpfData)

        spec.inputs["metadata"]["options"]["input_filename"].default = "aiida.in"
        spec.inputs["metadata"]["options"]["output_filename"].default = "aiida.out"
        spec.inputs["metadata"]["options"]["parser_name"].default = "sssp.pseudo.ld1"
        spec.inputs["metadata"]["options"]["resources"].default = {
            "num_machines": 1,
            "num_mpiprocs_per_machine": 1,
        }

    def prepare_for_submission(self, folder: Folder) -> CalcInfo:
        """Prepare the calculation for submission"""

        with folder.open(self.options.input_filename, "w", encoding="utf8") as handle:
            handle.write(inp_str)

        codeinfo = CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.stdin_name = self.options.input_filename
        codeinfo.stdout_name = self.options.output_filename

        calcinfo = CalcInfo()
        calcinfo.codes_info = [codeinfo]
        calcinfo.retrieve_list = [self.options.output_filename]

        return calcinfo
