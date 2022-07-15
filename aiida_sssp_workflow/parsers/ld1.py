import io

from aiida.parsers import Parser
from aiida import plugins

UpfData = plugins.DataFactory('pseudo.upf')

class Ld1Parser(Parser):
    """Parser for `Ld1Calculation` parse output to pseudo"""
    
    def parse(self, **kwargs):
        """Parse the contets of output of ld1 to pseudo files"""
        
        output_folder = self.retrieved
        
        with output_folder.open(self.node.inputs.filename.value, 'r') as handle:
            upf_content = handle.read()

            pseudo = UpfData.get_or_create(io.BytesIO(upf_content.encode('utf-8')))
            self.out('output_pseudo', pseudo.clone().store())
            