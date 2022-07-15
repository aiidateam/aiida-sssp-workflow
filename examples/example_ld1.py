import io
from aiida.engine import run_get_node
from aiida import orm 
from aiida.plugins import CalculationFactory

Ld1Calculation = CalculationFactory('sssp.pseudo.ld1')

from aiida_sssp_workflow.calculations.ld1 import Ld1Calculation

if __name__ == "__main__":
    inp_str = """ &input
   zed=3,
   rel=0,
   config='[He] 2s1 2p0',
   iswitch=3,
   dft='PBE'
 /
 &inputp
   lpaw=.false.,
   pseudotype=3,
   file_pseudopw='pseudo.upf',
   author='anonymous',
   lloc=-1,
   rcloc=0.6
   which_augfun='PSQ',
   rmatch_augfun_nc=.true.,
   tm=.true.
 /
4
1S  1  0  2.00  0.00  0.80  1.00  0.0
2S  2  0  1.00  0.00  0.80  1.00  0.0
2P  2  1  0.00  0.00  0.85  1.20  0.0
2P  2  1  0.00  1.00  0.85  1.20  0.0
"""

    input = {
        'code': orm.load_code("ld1@localhost"),
        'filename': orm.Str('pseudo.upf'),
        'parameters': orm.SinglefileData(file=io.BytesIO(inp_str.encode('utf-8'))),
        'metadata': {
            # 'dry_run': True,
            'options': {
                'resources': {
                    'num_machines': int(1)
                },
                'max_wallclock_seconds': int(60),
                'withmpi': False,
            },
        }
    }
    run_get_node(Ld1Calculation, **input)