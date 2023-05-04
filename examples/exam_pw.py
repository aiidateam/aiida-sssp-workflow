# from aiida.engine import run, submit
# from aiida import orm
# from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain

# inputs = {
#     "metadata": {"call_link_label": "SCF_for_cache"},
#     "pw": {
#         "structure": orm.load_node(369401),
#         "code": orm.load_node(1),
#         "pseudos": {
#             "H": orm.load_node(369335),
#         },
#         "parameters": orm.Dict(dict={
#             "CONTROL": {
#                 "calculation": "scf",
#             },
#             "ELECTRONS": {
#                 "conv_thr": 1e-05
#             },
#             "SYSTEM": {
#                 "degauss": 0.01,
#                 "ecutrho": 120,
#                 "ecutwfc": 30.1,
#                 "occupations": "smearing",
#                 "smearing": "cold"
#             }
#         }),
#         "parallelization": orm.Dict(dict={
#             "npool": 1,
#         }),
#         "metadata": {
#             "options": {
#                 "resources": {
#                     "num_machines": 1,
#                     "num_mpiprocs_per_machine": 2,
#                 },
#                 "max_wallclock_seconds": 1800,
#                 "withmpi": True,
#             },
#         },
#     },
#     "kpoints_distance": orm.Float(0.1),
# }

# # submit(PwBaseWorkChain, **inputs)
# run(PwBaseWorkChain, **inputs)

n = load_node(370105)
n._get_objects_to_hash()
