import io

from aiida.common import datastructures
from aiida import orm

def test_ld1_default(generate_inputs_ld1, fixture_sandbox, generate_calc_job, file_regression):
    """Test a default `sssp.pseudo.ld1`."""
    entry_point_name = 'sssp.pseudo.ld1'

    calc_info = generate_calc_job(fixture_sandbox, entry_point_name, generate_inputs_ld1())

    # Check the attributes of the returned `CalcInfo`
    assert isinstance(calc_info, datastructures.CalcInfo)
    assert isinstance(calc_info.codes_info[0], datastructures.CodeInfo)

    with fixture_sandbox.open('aiida.in') as handle:
        aiida_in = handle.read()

    # Checks on the files written to the sandbox folder as raw input
    file_regression.check(aiida_in, encoding='utf-8', extension='.in')