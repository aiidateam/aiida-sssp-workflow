from aiida import orm
from aiida.plugins import CalculationFactory

def test_birch_murnaghan_fit():
    """Birch–Murnaghan fit test"""
    birch_murnaghan_fit = CalculationFactory('sssp_workflow.birch_murnaghan_fit')

    input = orm.Dict(dict={
        "energies": {
            "0": -155.35513331955,
            "1": -155.36820553211,
            "2": -155.37566424034,
            "3": -155.37807292397,
            "4": -155.37593723854,
            "5": -155.36971834897,
            "6": -155.35983306562
        },
        "energy_unit": "eV/atom",
        "volume_unit": "A^3/atom",
        "volumes": {
            "0": 19.219217135197,
            "1": 19.628136649358,
            "2": 20.037056161747,
            "3": 20.445975677205,
            "4": 20.854895189195,
            "5": 21.263814702423,
            "6": 21.672734216892
        }
    })

    res_node = birch_murnaghan_fit(input)
    res = res_node.get_dict()
    assert 'volume0' in res
    assert 'bulk_modulus0' in res
    assert 'bulk_deriv0' in res
    assert 'residuals0' in res
    assert res['volume0_unit'] == 'A^3/atom'
    assert res['bulk_modulus0_unit'] == 'GPa'

def test_calculate_delta():
    """test calcfunction calculate_delta"""
    calculate_delta = CalculationFactory('sssp_workflow.calculate_delta')

    inputs = {
        'element': orm.Str('Si'),
        'v0': orm.Float(20.4530),
        'b0': orm.Float(88.545),
        'bp': orm.Float(4.31),
    }   # Si line of the file so that the delta will be exactly zero
    res = calculate_delta(**inputs)
    assert res.value == 0.0