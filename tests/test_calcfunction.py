"""
Test of process functions
"""
from aiida import orm
from aiida.plugins import CalculationFactory


def test_birch_murnaghan_fit():
    """Birch–Murnaghan fit test"""
    birch_murnaghan_fit = CalculationFactory(
        'sssp_workflow.birch_murnaghan_fit')

    inputs = orm.Dict(
        dict={
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

    res = birch_murnaghan_fit(inputs)
    assert 'volume0' in res
    assert 'bulk_modulus0' in res
    assert 'bulk_deriv0' in res
    assert 'residuals0' in res
    assert res['volume0_unit'].value == 'A^3/atom'
    assert res['bulk_modulus0_unit'].value == 'GPa'


def test_calculate_delta():
    """test calcfunction calculate_delta"""
    calculate_delta = CalculationFactory('sssp_workflow.calculate_delta')

    inputs = {
        'element': orm.Str('Si'),
        'v0': orm.Float(20.4530),
        'b0': orm.Float(88.545),
        'bp': orm.Float(4.31),
    }  # Si line of the file so that the delta will be exactly zero
    res = calculate_delta(**inputs)
    assert res['delta'] == 0.0


def test_calculate_delta_H():  # pylint: disable=invalid-name
    """test calcfunction calculate_delta
    make sure space in string is striped."""
    calculate_delta = CalculationFactory('sssp_workflow.calculate_delta')

    inputs = {
        'element': orm.Str('H'),
        'v0': orm.Float(17.3883),
        'b0': orm.Float(10.284),
        'bp': orm.Float(2.71),
    }  # Si line of the file so that the delta will be exactly zero
    res = calculate_delta(**inputs)
    assert res['delta'] == 0.0


def test_calculate_delta_rare_earth():
    """test calcfunction calculate_delta"""
    calculate_delta = CalculationFactory('sssp_workflow.calculate_delta')

    inputs = {
        'element': orm.Str('La'),
        'v0': orm.Float(18.77799),
        'b0': orm.Float(122.037),
        'bp': orm.Float(4.461),
    }  # LaN line of the file so that the delta will be exactly zero
    res = calculate_delta(**inputs)
    assert res['delta'] == 0.0


def test_delta_volume():
    """test calcusation of delta volume from convergence tests"""
    calculate_delta_volume = CalculationFactory(
        'sssp_workflow.calculate_delta_volume')

    pressure_list = [
        -0.10267932770498, -0.083408565628544, 0.0089734082951343,
        0.021330232832696, 0.021330232832696, 0.014857610455878,
        0.021624442940734, 0.025449174345217
    ]
    inputs = {
        'equilibrium_refs':
        orm.Dict(dict={
            'V0': 20.4530,
            'B0': 88.545,
            'BP': 4.31,
        }),
        'pressures':
        orm.List(list=pressure_list),
        'pressure_reference':
        orm.Float(0.02545),
    }
    res = calculate_delta_volume(**inputs)
    print(res.get_list())
    # assert res.get_list()[-1] < 0.01
