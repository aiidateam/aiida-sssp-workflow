# -*- coding: utf-8 -*-
"""
Test of process functions
"""
import pytest

from aiida import orm
from aiida.plugins import CalculationFactory


def test_birch_murnaghan_fit(data_regression):
    """Birch–Murnaghan fit test"""
    birch_murnaghan_fit = CalculationFactory(
        'sssp_workflow.birch_murnaghan_fit')

    inputs = orm.Dict(
        dict={
            'energies': {
                '0': -155.35513331955,
                '1': -155.36820553211,
                '2': -155.37566424034,
                '3': -155.37807292397,
                '4': -155.37593723854,
                '5': -155.36971834897,
                '6': -155.35983306562
            },
            'energy_unit': 'eV/atom',
            'volume_unit': 'A^3/atom',
            'volumes': {
                '0': 19.219217135197,
                '1': 19.628136649358,
                '2': 20.037056161747,
                '3': 20.445975677205,
                '4': 20.854895189195,
                '5': 21.263814702423,
                '6': 21.672734216892
            }
        })

    output_bmf = birch_murnaghan_fit(inputs)
    assert isinstance(output_bmf, orm.Dict)
    data_regression.check(output_bmf.get_dict())


def test_calculate_delta():
    """test calcfunction calculate_delta"""
    calculate_delta = CalculationFactory('sssp_workflow.calculate_delta')

    inputs = {
        'element': orm.Str('Si'),
        'V0': orm.Float(20.4530),
        'B0': orm.Float(88.545),
        'B1': orm.Float(4.31),
    }  # Si line of the file so that the delta will be exactly zero
    res = calculate_delta(**inputs)
    assert res['delta'] == 0.0


def test_calculate_delta_H():  # pylint: disable=invalid-name
    """test calcfunction calculate_delta
    make sure space in string is striped."""
    calculate_delta = CalculationFactory('sssp_workflow.calculate_delta')

    inputs = {
        'element': orm.Str('H'),
        'V0': orm.Float(17.3883),
        'B0': orm.Float(10.284),
        'B1': orm.Float(2.71),
    }  # Si line of the file so that the delta will be exactly zero
    res = calculate_delta(**inputs)
    assert res['delta'] == 0.0


def test_calculate_delta_rare_earth():
    """test calcfunction calculate_delta"""
    calculate_delta = CalculationFactory('sssp_workflow.calculate_delta')

    inputs = {
        'element': orm.Str('La'),
        'V0': orm.Float(18.77799),
        'B0': orm.Float(122.037),
        'B1': orm.Float(4.461),
    }  # LaN line of the file so that the delta will be exactly zero
    res = calculate_delta(**inputs)
    assert res['delta'] == 0.0


def test_get_v0_b0_b1():
    """
    doc
    """
    from aiida_sssp_workflow.helpers import helper_get_v0_b0_b1

    V0, B0, B1 = helper_get_v0_b0_b1('Si')
    assert V0 == 20.4530
    assert B0 == 88.545
    assert B1 == 4.31

    V0, B0, B1 = helper_get_v0_b0_b1('La')
    assert V0 == 18.77799
    assert B0 == 122.037
    assert B1 == 4.461

    V0, B0, B1 = helper_get_v0_b0_b1('F')
    assert V0 == 19.3583
    assert B0 == 74.0411
    assert B1 == 4.1599


@pytest.mark.skip(reason='Tackle after all convergence wf fixed')
def test_get_volume_from_pressure_birch_murnaghan():
    """
    doc
    """
    from aiida_sssp_workflow.workflows.convergence.pressure import helper_get_volume_from_pressure_birch_murnaghan

    V0 = 20.4530
    B0 = 88.545
    B1 = 4.31
    P = 0.01

    ret = helper_get_volume_from_pressure_birch_murnaghan(P, V0, B0, B1)

    # check that very small residual pressure correspond a small volume change
    assert ret - 20.4530 < 0.1


@pytest.mark.skip(reason='Tackle after all convergence wf fixed')
def test_phonon_frequencies_diff():
    """test of helper_get_relative_phonon_frequencies"""
    from aiida_sssp_workflow.workflows.convergence.phonon_frequencies import helper_phonon_frequencies_difference

    input_parameters = {'dynamical_matrix_0': {'frequencies': [1., 1., 1.]}}
    ref_parameters = {'dynamical_matrix_0': {'frequencies': [1., 1., 1.]}}

    res = helper_phonon_frequencies_difference(orm.Dict(dict=input_parameters),
                                               orm.Dict(dict=ref_parameters))

    assert res['relative_diff'] == 0.0
    assert res['relative_max_diff'] == 0.0
    assert res['absolute_diff'] == 0.0
    assert res['absolute_max_diff'] == 0.0
