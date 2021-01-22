"""
Example for functions for bands distance calculation
"""
from aiida import orm
from aiida.orm import load_node
from aiida.plugins import CalculationFactory
from aiida import load_profile
load_profile()

from aiida_sssp_workflow.calculations.calculate_bands_distance import calculate_eta_and_max_diff, \
    retrieve_bands, calculate_bands_distance

# calculate_bands_distance = CalculationFactory('sssp_workflow.calculate_bands_distance')

RY_TO_EV = 13.6056980659


def calc_band_distance(bandsdata_a, bandsdata_b, band_parameters_a,
                       band_parameters_b, is_metal):
    res = calculate_bands_distance(bandsdata_a, bandsdata_b,
                                   band_parameters_a, band_parameters_b,
                                   orm.Float(0.02 * RY_TO_EV),
                                   orm.Bool(is_metal))
    print(res.get_dict())


if __name__ == '__main__':
    # the bands of gold calculated with different pseudopotential
    bandsdata_a = load_node(11385)
    band_parameters_a = load_node(11387)
    bandsdata_b = load_node(14680)
    band_parameters_b = load_node(14682)

    # shift values are opposite to each other
    calc_band_distance(bandsdata_a, bandsdata_b, band_parameters_a,
                       band_parameters_b, True)
