import pytest

from aiida_sssp_workflow.utils import get_standard_structure


def test_get_standard_structure_default(data_regression):
    """Test get_standard_structure function."""
    with pytest.raises(ValueError):
        structure = get_standard_structure("Si", "delta")

    structure = get_standard_structure("Si", "delta", configuration="fcc")
    data_regression.check(structure.get_cif().get_content())


@pytest.mark.parametrize(
    "element, prop, configuration",
    [
        ("Fe", "delta", "gs"),
        ("Fe", "convergence", None),
        ("Te", "convergence", "xo"),
        ("Te", "delta", "xo"),
    ],
)
def test_get_standard_structure(element, prop, configuration, data_regression):
    """Test get_standard_structure function with different set of configuration."""
    structure = get_standard_structure(element, prop, configuration)

    data_regression.check(structure.get_cif().get_content())
