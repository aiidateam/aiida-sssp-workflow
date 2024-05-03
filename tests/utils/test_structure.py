import pytest
from aiida.engine import run_get_node
from aiida import orm

from aiida_sssp_workflow.utils import get_default_configuration


@pytest.mark.parametrize(
    "element, property, configuration",
    [
        ("Fe", "band", "GS"),
        ("Fe", "convergence", "DC"),
        ("Te", "convergence", "DC"),
        ("Te", "band", "GS"),
        ("La", "band", "LAN"),
        ("La", "convergence", "DC"),
    ],
)
def test_get_default_configuration(element, property, configuration):
    """Test get_default_configuration function."""
    r, _ = run_get_node(get_default_configuration, element=element, property=property)

    assert isinstance(r, orm.Str)
    assert r.value == configuration
