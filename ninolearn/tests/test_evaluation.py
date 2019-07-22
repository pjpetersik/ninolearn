import pytest
from ninolearn.learn import rmse

# =============================================================================
# Test
# =============================================================================
@pytest.mark.parametrize("test_input,expected", [("3+5", 8), ("2+4", 6), ("6*9", 54)])
def test_eval(test_input, expected):
    assert eval(test_input) == expected