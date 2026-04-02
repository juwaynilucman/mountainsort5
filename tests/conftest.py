import pytest
from pathlib import Path

def pytest_addoption(parser):
    parser.addoption(
        "--base-dir", 
        action="store", 
        required=True,  # <--- No default, user MUST provide this
        help="Base directory for the dataset (Required)"
    )
    parser.addoption(
        "--chan-map", 
        action="store", 
        required=True,  # <--- No default, user MUST provide this
        help="Path to the channel map (Required)"
    )

@pytest.fixture(scope="session")
def test_config(request):
    """Fixture to expose the CLI arguments to your tests globally."""
    class Config:
        base_dir = Path(request.config.getoption("--base-dir"))
        chan_map = Path(request.config.getoption("--chan-map"))
        npx_bin = base_dir / "subset_data" / "raw_1pct.bin"
    
    return Config