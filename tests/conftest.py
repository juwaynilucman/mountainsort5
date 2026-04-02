import pytest
from pathlib import Path

def pytest_addoption(parser):
    parser.addoption(
        "--base-dir", 
        action="store", 
        required=True,
        help="Root directory for the experiment (e.g., .../marquees)"
    )
    parser.addoption(
        "--bin-rel-path", 
        action="store", 
        required=True,
        help="Relative path to the binary file (e.g., c46/subset_data/raw_1pct.bin)"
    )
    parser.addoption(
        "--chan-map-rel-path", 
        action="store", 
        required=True,
        help="Relative path to the channel map (e.g., chanMap.mat)"
    )

@pytest.fixture(scope="session")
def test_config(request):
    """Fixture to expose the CLI arguments to your tests globally."""
    class Config:
        base_dir = Path(request.config.getoption("--base-dir"))
        
        # Dynamically build the absolute paths
        bin_rel = request.config.getoption("--bin-rel-path")
        npx_bin = base_dir / bin_rel
        
        chan_map_rel = request.config.getoption("--chan-map-rel-path")
        chan_map = base_dir / chan_map_rel
        
    return Config