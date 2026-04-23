"""Tests for satellite data functions using mocks (Issue #71)."""
import pytest
import pandas as pd
from unittest.mock import Mock, patch
from pathlib import Path

def test_search_planetary_computer_mocked(mock_planetary_computer):
    """Test searching Planetary Computer without real API."""
    from cyfi.data.satellite_data import search_planetary_computer
    
    # Call the function
    result = search_planetary_computer(
        latitude=41.424144,
        longitude=-73.206937,
        date="2023-07-01",
        collections=["sentinel-2-l2a"],
        days_search_window=30,
        meters_search_window=1000
    )
    
    assert result is not None
    # Verify mock was called
    mock_planetary_computer.search.assert_called_once()

def test_identify_satellite_data_mocked(mock_planetary_computer, mock_pc_sign):
    """Test satellite data identification with mocked API calls."""
    from cyfi.data.satellite_data import identify_satellite_data, search_planetary_computer
    
    # Create test samples
    samples = pd.DataFrame({
        'date': ['2023-07-01'],
        'latitude': [41.424144],
        'longitude': [-73.206937]
    }, index=['sample_001'])
    
    # Create mock config
    config = Mock()
    config.pc_days_search_window = 30
    config.pc_meters_search_window = 1000
    config.collections = ["sentinel-2-l2a"]
    config.n_sentinel_items = 1
    config.max_cloud_percent = 50
    config.satellite_meta_features = []
    config.use_sentinel_bands = ['B02', 'B03', 'B04']
    
    # Run function
    result = identify_satellite_data(samples, config)
    
    assert result is not None
    assert 'selected' in result.columns
    assert len(result) > 0

def test_get_item_metadata_mocked(mock_stac_search_response):
    """Test getting item metadata from search results."""
    from cyfi.data.satellite_data import get_items_metadata
    
    # Create mock search results
    mock_search = Mock()
    mock_item = Mock()
    mock_item.id = "test_item_001"
    mock_item.properties = {"datetime": "2023-07-01T15:48:19Z"}
    mock_item.to_dict.return_value = mock_stac_search_response['items'][0]
    mock_search.item_collection.return_value = [mock_item]
    
    result = get_items_metadata(mock_search)
    
    assert len(result) == 1
    assert result[0]['item_id'] == "test_item_001"

@pytest.mark.network
def test_search_planetary_computer_real():
    """Integration test that actually hits the API (optional)."""
    import pytest
    pytest.skip("Real API test - run with --run-network to enable")
    from cyfi.data.satellite_data import search_planetary_computer
    
    result = search_planetary_computer(
        latitude=41.424144,
        longitude=-73.206937,
        date="2023-07-01",
        collections=["sentinel-2-l2a"],
        days_search_window=30,
        meters_search_window=1000
    )
    
    assert result is not None
