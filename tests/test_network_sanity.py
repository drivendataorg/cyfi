import pytest
from cyfi.data.satellite_data import catalog

@pytest.mark.network
def test_planetary_computer_connectivity():
    """
    Sanity check to ensure we can connect to the Planetary Computer STAC API.
    This test hits the actual network and is marked with the 'network' marker.
    """
    # Try to get the catalog title to verify connectivity
    assert catalog.title == "Microsoft Planetary Computer STAC API"

@pytest.mark.network
def test_stac_search():
    """
    Verify that we can perform a simple search on the STAC API.
    """
    # Search for a small area and recent date
    bbox = [-122.5, 37.5, -122.4, 37.6]
    date_range = "2023-01-01/2023-01-02"
    search = catalog.search(collections=["sentinel-2-l2a"], bbox=bbox, datetime=date_range)
    items = list(search.item_collection())
    # We don't necessarily expect items to always exist, but the search should not fail
    assert isinstance(items, list)
