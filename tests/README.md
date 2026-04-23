# CyFi Testing Guide

## Mocking Strategy (Issue #71)

All tests use mocking to avoid hitting real APIs.

### What is mocked:
- search_planetary_computer - STAC API search
- _download_item_assets - Actual image downloads
- planetary_computer.sign_inplace - URL signing

### Running Tests

pytest tests/ -v
